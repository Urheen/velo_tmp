import abc
import functools
import time
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import chex
import flax
import gin
import jax
import jax.numpy as jnp
from learned_optimization import checkpoints
from learned_optimization import profile
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import truncated_step as truncated_step_mod
import numpy as onp
from typing_extensions import Protocol
from learned_optimization.outer_trainers.gradient_learner import (
    SingleMachineState, PRNGKey,
    MetaInitializer, GradientEstimator, GradientLearner,
    gradient_worker_compute
)

class SingleMachineFineTuner:
  """Finetune a learned optimzier with gradient estimators on a single machine.

  This is a convience wrapper calling the multi-worker interface -- namley
  both `GradientLearner` and `gradient_worker_compute`.
  """

  def __init__(self,
               meta_init: MetaInitializer,
               gradient_estimators: Sequence[GradientEstimator],
               theta_opt: opt_base.Optimizer,
               init_theta_from_path: Optional[str] = None,
               num_steps: Optional[int] = None):
    """Initializer.

    Args:
      meta_init: Class containing an init function to construct outer params.
      gradient_estimators: Sequence of gradient estimators used to calculate
        gradients.
      theta_opt: The optimizer used to train the weights of the learned opt.
      num_steps: Number of meta-training steps used by optimizer for schedules.
    """
    self.gradient_learner = GradientLearner(
        meta_init, theta_opt, num_steps=num_steps, init_theta_from_path=init_theta_from_path)
    self.gradient_estimators = gradient_estimators

  def init(self, key: PRNGKey) -> SingleMachineState:
    """Initial state.

    This initializes the learned optimizer weights randomly, and set's up
    optimizer variables for these weights. Additionally the first state of the
    gradient estimators is also initialized.

    Args:
      key: jax rng

    Returns:
      The initial state
    """

    key1, key = jax.random.split(key)
    theta_state = self.gradient_learner.init(key1)
    worker_weights = self.gradient_learner.get_state_for_worker(theta_state)
    keys = jax.random.split(key, len(self.gradient_estimators))
    unroll_states = [
        grad_est.init_worker_state(worker_weights, key)
        for key, grad_est in zip(keys, self.gradient_estimators)
    ]

    return SingleMachineState(
        gradient_learner_state=theta_state,
        gradient_estimator_states=unroll_states)

  def update(
      self,
      state,
      key: PRNGKey,
      with_metrics: Optional[bool] = False
  ) -> Tuple[SingleMachineState, jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Perform one outer-update to train the learned optimizer.

    Args:
      state: State of this class
      key: jax rng
      with_metrics: To compute metrics or not

    Returns:
      state: The next state from this class
      loss: loss from the current iteration
      metrics: dictionary of metrics computed
    """
    key1, key2 = jax.random.split(key)
    worker_weights = self.gradient_learner.get_state_for_worker(
        state.gradient_learner_state)
    worker_compute_out = gradient_worker_compute(
        worker_weights,
        self.gradient_estimators,
        state.gradient_estimator_states,
        key=key1,
        with_metrics=with_metrics)

    next_gradient_estimator_states = worker_compute_out.unroll_states

    next_theta_state, metrics = self.gradient_learner.update(
        state.gradient_learner_state, [worker_compute_out.to_put],
        key=key2,
        with_metrics=with_metrics)

    metrics = summary.aggregate_metric_list(
        [worker_compute_out.metrics, metrics])

    return (SingleMachineState(
        gradient_learner_state=next_theta_state,
        gradient_estimator_states=next_gradient_estimator_states),
            worker_compute_out.to_put.mean_loss, metrics)

  def get_meta_params(self, state: SingleMachineState) -> lopt_base.MetaParams:
    """Get the weights of the learned optimizer."""
    return self.gradient_learner.get_meta_params(state.gradient_learner_state)

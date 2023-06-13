import functools
import jax
import jax.numpy as jnp
from learned_optimization import tree_utils
from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import truncated_step
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.tasks import base as tasks_base
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.outer_trainers.lopt_truncated_step import (
    init_truncation_state,
    init_truncation_state_vec_theta,
    progress_or_reset_inner_opt_state,
    TruncatedUnrollState,
    # truncated_unroll_one_step,
    # truncated_unroll_one_step_vec_theta,
    # vectorized_loss_and_aux,
    VectorizedLOptTruncatedStep
)
from typing import Optional, Any, Tuple
from learned_optimization import training
from utils.train_utils import set_to_zero, multi_transform
from flax.core.frozen_dict import freeze
from flax import traverse_util
PRNGKey = jnp.ndarray

def _f(opt, params):
    return multi_transform(
        {'trainable': opt, 'frozen': set_to_zero()}, 
        freeze(traverse_util.path_aware_map(lambda path, _: 'frozen' if 'linear_1' in path else 'trainable', params))
    )

def _truncated_unroll_one_step(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: TruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key,
    override_num_steps: Optional[int] = None,
) -> Tuple[TruncatedUnrollState, truncated_step.TruncatedUnrollOut]:
    """Train a given inner problem state a single step or reset it when done."""
    key1, key2 = jax.random.split(key)

    if override_num_steps is not None:
        num_steps = override_num_steps
    else:
        num_steps = state.truncation_state.length
    # only optimize a part of the parameters
    opt = learned_opt.opt_fn(theta)
    #print(opt)
    params = opt.get_params(state.inner_opt_state)
    # _opt = _f(opt, params)
    next_inner_opt_state, task_param, next_inner_step, l = progress_or_reset_inner_opt_state(
        task_family=task_family,
        opt=opt,
        num_steps=num_steps,
        key=key1,
        inner_opt_state=state.inner_opt_state,
        task_param=state.task_param,
        inner_step=state.inner_step,
        is_done=state.is_done,
        data=data,
        meta_loss_with_aux_key=meta_loss_with_aux_key,
    )

    next_truncation_state, is_done = trunc_sched.next_state(
        state.truncation_state, next_inner_step, key2, outer_state)

    # summaries
    opt = learned_opt.opt_fn(theta, is_training=True)
    # summary.summarize_inner_params(opt.get_params(next_inner_opt_state))
    output_state = TruncatedUnrollState(
        inner_opt_state=next_inner_opt_state,
        inner_step=next_inner_step,
        truncation_state=next_truncation_state,
        task_param=task_param,
        is_done=is_done,
    )

    out = truncated_step.TruncatedUnrollOut(
        is_done=is_done,
        loss=l,
        mask=(next_inner_step != 0),
        iteration=next_inner_step,
        task_param=state.task_param)

    return output_state, out

@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "trunc_sched",
                     "meta_loss_with_aux_key"))
@functools.partial(
    jax.vmap, in_axes=(None, None, None, None, 0, 0, 0, None, None, None))
def truncated_unroll_one_step(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: TruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key: Optional[str],
    override_num_steps: Optional[int],
) -> Tuple[TruncatedUnrollState, truncated_step.TruncatedUnrollOut]:
    """Perform one step of inner training without vectorized theta."""
    return _truncated_unroll_one_step(
        task_family=task_family,
        learned_opt=learned_opt,
        trunc_sched=trunc_sched,
        theta=theta,
        key=key,
        state=state,
        data=data,
        outer_state=outer_state,
        meta_loss_with_aux_key=meta_loss_with_aux_key,
        override_num_steps=override_num_steps)


@functools.partial(
    jax.jit,
    static_argnames=("task_family", "learned_opt", "trunc_sched",
                     "meta_loss_with_aux_key"))
@functools.partial(
    jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, None, None, None))
def truncated_unroll_one_step_vec_theta(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    trunc_sched: truncation_schedule.TruncationSchedule,
    theta: lopt_base.MetaParams,
    key: PRNGKey,
    state: TruncatedUnrollState,
    data: Any,
    outer_state: Any,
    meta_loss_with_aux_key: Optional[str],
    override_num_steps: Optional[int],
) -> Tuple[TruncatedUnrollState, truncated_step.TruncatedUnrollOut]:
    """Perform one step of inner training with vectorized theta."""
    return _truncated_unroll_one_step(
        task_family=task_family,
        learned_opt=learned_opt,
        trunc_sched=trunc_sched,
        theta=theta,
        key=key,
        state=state,
        data=data,
        outer_state=outer_state,
        meta_loss_with_aux_key=meta_loss_with_aux_key,
        override_num_steps=override_num_steps)


@functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0))
def partial_vectorized_loss_and_aux(
    task_family: tasks_base.TaskFamily,
    learned_opt: lopt_base.LearnedOptimizer,
    theta: lopt_base.MetaParams, 
    inner_opt_state: Any,
    task_param: Any, 
    key: PRNGKey,
    data: Any) -> jnp.ndarray:
    """Vectorized computation of the task loss given data."""
    # TODO(lmetz) make use of eval task families?
    task = task_family.task_fn(task_param)
    opt = learned_opt.opt_fn(theta, is_training=True)
    # training a part of parameter with multi transform
    p, s = opt.get_params_state(inner_opt_state)
    l, _, aux = task.loss_with_state_and_aux(p, s, key, data)
    return l, aux


class PartialVectorizedLOptTruncatedStep(
    VectorizedLOptTruncatedStep
    # truncated_step.VectorizedTruncatedStep,
    # full_es.OverrideStepVectorizedTruncatedStep
):
    """PartialVectorizedTruncatedStep for learned optimizer inner training on a part of parameters of the model.

    This is more fully featured than VectorizedLOptTruncated step allowing for
    both task_family (rather than a single task), and truncation schedules.
    """

    def __init__(
        self,
        task_family: tasks_base.TaskFamily,
        learned_opt: lopt_base.LearnedOptimizer,
        trunc_sched: truncation_schedule.TruncationSchedule,
        num_tasks: int,
        meta_loss_split: Optional[str] = None,
        random_initial_iteration_offset: int = 0,
        outer_data_split="train",
        meta_loss_with_aux_key: Optional[str] = None,
        task_name: Optional[str] = None,
    ):
        """Initializer.

        Args:
        task_family: task family to do unrolls on.
        learned_opt: learned optimizer instance.
        trunc_sched: truncation schedule to use.
        num_tasks: number of tasks to vmap over.
        meta_loss_split: This can take 3 values: None, 'same_data', or a
            dataset split: {"train", "outer_valid", "inner_valid", "test"}.
            If set to a dataset split we use a new batch of data to compute the
            meta-loss which is evaluated on the newly created inner state (after
            applying the lopt.). If set to 'same_data', the same data is reused to
            evaluate the meta-loss. If None no additional computation is performed
            and the previous state's loss evaluated on the training batch is used.
        random_initial_iteration_offset: An initial offset for the inner-steps of
            each task. This is to prevent all tasks running in lockstep. This should
            be set to the max number of steps the truncation schedule.
        outer_data_split: Split of data to use when computing meta-losses.
        meta_loss_with_aux_key: Instead of using the loss, use a value from the
            returned auxiliary data.
        task_name: Optional string used to prefix summary.
            If not set, the name of the task family is used.
        """
        self.task_family = task_family
        self.learned_opt = learned_opt
        self.trunc_sched = trunc_sched
        self.num_tasks = num_tasks
        self.meta_loss_split = meta_loss_split
        self.random_initial_iteration_offset = random_initial_iteration_offset
        self.outer_data_split = outer_data_split
        self.meta_loss_with_aux_key = meta_loss_with_aux_key
        self._task_name = task_name

        self.data_shape = jax.tree_util.tree_map(
            lambda x: jax.ShapedArray(shape=x.shape, dtype=x.dtype),
            training.vec_get_batch(
                task_family, num_tasks, split="train", numpy=True))

    def init_step_state(self,
                      theta,
                      outer_state,
                      key,
                      theta_is_vector=False,
                      num_steps_override=None):
        if theta_is_vector:
            init_fn = init_truncation_state_vec_theta
        else:
            init_fn = init_truncation_state

        key1, key2 = jax.random.split(key)
        unroll_state = init_fn(self.task_family, self.learned_opt, self.trunc_sched,
                            theta, outer_state,
                            jax.random.split(key1,
                                                self.num_tasks), num_steps_override)
        # When initializing, we want to keep the trajectories not all in sync.
        # To do this, we can initialize with a random offset on the inner-step.
        if self.random_initial_iteration_offset:
            inner_step = jax.random.randint(
                key2,
                unroll_state.inner_step.shape,
                0,
                self.random_initial_iteration_offset,
                dtype=unroll_state.inner_step.dtype)
        unroll_state = unroll_state.replace(inner_step=inner_step)

        return unroll_state

    def get_batch(self, steps: Optional[int] = None):
        if steps is not None:
            data_shape = (steps, self.num_tasks)
        else:
            data_shape = (self.num_tasks,)
        tr_batch = training.get_batches(
            self.task_family,
            data_shape,
            numpy=True,
            split="train")

        if self.meta_loss_split == "same_data" or self.meta_loss_split is None:
            return tr_batch
        else:
            outer_batch = training.get_batches(self.task_family, data_shape, numpy=True, split=self.meta_loss_split)
        return (tr_batch, outer_batch)

    def get_outer_batch(self, steps: Optional[int] = None):
        if steps is not None:
            data_shape = (steps, self.num_tasks)
        else:
            data_shape = (self.num_tasks,)
        return training.get_batches(
            self.task_family, data_shape, numpy=True, split=self.outer_data_split)

    def unroll_step(self,
                    theta,
                    unroll_state,
                    key,
                    data,
                    outer_state,
                    theta_is_vector=False,
                    override_num_steps: Optional[int] = None):
        # per-step data changes depending on if we use a extra eval batch per step.
        if self.meta_loss_split == "same_data":
            # use same batch of data
            tr_data = data
            meta_data = data
        elif self.meta_loss_split is None:
            tr_data = data
            meta_data = None
        else:
            # Otherwise assume we passed a valid data split.
            tr_data, meta_data = data

        key1, key2 = jax.random.split(key)

        # This function is designed to be called with the unroll_state having the
        # same number of tasks as created initially. One can, however, call it with
        # with a bigger batchsize representing 2 perturbations stacked together.
        # When doing this, we want to share randomness across these 2 batches
        # as they are antithetic samples.
        # TODO(lmetz) consider passing stack_antithetic_samples in some capacity
        # rather than guessing it here.
        num_tasks_in_state = tree_utils.first_dim(unroll_state)
        if num_tasks_in_state == self.num_tasks * 2:
            stack_antithetic_samples = True
        else:
            stack_antithetic_samples = False

        # If stacking the antithetic samples, we want to share random keys across
        # the antithetic samples.
        vec_keys = jax.random.split(key1, self.num_tasks)
        if stack_antithetic_samples:
            vec_keys = jax.tree_util.tree_map(
                lambda a: jnp.concatenate([a, a], axis=0), vec_keys)

        fn = truncated_unroll_one_step_vec_theta if theta_is_vector else truncated_unroll_one_step
        next_unroll_state_, ys = fn(self.task_family, self.learned_opt,
                                    self.trunc_sched, theta, vec_keys, unroll_state,
                                    tr_data, outer_state,
                                    self.meta_loss_with_aux_key, override_num_steps)

        # Should we evaluate resulting state on potentially new data?
        if meta_data is not None:
            vec_keys = jax.random.split(key2, self.num_tasks)
            if stack_antithetic_samples:
                vec_keys = jax.tree_util.tree_map(lambda a: jnp.concatenate([a, a], axis=0), vec_keys)
            loss, aux = partial_vectorized_loss_and_aux(
                self.task_family, self.learned_opt,
                theta,
                next_unroll_state_.inner_opt_state,
                next_unroll_state_.task_param,
                vec_keys, meta_data
            )
            if self.meta_loss_with_aux_key:
                ys = ys.replace(loss=aux[self.meta_loss_with_aux_key])
            else:
                ys = ys.replace(loss=loss)

        @jax.vmap
        def norm(loss, task_param):
            return self.task_family.task_fn(task_param).normalizer(loss)

        ys = ys.replace(loss=norm(ys.loss, unroll_state.task_param))

        return next_unroll_state_, ys

    def meta_loss_batch(self,
                        theta: Any,
                        unroll_state: Any,
                        key: Any,
                        data: Any,
                        outer_state: Any,
                        theta_is_vector: bool = False):
        keys = jax.random.split(key, self.num_tasks)
        loss, aux_metrics = partial_vectorized_loss_and_aux(self.task_family,
                                                    self.learned_opt, theta,
                                                    unroll_state.inner_opt_state,
                                                    unroll_state.task_param, keys,
                                                    data)

        if self.meta_loss_with_aux_key:
            return aux_metrics[self.meta_loss_with_aux_key]
        else:
            @jax.vmap
            def norm(loss, task_param):
                return self.task_family.task_fn(task_param).normalizer(loss)

            # Then normalize the losses to a sane meta-training range.
            loss = norm(loss, unroll_state.task_param)

        return loss
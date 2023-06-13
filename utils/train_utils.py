import jax
import json
import jax.numpy as jnp  # JAX NumPy

from flax import linen as nn  # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np  # Ordinary NumPy
import os
import optax  # Optimizers
import tensorflow_datasets as tfds  # TFDS for MNIST
import tensorflow as tf

from typing import Any, Callable, Dict, Iterator, Tuple, List, Optional
from tqdm import tqdm

from flax import struct, core
from clu import metrics
from typing import Callable, Any
import optax
from optax import GradientTransformation, EmptyState

import yaml


def read_yaml(path):
    with open(path, encoding='utf-8') as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data


class DataBaseObj(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val


class TrainState(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

    Synopsis::

        state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
        grad_fn = jax.grad(make_loss_fn(state.apply_fn))
        for batch in data:
            grads = grad_fn(state.params, batch)
            state = state.apply_gradients(grads=grads)

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Args:
        step: Counter starts at 0 and is incremented by every call to
        `.apply_gradients()`.
        apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
        convenience to have a shorter params list for the `train_step()` function
        in your training loop.
        params: The parameters to be updated by `tx` and used by `apply_fn`.
        tx: An Optax gradient transformation.
        opt_state: The state for `tx`.
    """
    step: int
    metrics: metrics.Collection
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    def apply_gradients(self, *, grads, is_velo, loss, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
        grads: Gradients that have the same pytree structure as `.params`.
        **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
        An updated instance of `self` with `step` incremented by one, `params`
        and `opt_state` updated by applying `grads`, and additional attributes
        replaced as specified by `kwargs`.
        """
        if is_velo:
            updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, extra_args={"loss": loss})
        else:
            updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
from optax import MultiTransformState, GradientTransformation
from optax._src import wrappers    
from jax.tree_util import tree_map

def masked(
    inner,
    mask
):
  """Mask updates so only some are transformed, the rest are passed through.

  For example, it is common to skip weight decay for BatchNorm scale and all
  bias parameters. In many networks, these are the only parameters with only
  one dimension. So, you may create a mask function to mask these out as
  follows::

    mask_fn = lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask_fn)

  You may alternatively create the mask pytree upfront::

    mask = jax.tree_util.tree_map(lambda x: x.ndim != 1, params)
    weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask)

  For the ``inner`` transform, state will only be stored for the parameters that
  have a mask value of ``True``.

  Args:
    inner: Inner transformation to mask.
    mask: a PyTree with same structure as (or a prefix of) the params PyTree, or
      a Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, ``True`` for leaves/subtrees you want to apply the
      transformation to, and ``False`` for those you want to skip. The mask must
      be static for the gradient transformation to be jit-compilable.

  Returns:
    New GradientTransformation wrapping ``inner``.
  """
  def mask_pytree(pytree, mask_tree):
    return tree_map(lambda p, m: p if m else wrappers.MaskedNode(), pytree, mask_tree)

  def init_fn(params):
    mask_tree = mask(params) if callable(mask) else mask
    masked_params = mask_pytree(params, mask_tree)
    return wrappers.MaskedState(inner_state=inner.init(masked_params))

  def update_fn(updates, state, params=None, extra_args=None):
    mask_tree = mask(updates) if callable(mask) else mask
    masked_updates = mask_pytree(updates, mask_tree)
    masked_params = None if params is None else mask_pytree(params, mask_tree)
    try:
      new_masked_updates, new_inner_state = inner.update(
          masked_updates, state.inner_state, masked_params, extra_args=extra_args)
    except:
      new_masked_updates, new_inner_state = inner.update(
          masked_updates, state.inner_state, masked_params)

    new_updates = tree_map(
        lambda m, new_u, old_u: new_u if m else old_u,
        mask_tree, new_masked_updates, updates)
    return new_updates, wrappers.MaskedState(inner_state=new_inner_state)

  return GradientTransformation(init_fn, update_fn)

def set_to_zero() -> GradientTransformation:
  """Stateless transformation that maps input gradients to zero.

  The resulting update function, when called, will return a tree of zeros
  matching the shape of the input gradients. This means that when the updates
  returned from this transformation are applied to the model parameters, the
  model parameters will remain unchanged.

  This can be used in combination with `multi_transform` or `masked` to freeze
  (i.e. keep fixed) some parts of the tree of model parameters while applying
  gradient updates to other parts of the tree.

  When updates are set to zero inside the same jit-compiled function as the
  calculation of gradients, optax transformations, and application of updates to
  parameters, unnecessary computations will in general be dropped.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return EmptyState()

  def update_fn(updates, state, params=None, extra_args=None):
    del params  # Unused by the zero transform.
    return jax.tree_util.tree_map(jnp.zeros_like, updates), state

  return GradientTransformation(init_fn, update_fn)



def multi_transform(
    transforms,
    param_labels
):
  """Partitions params and applies a different transformation to each subset.

  Below is an example where we apply Adam to the weights and SGD to the biases
  of a 2-layer neural network::

    import optax
    import jax
    import jax.numpy as jnp

    def map_nested_fn(fn):
      '''Recursively apply `fn` to the key-value pairs of a nested dict'''
      def map_fn(nested_dict):
        return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                for k, v in nested_dict.items()}
      return map_fn

    params = {'linear_1': {'w': jnp.zeros((5, 6)), 'b': jnp.zeros(5)},
              'linear_2': {'w': jnp.zeros((6, 1)), 'b': jnp.zeros(1)}}
    gradients = jax.tree_util.tree_map(jnp.ones_like, params)  # dummy gradients

    label_fn = map_nested_fn(lambda k, _: k)
    tx = optax.multi_transform({'w': optax.adam(1.0), 'b': optax.sgd(1.0)},
                               label_fn)
    state = tx.init(params)
    updates, new_state = tx.update(gradients, state, params)
    new_params = optax.apply_updates(params, updates)

  Instead of providing a ``label_fn``, you may provide a PyTree of labels
  directly.  Also, this PyTree may be a prefix of the parameters PyTree. This
  is demonstrated in the GAN pseudocode below::

    generator_params = ...
    discriminator_params = ...
    all_params = (generator_params, discriminator_params)
    param_labels = ('generator', 'discriminator')

    tx = optax.multi_transform(
        {'generator': optax.adam(0.1), 'discriminator': optax.adam(0.5)},
        param_labels)

  If you would like to not optimize some parameters, you may wrap
  ``optax.multi_transform`` with :func:`optax.masked`.

  Args:
    transforms: A mapping from labels to transformations. Each transformation
      will be only be applied to parameters with the same label.
    param_labels: A PyTree that is the same shape or a prefix of the
      parameters/updates (or a function that returns one given the parameters as
      input). The leaves of this PyTree correspond to the keys of the transforms
      (therefore the values at the leaves must be a subset of the keys).

  Returns:
    An ``optax.GradientTransformation``.
  """
  def make_mask(labels, group):
    return jax.tree_util.tree_map(lambda label: label == group, labels)

  def init_fn(params):
    labels = param_labels(params) if callable(param_labels) else param_labels
    label_set = set(jax.tree_util.tree_leaves(labels))
    if not label_set.issubset(transforms.keys()):
      raise ValueError('Some parameters have no corresponding transformation.\n'
                       f'Parameter labels: {list(sorted(label_set))} \n'
                       f'Transforms keys: {list(sorted(transforms.keys()))} \n')

    # print(transforms)
  
    inner_states = {
        group: masked(tx, make_mask(labels, group)).init(params)
        for group, tx in transforms.items()
    }
    return MultiTransformState(inner_states)

  def update_fn(updates, state, params=None, extra_args=None):
    labels = param_labels(updates) if callable(param_labels) else param_labels
    new_inner_state = {}
    for group, tx in transforms.items():
      masked_tx = masked(tx, make_mask(labels, group))
      updates, new_inner_state[group] = masked_tx.update(
          updates, state.inner_states[group], params, extra_args)
    return updates, MultiTransformState(new_inner_state)

  return GradientTransformation(init_fn, update_fn)


def create_learning_rate_fn(
        num_train_stpes, learning_rate, num_warmup_steps
):
    """Create learning rate function"""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=num_warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0.0,
        transition_steps=num_train_stpes - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


def create_optimizer(cfg):
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        num_train_stpes=cfg.num_train_steps, learning_rate=cfg.lr,
        num_warmup_steps=cfg.num_warmup_steps
    )
    if cfg.optim == 'sgd':
        return optax.sgd(learning_rate=linear_decay_lr_schedule_fn)
    elif cfg.optim == 'adamw':
        return optax.adamw(learning_rate=linear_decay_lr_schedule_fn)
    elif cfg.optim == 'adam':
        return optax.adam(learning_rate=linear_decay_lr_schedule_fn)
    elif cfg.optim == 'velo':
        from learned_optimization.research.general_lopt import prefab
        return prefab.optax_lopt(cfg.num_train_steps)
    else:
        raise NotImplementedError


from typing import Any, Optional

import flax
import gin
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
from flax import traverse_util
from flax.core import freeze
from utils.mytree_utils import DictKey, tree_map_with_path

PRNGKey = jnp.ndarray


def _second_moment_normalizer(x, axis, eps=1e-5):
  return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def _tanh_embedding(iterations):
  f32 = jnp.float32

  def one_freq(timescale):
    return jnp.tanh(iterations / (f32(timescale)) - 1.0)

  timescales = jnp.asarray(
      [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
      dtype=jnp.float32)
  return jax.vmap(one_freq)(timescales)


@flax.struct.dataclass
class MLPLOptState:
  params: Any
  rolling_features: common.MomAccumulator
  iteration: jnp.ndarray
  state: Any

@flax.struct.dataclass
class _EmptyNode:
  pass

empty_node = _EmptyNode()

def flatten_dict(xs, keep_empty_nodes=False, is_leaf=None, sep=None):
  assert isinstance(xs, (flax.core.FrozenDict, dict)), f'expected (frozen)dict; got {type(xs)}'

  def _key(path):
    if sep is None:
      return path
    return sep.join(path)

  def _flatten(xs, prefix):
    if not isinstance(xs, (flax.core.FrozenDict, dict)) or (
        is_leaf and is_leaf(prefix, xs)):
      return {_key(prefix): _key(prefix)}
    result = {}
    is_empty = True
    for key, value in xs.items():
      is_empty = False
      path = prefix + (key,)
      result.update(_flatten(value, path))
    if keep_empty_nodes and is_empty:
      if prefix == ():  # when the whole input is empty
        return {}
      return {_key(prefix): empty_node}
    return result
  return freeze(_flatten(xs, ()))

def tree_keys(tree):
    from jax.tree_util import tree_flatten
    leaves, treedef = tree_flatten(tree, None)
    return treedef.unflatten(flatten_dict(tree, sep='/'))

class PartialMLPLOpt(lopt_base.LearnedOptimizer):
  """Learned optimizer leveraging a per parameter MLP.

  This is also known as LOLv2.
  """

  def __init__(self,
               update_params,
               exp_mult=0.001,
               step_mult=0.001,
               hidden_size=32,
               hidden_layers=2,
               compute_summary=True):

    super().__init__()
    self.update_params = update_params
    self._step_mult = step_mult
    self._exp_mult = exp_mult
    self._compute_summary = compute_summary

    def ff_mod(inp):
      return hk.nets.MLP([hidden_size] * hidden_layers + [2])(inp)

    self._mod = hk.without_apply_rng(hk.transform(ff_mod))

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    # There are 19 features used as input. For now, hard code this.
    return self._mod.init(key, jnp.zeros([0, 19]))

  def opt_fn(self,
             theta: lopt_base.MetaParams,
             is_training: bool = False) -> opt_base.Optimizer:
    decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])

    mod = self._mod
    exp_mult = self._exp_mult
    step_mult = self._step_mult
    compute_summary = self._compute_summary
    parent=self

    class _Opt(opt_base.Optimizer):
      """Optimizer instance which has captured the meta-params (theta)."""

      def init(self,
               params: lopt_base.Params,
               model_state: Any = None,
               num_steps: Optional[int] = None,
               key: Optional[PRNGKey] = None) -> MLPLOptState:
        """Initialize inner opt state."""

        return MLPLOptState(
            params=params,
            state=model_state,
            rolling_features=common.vec_rolling_mom(decays).init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32))

      def update(self,
                 opt_state: MLPLOptState,
                 grad: Any,
                 loss: float,
                 model_state: Any = None,
                 is_valid: bool = False,
                 key: Optional[PRNGKey] = None) -> MLPLOptState:

        next_rolling_features = common.vec_rolling_mom(decays).update(
            opt_state.rolling_features, grad)

        training_step_feature = _tanh_embedding(opt_state.iteration)

        def _update_tensor(
            path, 
            p, g, m):
          path = '/'.join([p.key for p in path])
          update = False
          for up in parent.update_params:
            if up in path:
              update = True
          if not update:
            print(f"not updating {path}")
            return g
          else:
            print(f"updateing {path}")
          # this doesn't work with scalar parameters, so let's reshape.
          if not p.shape:
            p = jnp.expand_dims(p, 0)
            g = jnp.expand_dims(g, 0)
            m = jnp.expand_dims(m, 0)
            did_reshape = True
          else:
            did_reshape = False

          inps = []

          # feature consisting of raw gradient values
          batch_g = jnp.expand_dims(g, axis=-1)
          inps.append(batch_g)

          # feature consisting of raw parameter values
          batch_p = jnp.expand_dims(p, axis=-1)
          inps.append(batch_p)

          # feature consisting of all momentum values
          inps.append(m)

          inp_stack = jnp.concatenate(inps, axis=-1)
          axis = list(range(len(p.shape)))

          inp_stack = _second_moment_normalizer(inp_stack, axis=axis)

          # once normalized, add features that are constant across tensor.
          # namly the training step embedding.
          stacked = jnp.reshape(training_step_feature, [1] * len(axis) +
                                list(training_step_feature.shape[-1:]))
          stacked = jnp.tile(stacked, list(p.shape) + [1])

          inp = jnp.concatenate([inp_stack, stacked], axis=-1)

          # apply the per parameter MLP.
          output = mod.apply(theta, inp)

          # split the 2 outputs up into a direction and a magnitude
          direction = output[..., 0]
          magnitude = output[..., 1]

          # compute the step
          step = direction * jnp.exp(magnitude * exp_mult) * step_mult
          step = step.reshape(p.shape)
          new_p = p - step
          if did_reshape:
            new_p = jnp.squeeze(new_p, 0)

          if compute_summary:
            for fi, f in enumerate(inp):
              summary.summary(f"mlp_lopt/inp{fi}/mean_abs",
                              jnp.mean(jnp.abs(f)))

            avg_step_size = jnp.mean(jnp.abs(step))
            summary.summary("mlp_lopt/avg_step_size", avg_step_size)

            summary.summary(
                "mlp_lopt/avg_step_size_hist",
                avg_step_size,
                aggregation="collect")

            summary.summary("mlp_lopt/direction/mean_abs",
                            jnp.mean(jnp.abs(direction)))
            summary.summary("mlp_lopt/magnitude/mean_abs",
                            jnp.mean(jnp.abs(magnitude)))
            summary.summary("mlp_lopt/magnitude/mean", jnp.mean(magnitude))

            summary.summary("mlp_lopt/grad/mean_abs", jnp.mean(jnp.abs(g)))

          return new_p
        # print(tree_keys(opt_state.params))
        from utils.mytree_utils import tree_map
        #next_params = tree_map(
        #  _update_tensor, 
        #  opt_state.params,
        #  grad, 
        #  next_rolling_features.m,
         # tree_keys(opt_state.params)
        #)
        next_params = tree_map_with_path(          
          _update_tensor, 
          opt_state.params,
          grad, 
          next_rolling_features.m
        )
        next_opt_state = MLPLOptState(
            params=tree_utils.match_type(next_params, opt_state.params),
            rolling_features=tree_utils.match_type(next_rolling_features,
                                                   opt_state.rolling_features),
            iteration=opt_state.iteration + 1,
            state=model_state)
        return next_opt_state

    return _Opt()
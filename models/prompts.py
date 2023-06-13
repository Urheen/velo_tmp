import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Optional, Sequence, Tuple, Protocol
# from flaxformer.types import Array
# from flaxformer.types import DType
from flax.linen import partitioning

Array = jnp.ndarray
DType = jnp.dtype
Initializer = Callable[[Array, Sequence[int]], Array]


class _Prompt(nn.Module):
  """A module that produces a learnable prompt.
  Attributes:
    length: The length of the prompt, P.
    prompt_init: An initializer function for the variable.
    axis_names: Logical names for the parameter axes. Note: We use
      "prompt_embed" as the second dimension so that the prompt is always
      replicated, even when using 2-way parameter partitioning when the "embed"
      dimension would get partitioned. This makes it possible to save the prompt
      as a numpy file. If the prompt needs to be partitioned, one can change the
      second dimension to "embed", but the prompt variable will need to be
      managed by the t5x checkpointing utilities (i.e. the numpy checkpoint will
      not be the full prompt and you will need to save multiple t5x checkpoints)
      and `prompt_tuning.scripts.extract_variable` to create a numpy checkpoint.
    dtype: The dtype of the activations for this module.
  """
  length: int
  prompt_init: Initializer = nn.initializers.uniform()
  axis_names: Tuple[str, str] = ("prompt", "prompt_embed")
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x, x_embed):
    """Get the prompt variable.
    Args:
      x: [B, T] The sequence of input tokens.
      x_embed: [B, T, H] The sequence of embedded input tokens.
    Returns:
      The prompt variable. [P, H]
    """
    embed_dim = x_embed.shape[-1]
    prompt = partitioning.param_with_axes(
        "prompt",
        self.prompt_init,
        (self.length, embed_dim),
        axes=self.axis_names)
    prompt = prompt.astype(self.dtype)
    return prompt
  
def expand_to_batch(x, y):
  """Expand unbatched `x` to the same batch size as `y`."""
  batch_size = y.shape[0]
  return jnp.tile(jnp.expand_dims(x, axis=0), [batch_size] + [1 for _ in x.shape])

class CombinationFn(Protocol):
  """Combine a prompt and the embedded input."""

  def __call__(self, prompt, x_embed, x):
    """Combine the `prompt` and the embedded input (`x_embed`).
    Note:
      x (the integer values of the input) is not always required but we it as
      a parameter for all combination functions so we can easily swap them out.
    Args:
      prompt: The prompt variable.
      x_embed: The embedded input.
      x: The integer tokens for the input, this is required for some
        combinations such as adding the prompt after the input.
    Returns:
      The embedded input with the prompt added to it.
    """
    pass


def prefix_prompt(prompt, x_embed, x):
  """Concatenate `prompt` to the beginning of `x_embed`.
  Args:
    prompt: [B, P, H] The prompt.
    x_embed: [B, T, H] The embedded input.
    x: [B, T] The non-embedded input, used for finding the lengths of examples.
  Returns:
    The input with the prompt concatenated to the front. [B, P + T, H]
  """
  del x
  return jnp.concatenate([prompt, x_embed], axis=1)
  

class Prompt(nn.Module):
  """Generate a Prompt and concatenate it with the input.
  This is the training time version of prompting a model. Calling the injected
  `prompt` module will generate your unbatched prompt. This model then
  replicates it for the batched input and concatenates them together.
  Attributes:
    prompt: The module that actually generates the unbatched prompt.
    combine: A function that combines the prompt and the embedded input.
  """
  prompt_length: int 
  combine: CombinationFn = prefix_prompt

  def setup(self):
     self.prompt = _Prompt(length=self.prompt_length)

  def __call__(self, x, x_embed):
    prompt = self.prompt(x, x_embed)
    prompt = expand_to_batch(prompt, x_embed)
    # TODO: Create a minimum reproducible example and bug for the
    # pytype error when calling a function bound to the class attribute.
    #
    # Pytype is throwing a false positive here, it probably thinks
    # `self.combine` is a method call that is giving a `self` parameter but it
    # is actually just a function so there are only 2 arguments, like the type
    # annotation says.
    return self.combine(prompt, x_embed, x)  # pylint: disable=too-many-function-args
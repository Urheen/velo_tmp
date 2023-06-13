import gin
import haiku as hk
import jax
import chex
import jax.numpy as jnp
from models.prompt_t5 import FlaxPromptT5ForConditionalGeneration
from .crossfit_partitions import SOURCES
from learned_optimization.tasks import base
from learned_optimization.tasks.base import Batch, ModelState, PRNGKey, Params, Task, TaskCfg
from learned_optimization.tasks.datasets import base as datasets_base
from transformers import T5Tokenizer
from optax import softmax_cross_entropy_with_integer_labels
from learned_optimization.time_filter import model_paths
from learned_optimization.time_filter import time_model
from learned_optimization.tasks.parametric import cfgobject
from learned_optimization import profile
from learned_optimization.tasks.parametric import parametric_utils


@gin.configurable
class ParametricCrossfitPT5(base.TaskFamily):

    def __init__(
        self,
        datasets: datasets_base.Datasets,
        model_name
        ) -> None:
        super().__init__()
        self.datasets = datasets
        self.model_name = model_name

        self.model = FlaxPromptT5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def sample(self, key: chex.PRNGKey) -> cfgobject.CFGNamed:
        return cfgobject.CFGNamed("ParametricCrossfitPT5", {})

    def task_fn(self, cfg: TaskCfg) -> base.Task:
        
        parent = self

        class _Task(base.Task):
            "Construct a task sample"

            def __init__(self) -> None:
                super().__init__()
                self.datasets = parent.datasets
                self.model = parent.model
                self.tokenizer = parent.tokenizer

            def init(self, key: PRNGKey) -> Params:
                return self.model.params
            
            def loss(
                self, 
                params: Params, 
                key: PRNGKey, 
                batch: Batch
            ):
                """
                batch["input_ids"], batch["attention_mask"] = trim_batch(
                    input_ids=batch["input_ids"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    attention_mask=batch["attention_mask"],
                )

                batch["decoder_input_ids"], batch["decoder_attention_mask"] = trim_batch(
                    input_ids=batch["decoder_input_ids"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    attention_mask=batch["decoder_attention_mask"],
                )

                """

                t5_output = self.model.__call__(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']  ,
                    decoder_input_ids=batch['decoder_input_ids'],
                    decoder_attention_mask=batch['decoder_attention_mask'],
                    params=params,
                    dropout_rng=key,
                    train=True
                )
                lm_logits = t5_output.logits

                loss = softmax_cross_entropy_with_integer_labels(
                    logits=lm_logits,
                    labels=batch['labels']
                )
                return jnp.sum(loss * batch['decoder_attention_mask']) / jnp.sum(batch['decoder_attention_mask'])

        return _Task()
    

@gin.configurable
def sample_crossfit_T5_base(
    key: chex.PRNGKey, 
    partitions: str = 'debug',
    seeds = [13, 21, 42, 87, 100]
) -> cfgobject.CFGObject:
    rng = hk.PRNGSequence(key)
    model_name = "T5-small"
    sources = SOURCES[partitions]
    dataset_name = parametric_utils.choice(next(rng), sources)
    seed = parametric_utils.choice(next(rng), seeds)
    from .crossfit_dataset import _make_dataset
    datasets = _make_dataset(
        data_name=dataset_name,
        seed=seed,
        tokenizer_name=model_name
    )
    return ParametricCrossfitPT5(
        datasets=datasets,
        model_name=model_name
    )


@gin.configurable
def timed_sample_crossfit_pT5(key: chex.PRNGKey, max_time: float = 1e-4):
    model_path = model_paths.models[("sample_lm_transformer", "time")]
    valid_path = model_paths.models[("sample_lm_transformer", "valid")]
    return time_model.rejection_sample(
        sample_crossfit_T5_base,
        model_path,
        key,
        max_time,
        model_path_valid_suffix=valid_path)



@gin.configurable
@profile.wrap()
def random_source_distribution(key: chex.PRNGKey) -> base.TaskFamily:
    rng = hk.PRNGSequence(key)
    





                
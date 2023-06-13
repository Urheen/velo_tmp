"""Tasks for prompt-tuning bert on glue benchmark"""
import jax
import optax
import jax.numpy as jnp
from models.prompt_bert import FlaxPromptBertForSequenceClassification
from learned_optimization.tasks import base
from typing import Any
from flax.training.common_utils import onehot
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from .glue_dataset import glue_datasets
Params = AnyParams = Any
ModelState = Any
PRNGKey = jnp.ndarray


class ParametricPromptBertGLUE(base.TaskFamily):
    """A parametruc glue task based on bert model with prompt"""

    def __init__(self, model_args, data_args) -> None:
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args

    def sample():
        pass

    def task_fn(self):
        
        parent = self
        class _Task(base.Task):
            """constructed task sample"""

            def __init__(self) :
                super().__init__()




class _PromptBertGLUETask(base.Task):

    def __init__(self, 
                 model_args,
                 data_args,
        ) -> None:
        super().__init__()

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_slow_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
            cache_dir='./hf_models/'
        )

        datasets, raw_datasets, is_regression, num_labels, config = glue_datasets(
            model_args, data_args, tokenizer,
        )

        self.model = FlaxPromptBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            use_auth_token=None,
            cache_dir='./hf_models/'
        )
        
        self.datasets = datasets
        self.is_regression = is_regression
        self.num_labels = num_labels
        self.raw_datasets = raw_datasets

    def init(self, key):
        #from flax import traverse_util
        #flat_params = traverse_util.flatten_dict(jax.tree_map(lambda x: x.shape, self.model.params), sep='/')
        # print(flat_params)
        #for k, v in flat_params.items():
        #    print(f"{k}: {v}")
        #exit()
        return self.model.params
    
    def loss(self, params, key, data):
        # dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def _loss(params, key, data):
            labels=data.pop("labels")
            logits = self.model.__call__(
                **data,
                params=params,
                dropout_rng=key,
                train=True,
            )[0]
            if self.is_regression:
                return jnp.mean((logits[..., 0] - labels) ** 2)
            else:
                xentropy = optax.softmax_cross_entropy(
                    logits, onehot(labels, num_classes=self.num_labels))
                return jnp.mean(xentropy)
        # loss_fn = jax.pmap(_loss, axis_name="batch", donate_argnums=(0,))
        loss_fn = _loss
        return loss_fn(params, key, data)
        # print(f"This is data input ids {data['input_ids'].shape}")
        
def promptbert_glue(
        model_args, 
        data_args):
    return _PromptBertGLUETask(model_args, data_args)

def sample_prompt_glue():
    # randomly sample prompt length and the dataset for meta-training on
    pass
"""changed from the original huggingface file
https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_flax_bert.py
"""
import flax.linen as nn
import jax.numpy as jnp
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_flax_bert import (
   FlaxBertEmbeddings, 
   FlaxBertEncoder, 
   FlaxBertPooler,
   FlaxBertForSequenceClassificationModule,
   FlaxBaseModelOutputWithPoolingAndCrossAttentions,
   FlaxSequenceClassifierOutput,
   FlaxBertPreTrainedModel
   )
from typing import Callable, Optional, Sequence, Tuple, Protocol
# from flaxformer.types import Array
# from flaxformer.types import DType
from flax.linen import partitioning
from .prompts import Prompt

# Initializer = Callable[[Array, Sequence[int]], Array]

class FlaxPromptBertModule(nn.Module):
    config:BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    prompt_length: int = 20
    add_pooling_layer: bool = True
    gradient_checkpointing: bool = False

    def setup(self):
        self.prompt = Prompt(prompt_length=self.prompt_length)
        self.embeddings = FlaxBertEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBertEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.pooler = FlaxBertPooler(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        #print(f"befire prompting: {hidden_states}")
        hidden_states = self.prompt(input_ids, hidden_states)
        #print(f"After prompting: {hidden_states}")
        #print(f"Attention mask: {attention_mask}")
        bsz = input_ids.shape[0]
        attention_mask = jnp.concatenate((jnp.ones((bsz, self.prompt_length), attention_mask.dtype), attention_mask), axis=-1)
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
    

class FlaxPromptBertModel(FlaxBertPreTrainedModel):
   module_class = FlaxPromptBertModule


class FlaxPromptBertForSequenceClassificationModule(FlaxBertForSequenceClassificationModule):
    config:BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    prompt_length: int = 100
    add_pooling_layer: bool = True
    gradient_checkpointing: bool = False

    def setup(self):
        self.bert = FlaxPromptBertModule(
            config=self.config,
            dtype=self.dtype,
            prompt_length=self.prompt_length,
            gradient_checkpointing=self.gradient_checkpointing
        )
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )
    

class FlaxPromptBertForSequenceClassification(FlaxBertPreTrainedModel):
    module_class = FlaxPromptBertForSequenceClassificationModule


from dataclasses import dataclass, field
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_slow_tokenizer: Optional[bool] = field(
        default=False,
        metadata={"help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

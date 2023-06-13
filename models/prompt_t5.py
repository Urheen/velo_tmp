"""This script includes codes for prmtping T5"""
import jax
import copy
import jax.numpy as jnp
import flax.linen as nn
from transformers.models.t5.modeling_flax_t5 import (
    FlaxT5Stack,
    T5Config,
    FlaxT5BlockCollection,
    FlaxT5LayerNorm,
    FlaxT5ForConditionalGeneration
)
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from .prompts import Prompt


class FlaxPromptT5Stack(nn.Module):
    config: T5Config
    embed_tokens: nn.Embed
    prompt_length: int
    prompt: Prompt
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.causal = self.config.causal
        self.block = FlaxT5BlockCollection(
            self.config, dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing
        )
        self.final_layer_norm = FlaxT5LayerNorm(
            self.config.d_model,
            eps=self.config.layer_norm_epsilon,
            dtype=self.dtype
        )

        self.dropout = nn.Dropout(self.config.dropout_rate)


    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        init_cache: bool = False,
        ):
        hidden_states = self.embed_tokens(input_ids)
        bsz = input_ids.shape[0]
        if self.prompt != None and not self.config.causal:
            hidden_states = self.prompt(input_ids, hidden_states)
            attention_mask = jnp.concatenate(
                (jnp.ones((bsz, self.prompt_length), attention_mask.dtype), attention_mask), 
                axis=-1
            )

        if encoder_attention_mask is not None and self.config.causal:
            encoder_attention_mask = jnp.concatenate(
            (jnp.ones((bsz, self.prompt_length), encoder_attention_mask.dtype), encoder_attention_mask), 
            axis=-1
        )
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        
        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
        )
        hidden_states = outputs[0]
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # Add last layer
        all_hidden_states = None

        if output_hidden_states:
            all_hidden_states = outputs.hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            if output_hidden_states:
                return (
                    hidden_states,
                    all_hidden_states,
                ) + outputs[2:]
            return (hidden_states,) + outputs[1:]

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class FlaxPromptT5ForConditionalGenerationModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False,
    prompt_length: float = 100

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder
    
    def setup(self):
        self.model_dim = self.config.d_model
        self.shared_prompt = Prompt(prompt_length=self.prompt_length)

        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0),
            dtype=self.dtype
        )

        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FlaxPromptT5Stack(
            encoder_config,
            prompt_length=self.prompt_length,
            prompt=self.shared_prompt,
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing
        )

        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxPromptT5Stack(
            decoder_config,
            prompt_length=self.prompt_length,
            prompt=None,  # do not use prompt for decoder layer
            embed_tokens=self.shared,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing
        )

        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_factor),
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        deterministic: bool = True,
    ): 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        if self.config.tie_word_embeddings:
            shared_embedding = self.shared.variables["params"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, sequence_output)
        else:
            lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            return (lm_logits,) + decoder_outputs[1:] + encoder_outputs

        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
class FlaxPromptT5ForConditionalGeneration(FlaxT5ForConditionalGeneration):
    module_class = FlaxPromptT5ForConditionalGenerationModule










import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple

from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

# Qwen3 helpers (must match runtime HF package, not your generated reference copy)
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from .SlashAttnModifier import SlashAttnModifier

logger = logging.get_logger(__name__)

"""
Modified functions (Qwen3):
- eager_attention_forward
- Qwen3Attention_forward
- Qwen3DecoderLayer_forward
- Qwen3Model_forward
"""


def get_modified_forward_qwen3(layers_heads_to_modify=None, gamma=0.6, first_token_idx=0):
    modifier = SlashAttnModifier(layers_heads_to_modify, gamma, first_token_idx)

    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        # ###############BEGIN: MODIFICATION###############
        layer_to_modify: Optional[int] = None,
        # ################END: MODIFICATION################
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            # ###############BEGIN: MODIFICATION###############
            causal_mask = causal_mask.to(attn_weights.device)
            # ################END: MODIFICATION################
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

        # ###############BEGIN: MODIFICATION###############
        if layer_to_modify is not None:
            attn_weights = modifier.modify_probs(attn_weights, layer_to_modify)
        # ################END: MODIFICATION################

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    def Qwen3Attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # ###############BEGIN: MODIFICATION###############
        layer_to_modify: Optional[int] = None,
        # ################END: MODIFICATION################
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # keep Qwen3 specifics: q_norm/k_norm on head_dim
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # ###############BEGIN: MODIFICATION###############
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)
        # ################END: MODIFICATION################
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ###############BEGIN: MODIFICATION###############
        attention_interface: Callable = eager_attention_forward
        if layer_to_modify is None:
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                        "Falling back to eager attention."
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        # ################END: MODIFICATION################

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            # ###############BEGIN: MODIFICATION###############
            layer_to_modify=layer_to_modify,
            # ################END: MODIFICATION################
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def Qwen3DecoderLayer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # ###############BEGIN: MODIFICATION###############
        layer_to_modify: Optional[int] = None,
        # ################END: MODIFICATION################
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,  # unused by attention; kept for BC
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            # ###############BEGIN: MODIFICATION###############
            layer_to_modify=layer_to_modify,
            # ################END: MODIFICATION################
            **kwargs,
        )

        # device-map safety
        residual = residual.to(hidden_states.device)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

    def Qwen3Model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # ###############BEGIN: MODIFICATION###############
            if layers_heads_to_modify is None:
                layer_to_modify = idx
            else:
                heads = layers_heads_to_modify.get(str(idx)) or layers_heads_to_modify.get(idx, [])
                layer_to_modify = idx if heads else None
            # ################END: MODIFICATION################

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # keep HF behavior; do not inject layer_to_modify in checkpoint path (same as your llama file)
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    # ###############BEGIN: MODIFICATION###############
                    layer_to_modify=layer_to_modify,
                    # ################END: MODIFICATION################
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return Qwen3Model_forward, Qwen3DecoderLayer_forward, Qwen3Attention_forward
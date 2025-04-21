"""
Model overrides and wrappers.
"""

import torch
from typing import Optional, Tuple, Union
from transformers import GPT2Model, GPTNeoXModel, OlmoModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack


"""
Overrides the GPT2Model to save the final layer hidden state before the final
layer norm, to be comparable to other layers. BERTEncoder already saves
the hidden states immediately after the Transformer blocks, so it does
not need to be overridden.

This also allows the hidden states to be interpreted as a residual stream; all
hidden states are outputted from immediately before the residual connection
(and before layernorm, because GPT2 is a pre-layernorm model, i.e. the residual
connection begins before the layernorm).
"""
# Modified code labeled in comments with MODIFIED.
class GPT2ModelOverride(GPT2Model):
    # Only this function is overridden.
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        # MODIFIED: flipped the order of these to save the final hidden_states
        # immediately after the Transformer block, as with the other layers.
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # Do the layer norm after.
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        # END MODIFIED.

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


"""
Analogous to the GPT2Model override, this overrides the GPTNeoXModel to save the
final layer hidden state before the final layer norm, to be comparable to other
layers.
"""
# Modified code labeled in comments with MODIFIED.
class GPTNeoXModelOverride(GPTNeoXModel):
    # Only this function is overridden.
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict = True,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

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

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        converted_head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # Flex Attention converts it to a separate mask
        if head_mask is not None:
            converted_head_mask = ~converted_head_mask.bool() * torch.finfo(inputs_embeds.dtype).min
            converted_head_mask = converted_head_mask.to(dtype=self.dtype, device=self.device)
        head_mask = converted_head_mask

        hidden_states = self.emb_dropout(inputs_embeds)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    past_key_values,
                    output_attentions,
                    cache_position,
                    position_embeddings,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
            hidden_states = outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        # MODIFIED: flipped the order of these to save the final hidden_states
        # immediately after the Transformer block, as with the other layers.
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # Do the layer norm after.
        hidden_states = self.final_layer_norm(hidden_states)
        # END MODIFIED.

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


"""
Analogous to the GPT2Model override, this overrides the OlmoModel to save the
final layer hidden state before the final layer norm, to be comparable to other
layers.
"""
# Modified code labeled in comments with MODIFIED.
class OlmoModelOverride(OlmoModel):
    # Only this function is overridden.
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict = True,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
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

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
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
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # MODIFIED: flipped the order of these to save the final hidden_states
        # immediately after the Transformer block, as with the other layers.
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        # Do the layer norm after.
        hidden_states = self.norm(hidden_states)
        # END MODIFIED.

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


"""
A wrapper around the GPT2Model that skips some final layers, and optionally
applies a linear transformation between the last included layer and the final
layer norm before the LM head.
"""
class GPT2LensWrapper(torch.nn.Module):

    def __init__(self, orig_model, n_layers_to_keep,
            linear_map=None, linear_bias=None, apply_lnf=True):
        super().__init__()
        self.orig_model = orig_model
        # Final layer norm will be applied at the end, instead of within orig_model.
        self.ln_f = self.orig_model.ln_f if apply_lnf else None
        self.orig_model.ln_f = torch.nn.Identity()  # Identity.
        self.orig_model.h = self.orig_model.h[:n_layers_to_keep]
        # This will be applied between orig_model and ln_f.
        self.linear_map = None if linear_map is None else torch.tensor(linear_map).float()
        self.linear_bias = None if linear_bias is None else torch.tensor(linear_bias).float().reshape(1, 1, -1)

    def forward(self, input_ids: Optional[torch.LongTensor] = None, **kwargs,):
        # Get last hidden states from the GPT2Model (with some layers and ln_f
        # removed).
        orig_out = self.orig_model(input_ids, **kwargs)
        return_dict = kwargs['return_dict']
        if return_dict:
            hidden_states = orig_out['last_hidden_state']
        else:
            hidden_states = orig_out[0]
        # Apply linear transformation.
        if self.linear_map is not None:
            self.linear_map = self.linear_map.to(hidden_states.device)
            hidden_states = torch.einsum('ij,bsj->bsi', self.linear_map, hidden_states)
        if self.linear_bias is not None:
            self.linear_bias = self.linear_bias.to(hidden_states.device)
            hidden_states = hidden_states + self.linear_bias
        # Apply ln_f.
        if self.ln_f is not None:
            hidden_states = self.ln_f(hidden_states)
        # Return.
        if return_dict:
            orig_out['last_hidden_state'] = hidden_states
            return orig_out
        else:
            return hidden_states + orig_out[1:]


"""
A wrapper around the GPTNeoXModel that skips some final layers, and optionally
applies a linear transformation between the last included layer and the final
layer norm before the LM head.
"""
class GPTNeoXLensWrapper(torch.nn.Module):

    def __init__(self, orig_model, n_layers_to_keep,
            linear_map=None, linear_bias=None, apply_lnf=True):
        super().__init__()
        self.orig_model = orig_model
        # Final layer norm will be applied at the end, instead of within orig_model.
        self.ln_f = self.orig_model.final_layer_norm if apply_lnf else None
        self.orig_model.final_layer_norm = torch.nn.Identity()  # Identity.
        self.orig_model.layers = self.orig_model.layers[:n_layers_to_keep]
        # This will be applied between orig_model and ln_f.
        self.linear_map = None if linear_map is None else torch.tensor(linear_map).float()
        self.linear_bias = None if linear_bias is None else torch.tensor(linear_bias).float().reshape(1, 1, -1)

    def forward(self, input_ids: Optional[torch.LongTensor] = None, **kwargs,):
        # Get last hidden states from the GPT2Model (with some layers and ln_f
        # removed).
        orig_out = self.orig_model(input_ids, **kwargs)
        return_dict = kwargs['return_dict']
        if return_dict:
            hidden_states = orig_out['last_hidden_state']
        else:
            hidden_states = orig_out[0]
        # Apply linear transformation.
        if self.linear_map is not None:
            self.linear_map = self.linear_map.to(hidden_states.device)
            hidden_states = torch.einsum('ij,bsj->bsi', self.linear_map, hidden_states)
        if self.linear_bias is not None:
            self.linear_bias = self.linear_bias.to(hidden_states.device)
            hidden_states = hidden_states + self.linear_bias
        # Apply ln_f.
        if self.ln_f is not None:
            hidden_states = self.ln_f(hidden_states)
        # Return.
        if return_dict:
            orig_out['last_hidden_state'] = hidden_states
            return orig_out
        else:
            return hidden_states + orig_out[1:]


"""
A wrapper around the OlmoModel that skips some final layers, and optionally
applies a linear transformation between the last included layer and the final
layer norm before the LM head.
"""
class OlmoLensWrapper(torch.nn.Module):

    def __init__(self, orig_model, n_layers_to_keep,
            linear_map=None, linear_bias=None, apply_lnf=True):
        super().__init__()
        self.orig_model = orig_model
        # Final layer norm will be applied at the end, instead of within orig_model.
        self.ln_f = self.orig_model.norm if apply_lnf else None
        self.orig_model.norm = torch.nn.Identity()  # Identity.
        self.orig_model.layers = self.orig_model.layers[:n_layers_to_keep]
        # This will be applied between orig_model and ln_f.
        self.linear_map = None if linear_map is None else torch.tensor(linear_map).float()
        self.linear_bias = None if linear_bias is None else torch.tensor(linear_bias).float().reshape(1, 1, -1)

    def forward(self, input_ids: Optional[torch.LongTensor] = None, **kwargs,):
        # Get last hidden states from the GPT2Model (with some layers and ln_f
        # removed).
        orig_out = self.orig_model(input_ids, **kwargs)
        return_dict = kwargs['return_dict']
        if return_dict:
            hidden_states = orig_out['last_hidden_state']
        else:
            hidden_states = orig_out[0]
        # Apply linear transformation.
        if self.linear_map is not None:
            self.linear_map = self.linear_map.to(hidden_states.device)
            hidden_states = torch.einsum('ij,bsj->bsi', self.linear_map, hidden_states)
        if self.linear_bias is not None:
            self.linear_bias = self.linear_bias.to(hidden_states.device)
            hidden_states = hidden_states + self.linear_bias
        # Apply ln_f.
        if self.ln_f is not None:
            hidden_states = self.ln_f(hidden_states)
        # Return.
        if return_dict:
            orig_out['last_hidden_state'] = hidden_states
            return orig_out
        else:
            return hidden_states + orig_out[1:]

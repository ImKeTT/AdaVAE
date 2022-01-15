#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: vae.py
@author: ImKe at 2021/12/23
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
## core codes for adapter applying
import logging

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import BertModel
# from transformers.modeling_bert import ACT2FN, BertSelfOutput
from transformers.modeling_gpt2 import ACT2FN, Attention, GPT2Model, Block, MLP, GPT2LMHeadModel
# from transformers.models.gpt2.modeling_gpt2 import ACT2FN, GPT2Attention, GPT2Model, GPT2Block, GPT2MLP, GPT2LMHeadModel
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary

from adapters.common import AdapterConfig


logging.basicConfig(level=logging.INFO)


## attention averaged block to produce latent variable, essentially a self-attention process
class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size, ada_config):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        if isinstance(ada_config.adapter_act, str):
            self.activation = ACT2FN[ada_config.adapter_act]
        else:
            self.activation = ada_config.adapter_act

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.activation(inputs.matmul(self.attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze(1)

        return representations, scores


# Pseudo self-attention
## PSA for additive z infusion
class Cond_Attention(Attention):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        # self.output_attentions = config.output_attentions
        self.output_attentions = False

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

        # add code here
        self.c_z = Conv1D(n_state * 2, nx)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd : ns, :ns]
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # add code here: w size has been bsz * n_heads * L * (L+1), mask bsz * 1 * 1 * L
            assert attention_mask.size()[-1] == w.size()[-1] - 1
            zeros = torch.zeros(attention_mask.size()[:-1], device=attention_mask.device, dtype=attention_mask.dtype).unsqueeze(-1)
            attention_mask = torch.cat((zeros, attention_mask), dim=-1)

            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def forward(self, x, z,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                output_attentions=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        z_conv = self.c_z(z)
        key_z, value_z = z_conv.split(self.split_size, dim=2)
        key_z = self.split_heads(key_z, k=True)
        value_z = self.split_heads(value_z)
        key = torch.cat((key_z, key), dim=-1)
        value = torch.cat((value_z, value), dim=-2)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


## adapter block
class GPT2Adapter(nn.Module):
    def __init__(self, ada_config: AdapterConfig):
        super(GPT2Adapter, self).__init__()
        self.down_project = nn.Linear(ada_config.hidden_size, ada_config.adapter_size)
        ## initialize down_project weight and bias
        nn.init.normal_(self.down_project.weight, std=ada_config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)

        if isinstance(ada_config.adapter_act, str):
            self.activation = ACT2FN[ada_config.adapter_act]
        else:
            self.activation = ada_config.adapter_act

        self.up_project = nn.Linear(ada_config.adapter_size, ada_config.hidden_size)
        ## initialize up_project weight and bias
        nn.init.normal_(self.up_project.weight, std=ada_config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states: torch.Tensor):
        ## essentially a ''down mapping'' and an ''up mapping'' process
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected

class Cond_GPT2Adapter(nn.Module):
    """GPT2Adapter with label embedding infused during generation"""
    def __init__(self, ada_config: AdapterConfig):
        super(Cond_GPT2Adapter, self).__init__()
        self.down_project = nn.Linear(ada_config.hidden_size, ada_config.adapter_size)
        ## initialize down_project weight and bias
        nn.init.normal_(self.down_project.weight, std=ada_config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)

        if isinstance(ada_config.adapter_act, str):
            self.activation = ACT2FN[ada_config.adapter_act]
        else:
            self.activation = ada_config.adapter_act

        self.up_project = nn.Linear(ada_config.adapter_size + ada_config.label_emb_size, ada_config.hidden_size)
        # self.infuser = nn.Linear(ada_config.label_emb_size+ada_config.hidden_size, ada_config.hidden_size)
        ## initialize up_project weight and bias
        nn.init.normal_(self.up_project.weight, std=ada_config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states: torch.Tensor, label_embedding: torch.Tensor):
        assert len(label_embedding.size()) == 3, f'label embedding should have dimension of 3'
        ## essentially a ''down mapping'' and an ''up mapping'' process
        down_projected = self.down_project(hidden_states)
        infused_projected = torch.cat([label_embedding, down_projected], -1)
        activated = self.activation(infused_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected

"""
class BertAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: BertSelfOutput,
                 config: AdapterConfig):
        super(BertAdaptedSelfOutput, self).__init__()
        self.self_output = self_output
        self.adapter = BertAdapter(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_output.dense(hidden_states)
        hidden_states = self.self_output.dropout(hidden_states)
        ## instantiate adapter obj
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states
"""


####################### auxiliary attention blocks #######################
class Unmasked_Attention(Attention):
    """
    unmasked attention layer for encoder, re-define _atten function
    """
    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

class Unmasked_Block(Block):
    """
    base block of Encoder, unmasked/bi-directional structure in the encoder
    to allow full information scope.
    Optimus uses BERT (bi-directional) to achieve this goal
    """
    def __init__(self, n_ctx, config, ada_config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Unmasked_Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

## Additive attention block for method 2 in the paper
class Cond_Block(Block):
    def __init__(self, n_ctx, config, ada_config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Cond_Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, z, layer_past=None, attention_mask=None, head_mask=None):
        output_attn = self.attn(
            self.ln_1(x), z, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)

class Unmasked_AdapterBlock(Block):
    """
    base block of Encoder, unmasked/bi-directional structure in the encoder
    to allow full information scope.
    Optimus uses BERT (bi-directional) to achieve this goal
    """
    def __init__(self, n_ctx, config, ada_config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Unmasked_Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(nx, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.adapter = GPT2Adapter(ada_config)

    def forward(
            self, x, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, use_cache=False, output_attentions=False
    ):
        output_attn = self.attn(self.ln_1(x),
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask,
                                use_cache=use_cache,
                                output_attentions=output_attentions,)
        a = output_attn[0]  # output_attn: a, present, (attentions)
        outputs = output_attn[1:]

        x = x + a
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(x),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            x = x + attn_output
            outputs = outputs + cross_attn_outputs[1:]  # add cross attentions if we output attention weights
        # todo: where to add adapter
        x = self.adapter(x)
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + outputs
        return outputs  # x, present, (attentions)

## Additive attention block for method 2 in the paper
class Cond_AdapterBlock(Block):
    """
    to infuse latent variable z to attention layers
    todo: infuse condition embedding too
    """
    def __init__(self, n_ctx, config, ada_config, cond_adapter=False, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Cond_Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(nx, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.cond_adapter = cond_adapter
        if cond_adapter:
            self.adapter = Cond_GPT2Adapter(ada_config)
        else:
            self.adapter = GPT2Adapter(ada_config)

    def forward(self, x, z,
                label_emb=None,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                output_attentions=False,):
        output_attn = self.attn(
            self.ln_1(x), z,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
        )
        ## [bs, max_len, hidden size]
        a = output_attn[0]  # output_attn: a, present, (attentions)
        outputs = output_attn[1:]

        ## residual connection
        x = x + a
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(x),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            x = x + attn_output
            outputs = outputs + cross_attn_outputs[1:]  # add cross attentions if we output attention weights
        if self.cond_adapter:
            assert (label_emb is not None), 'get none label embedding'
            label_emb = label_emb.unsqueeze(1).repeat(1, x.size(1), 1)
            x = self.adapter(x, label_emb)
        else:
            x = self.adapter(x)
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + outputs
        return outputs  # x, present, (attentions)

####################### transformer-based vae #######################
class Encoder(GPT2Model):
    def __init__(self, config, ada_config):
        super(GPT2Model, self).__init__(config)
        # self.output_hidden_states = config.output_hidden_states
        # self.output_attentions = config.output_attentions ## True is return hidden_states
        # self.output_past = config.output_past
        self.output_hidden_states = False
        self.output_attentions = False ## True is return hidden_states
        self.output_past = False

        ## wte is word token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        ## wpe is word position embedding
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        # manually modify number of layers in encoder to accommodate GPU memory
        n = 6  # config.n_layer
        self.h = nn.ModuleList([Unmasked_AdapterBlock(config.n_ctx, config, ada_config, scale=True) for _ in range(n)])
        ## Fine-tuning encoder block
        # self.h = nn.ModuleList([Unmasked_Block(config.n_ctx, config, scale=True) for _ in range(n)])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon) ##  The epsilon to use in the layer normalization layers

        self.init_weights()

        # added code here
        self.averageSelfAttention = AverageSelfAttention(config.n_embd, ada_config)
        nx = config.n_embd
        nz = ada_config.latent_size
        self.mean = Conv1D(nz, nx)
        self.logvar = Conv1D(nz, nx)

    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        ## hidden states of a block
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
            )

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # added code here
        ## latent space parameterization
        representations, _ = self.averageSelfAttention(hidden_states, attention_mask.squeeze(1).squeeze(1))
        mean = self.mean(representations)
        logvar = self.logvar(representations)

        outputs = (mean, logvar, hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        return outputs  # mean, logvar, last hidden state, (presents), (all hidden_states), (attentions)


class Decoder(GPT2Model):
    def __init__(self, config, ada_config, add_input=False, add_attn=False, attn_proj_vary=False, label_cond=False):
        """

        :param config:
        :param add_input:
        :param add_attn:
        :param attn_proj_vary:
        :param cond: whether add label embed to decoder adapter
        """
        super(GPT2Model, self).__init__(config)

        # added code here
        self.add_input = add_input
        self.add_attn = add_attn
        self.attn_proj_vary = attn_proj_vary
        self.label_cond = label_cond

        # self.output_hidden_states = config.output_hidden_states
        # self.output_attentions = config.output_attentions
        # self.output_past = config.output_past
        self.output_hidden_states = False
        self.output_attentions = False  ## True is return hidden_states
        self.output_past = True

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        ## choose different conditional generation methods (word embedding/attention bolck/softmax decoding)
        if self.add_input:
            nz = ada_config.latent_size
            nx = config.n_embd
            nl = ada_config.label_emb_size
            self.input_proj = nn.Linear(nz + nl, nx, bias=False)

        if self.add_attn:
            nz = ada_config.latent_size
            nx = config.n_embd
            n = config.n_layer
            nl = ada_config.label_emb_size

            if self.attn_proj_vary:
                self.attn_proj = nn.Linear(nz + nl, nx * n, bias=False)
            else:
                self.attn_proj = nn.Linear(nz + nl, nx, bias=False)

            self.h = nn.ModuleList([Cond_AdapterBlock(config.n_ctx, config, ada_config,
                                                      cond_adapter=label_cond, scale=True) for _ in range(config.n_layer)])
            ## Fine-tuing decoder block
            # self.h = nn.ModuleList([Cond_Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        else:
            self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            representations=None,
            label_emb = None
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds

        # add code here
        ## method 1 in the paper: add to word embedding
        if self.add_input:
            assert (representations is not None)
            representations = torch.cat([representations, label_emb], dim=-1)
            input_proj = self.input_proj(representations).unsqueeze(1)
            hidden_states = hidden_states + input_proj

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        # add code here
        ## method 2 in the paper: add to attention layers
        if self.add_attn:
            assert (representations is not None)
            ## add condition to latent representation via concatenation
            representations = torch.cat([representations, label_emb], dim=-1)
            attn_proj = self.attn_proj(representations).unsqueeze(1)
            if self.attn_proj_vary:
                attn_proj = attn_proj.split(hidden_states.size(-1), dim=-1)
                assert len(attn_proj) == len(self.h)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            if self.add_attn:
                if self.attn_proj_vary:
                    z = attn_proj[i]
                else:
                    z = attn_proj
                ## add label embedding to decoder adapter
                if self.label_cond:
                    outputs = block(
                        hidden_states, z, label_emb=label_emb, layer_past=layer_past, attention_mask=attention_mask,
                        head_mask=head_mask[i]
                    )
                else:
                    outputs = block(
                        hidden_states, z, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
                    )
            else:
                outputs = block(
                    hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
                )

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


class LM_head_rep(nn.Module):
    def __init__(self, in_dim=768, out_dim=50257):
        super().__init__()

        self.Nu_fc1 = nn.Linear(in_dim, 1024)
        self.Nu_fc2 = nn.Linear(1024, out_dim)

    def forward(self, z):
        z = F.leaky_relu(self.Nu_fc1(z))
        z = self.Nu_fc2(z)
        return z


class VAEModel(GPT2LMHeadModel):
    def __init__(self, config, ada_config, add_input=False, add_attn=False, add_softmax=False,
                 attn_proj_vary=False, learn_prior=False, label_cond=False):
        super(GPT2LMHeadModel, self).__init__(config)

        # add code here
        self.add_input = add_input
        self.add_attn = add_attn
        self.add_softmax = add_softmax
        self.attn_proj_vary = attn_proj_vary
        self.learn_prior = learn_prior

        self.transformer = Decoder(config, ada_config, add_input, add_attn, attn_proj_vary, label_cond)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.encoder = Encoder(config, ada_config)
        if self.learn_prior:
            self.encoder_prior = Encoder(config, ada_config)

        if self.add_softmax:
            nz = config.n_embd
            self.lm_head_rep = Conv1D(config.vocab_size, nz)
            # self.lm_head_rep = LM_head_rep(nz, config.vocab_size)

    def reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = torch.randn(std.size(), device=mean.device, dtype=mean.dtype)
        return z.mul(std) + mean

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        x_mask=None,
        x_tokens=None,
        y_mask=None,
        y_tokens=None,
        from_prior=False,
        from_mean=False
    ):
        # latent representation
        posterior_mean, posterior_logvar = self.encoder(input_ids=y_tokens, attention_mask=y_mask)[:2]

        if self.learn_prior:
            prior_mean, prior_logvar = self.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
        else:
            prior_mean = prior_logvar = torch.zeros([input_ids.size(0), self.config.n_embd], device=input_ids.device)
            prior_mean, prior_logvar = prior_mean.to(posterior_mean.dtype), prior_logvar.to(posterior_logvar.dtype)

        if from_prior:
            latent_mean, latent_logvar = prior_mean, prior_logvar
        else:
            latent_mean, latent_logvar = posterior_mean, posterior_logvar

        if from_mean:
            z = latent_mean
        else:
            z = self.reparameterize(latent_mean, latent_logvar)
        assert not torch.isnan(z).any(), 'training get nan z'

        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds,
                                               representations=z)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        if self.add_softmax:
            lm_logits_rep = self.lm_head_rep(z)
            lm_logits = lm_logits + lm_logits_rep.unsqueeze(dim=1)
        outputs = (lm_logits,) + transformer_outputs[1:]

        # kl_loss
        kl_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar).unsqueeze(0)
        outputs = outputs + (kl_loss,)

        return outputs  # lm_logits, presents, (all hidden_states), (attentions), (kl_loss)

class AdaVAEModel(GPT2LMHeadModel):
    def __init__(self, config, ada_config, add_input=False, add_attn=False, add_softmax=False,
                 attn_proj_vary=False, learn_prior=False, adv_loss=False, label_cond=False):
        super(GPT2LMHeadModel, self).__init__(config)

        # add code here
        self.add_input = add_input
        self.add_attn = add_attn
        self.add_softmax = add_softmax
        self.attn_proj_vary = attn_proj_vary
        self.learn_prior = learn_prior
        self.use_adv_loss = adv_loss
        self.label_cond = label_cond
        self.ada_config = ada_config

        self.transformer = Decoder(config, ada_config, add_input, add_attn,
                                   attn_proj_vary, label_cond)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if self.label_cond:
            self.label_embedding = nn.Sequential(
                nn.Linear(ada_config.class_num, ada_config.label_emb_size),
                nn.ReLU(),
                nn.Linear(ada_config.label_emb_size, ada_config.label_emb_size)
            )

        self.encoder = Encoder(config, ada_config)
        if self.learn_prior:
            self.encoder_prior = Encoder(config, ada_config)

        if self.add_softmax:
            nz = config.n_embd
            self.lm_head_rep = Conv1D(config.vocab_size, nz)
            # self.lm_head_rep = LM_head_rep(nz, config.vocab_size)
        if self.use_adv_loss:
            self.discriminator = nn.Sequential(nn.Linear(config.n_embd, config.dis_emb),
                                               nn.ReLU(),
                                               nn.Linear(config.dis_emb, 1),
                                               nn.Softmax())

    def reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = torch.randn(std.size(), device=mean.device, dtype=mean.dtype)
        return z.mul(std) + mean

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def adv_loss(self, mean1, logvar1, mean2, logvar2):
        """
        adversarial loss for wasserstrain distance calculation
        from Educating Text Autoencoders: Latent Representation Guidance via Denoising
        https://arxiv.org/abs/1905.12777
        :param mean1: posterior mean
        :param logvar1: posterior
        :param mean2: prior
        :param logvar2: prior
        :return:
        """
        z = self.reparameterize(mean1, logvar1)
        zn = self.reparameterize(mean2, logvar2) # drawn from prior
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        ## discriminator loss
        loss_d = F.binary_cross_entropy(self.discriminator(z.detach()), zeros) + \
                 F.binary_cross_entropy(self.discriminator(zn), ones)
        ## generator loss
        loss_g = F.binary_cross_entropy(self.discriminator(z), ones)
        return loss_d, loss_g


    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_onehot=None,
        from_prior=False,
        from_mean=False
    ):
        # latent representation
        ## mean, logvar, last hidden state, (presents), (all hidden_states), (attentions)
        posterior_mean, posterior_logvar = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[:2]

        prior_mean = prior_logvar = torch.zeros([input_ids.size(0), self.ada_config.latent_size], device=input_ids.device)
        prior_mean, prior_logvar = prior_mean.to(posterior_mean.dtype), prior_logvar.to(posterior_logvar.dtype)

        if from_prior:
            latent_mean, latent_logvar = prior_mean, prior_logvar
        else:
            latent_mean, latent_logvar = posterior_mean, posterior_logvar

        if from_mean:
            z = latent_mean
        else:
            z = self.reparameterize(latent_mean, latent_logvar)
        assert not torch.isnan(z).any(), 'training get nan z'

        if self.label_cond:
            assert (label_onehot is not None), 'one hot label embedding get nan'
            label_emb = self.label_embedding(label_onehot)
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds,
                                               representations=z,
                                               label_emb=label_emb)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        if self.add_softmax:
            lm_logits_rep = self.lm_head_rep(z)
            lm_logits = lm_logits + lm_logits_rep.unsqueeze(dim=1)
        outputs = (lm_logits,) + transformer_outputs[1:]

        if self.use_adv_loss:
            regularization_loss = self.adv_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar)
        else:
            # kl_loss
            regularization_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar).unsqueeze(0)
        outputs = outputs + (regularization_loss,)

        return outputs  # lm_logits, presents, (all hidden_states), (attentions), (regularization_loss)
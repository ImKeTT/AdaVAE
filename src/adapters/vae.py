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
import math, sys
import torch.nn.functional as F
from transformers.modeling_gpt2 import ACT2FN, Attention, GPT2Model, Block, MLP, GPT2LMHeadModel
# from transformers.models.gpt2.modeling_gpt2 import ACT2FN, GPT2Attention, GPT2Model, GPT2Block, GPT2MLP, GPT2LMHeadModel
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
sys.path.append('../')
from .common import AdapterConfig, init_lisa_params, init_bert_weights, init_bias_mlp, init_zero_weights, LoRALinear, Adapter_Layer, Prefix


logging.basicConfig(level=logging.INFO)


## attention averaged block to produce latent variable, essentially a self-attention process
class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size, AdapterConfig):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        if isinstance(AdapterConfig.adapter_act, str):
            self.activation = ACT2FN[AdapterConfig.adapter_act]
        else:
            self.activation = AdapterConfig.adapter_act

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
    def __init__(self, nx, n_ctx, config, AdapterConfig, scale=False):
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

class MAM_Attention(Attention):
    """
    parallel adapter with prefix-tuning and LoRA component
    """
    def __init__(self, nx, n_ctx, config, AdapterConfig, add_attn=True, add_mem=False, bias_bool=True, scale=False):
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
        self.config = config
        self.attn_mode = AdapterConfig.attn_mode
        self.attn_option = AdapterConfig.attn_option
        self.add_mem = add_mem
        self.add_attn = add_attn

        # self.c_attn = Conv1D(n_state * 3, nx)
        ## use linear layer (which is approximately equivelent to Conv1D) to parameterize q, k, v
        if AdapterConfig.attn_mode == "lora": # not compatible yet
            self.q_proj = LoRALinear(nx, n_state, r=config.attn_bn, lora_alpha=AdapterConfig.lora_alpha,
                                 lora_dropout=config.lora_dropout)
            self.v_proj = LoRALinear(nx, n_state, r=config.attn_bn, lora_alpha=AdapterConfig.lora_alpha,
                                 lora_dropout=config.lora_dropout)
        else:
            self.c_attn = Conv1D(n_state * 3, nx)

        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

        if add_mem:
            self.latent2mem = nn.Linear(nx, nx, bias=False)

        if add_attn:
            # add code here
            self.c_z = Conv1D(n_state * 2, nx)

        if 'prefix' in self.attn_mode:
            if self.attn_option == 'cross_attn' or self.attn_option == 'cross_attn_relu':
                self.ef_transform_layer_norm = nn.LayerNorm(config.hidden_size)

        elif self.attn_mode == 'adapter':
            self.ef_attn_adapter = GPT2Adapter(AdapterConfig)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        ## attention scores
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
                output_attentions=False,
                prefix_state=None,
                ):
        bsz = x.size(0)
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if self.add_mem:
            layer_past = [self.latent2mem(z),
                    self.latent2mem(z)]  # query, key
            # layer_past = [past] * self.decoder_n_layer
            attention_mask = None
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            if len(past_key.size()) != len(key.size()):
                past_key = self.split_heads(past_key.transpose(-2, -1), k=True)
            key = torch.cat((past_key, key), dim=-1)
            if len(past_value.size()) != len(value.size()):
                past_value = self.split_heads(past_value)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        if prefix_state is not None and "prefix" in self.attn_mode:
            # legacy
            prefix_key = prefix_state['prev_key']  # bsz x nhead, attn_bn, head_dim
            prefix_value = prefix_state['prev_value']
            prefix_mask = prefix_state['prev_key_padding_mask']  # bsz, attn_bn: zeros

            # (bsz, nhead, attn_bn, head_im)
            ## GPT2 key dim is different from BERT
            prefix_key = prefix_key.view(bsz, self.config.n_head, *prefix_key.size()[-2:]).permute(0, 1, 3, 2)
            prefix_value = prefix_value.view(bsz, self.config.n_head, *prefix_value.size()[-2:])

            # import pdb; pdb.set_trace()
            # original lisa prefix-tuning
            # if self.cache_key == "self":
            #     key_states = torch.cat([key_states[:, 0, :].unsqueeze(1), prefix_key, key_states[:, 1:, :]], dim=1)
            #     value_states = torch.cat([value_states[:, 0, :].unsqueeze(1), prefix_value, value_states[:, 1:, :]], dim=1)
            # else:
            key = torch.cat([prefix_key, key], dim=3)
            value = torch.cat([prefix_value, value], dim=2)

            # import pdb; pdb.set_trace()
            if attention_mask is not None:
                # import pdb; pdb.set_trace()
                expanded_prefix_mask = prefix_mask[:, None, None, :].expand(bsz, 1, attention_mask.size(2),
                                                                            prefix_mask.size(1)).to(attention_mask.dtype)
                attention_mask = torch.cat([expanded_prefix_mask, attention_mask], dim=-1)
        elif self.attn_mode == "adapter":
            pass

        if self.add_attn:
            ## PSA concat
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
    def __init__(self, AdapterConfig: AdapterConfig):
        super(GPT2Adapter, self).__init__()
        self.down_project = nn.Linear(AdapterConfig.hidden_size, AdapterConfig.adapter_size)
        self.up_project = nn.Linear(AdapterConfig.adapter_size, AdapterConfig.hidden_size)
        ## initialize down_project weight and bias
        if AdapterConfig.init == "bert":
            self.apply(init_bert_weights)
        elif AdapterConfig.init == "lisa":
            self.apply(init_lisa_params)
        elif AdapterConfig.init == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_project.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_project.weight)
                nn.init.zeros_(self.down_project.bias)
                nn.init.zeros_(self.up_project.bias)
        else:
            nn.init.normal_(self.up_project.weight, std=AdapterConfig.adapter_initializer_range)
            nn.init.zeros_(self.up_project.bias)
            nn.init.normal_(self.down_project.weight, std=AdapterConfig.adapter_initializer_range)
            nn.init.zeros_(self.down_project.bias)

        if AdapterConfig.adapter_scalar=="learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(AdapterConfig.adapter_scalar)

        if isinstance(AdapterConfig.adapter_act, str):
            self.activation = ACT2FN[AdapterConfig.adapter_act]
        else:
            self.activation = AdapterConfig.adapter_act


    def forward(self, hidden_states: torch.Tensor, adapter_res:bool=True, residual: torch.Tensor=None, adapter_layernorm_option:str=None):
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(hidden_states.size(-1))
        if adapter_layernorm_option == 'in':
            hidden_states = self.adapter_layer_norm_before(hidden_states)
        ## essentially a ''down mapping'' and an ''up mapping'' process
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        up_projected *= self.scale
        if adapter_layernorm_option == 'out':
            up_projected = self.adapter_layer_norm_before(up_projected)
        if residual is not None:
            up_projected += residual
        if adapter_res:
            out = hidden_states + up_projected
        else:
            out = up_projected
        return out

## Conditional GPT2 Adapter, support parallel or sequential adapter (LoRA/Adapter)
class Cond_GPT2Adapter(nn.Module):
    """GPT2Adapter with label embedding infused during generation"""
    def __init__(self, AdapterConfig: AdapterConfig):
        super(Cond_GPT2Adapter, self).__init__()
        self.down_project = nn.Linear(AdapterConfig.hidden_size, AdapterConfig.adapter_size)
        self.up_project = nn.Linear(AdapterConfig.adapter_size + AdapterConfig.label_emb_size, AdapterConfig.hidden_size)
        # self.infuser = nn.Linear(AdapterConfig.label_emb_size+AdapterConfig.hidden_size, AdapterConfig.hidden_size)
        ## initialize up_project weight and bias
        ## initialize down_project weight and bias
        if AdapterConfig.init == "bert":
            self.apply(init_bert_weights)
        elif AdapterConfig.init == "lisa":
            self.apply(init_lisa_params)
        elif AdapterConfig.init == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_project.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_project.weight)
                nn.init.zeros_(self.down_project.bias)
                nn.init.zeros_(self.up_project.bias)
        else:
            nn.init.normal_(self.up_project.weight, std=AdapterConfig.adapter_initializer_range)
            nn.init.zeros_(self.up_project.bias)
            nn.init.normal_(self.down_project.weight, std=AdapterConfig.adapter_initializer_range)
            nn.init.zeros_(self.down_project.bias)

        if AdapterConfig.adapter_scalar=="learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(AdapterConfig.adapter_scalar)

        if isinstance(AdapterConfig.adapter_act, str):
            self.activation = ACT2FN[AdapterConfig.adapter_act]
        else:
            self.activation = AdapterConfig.adapter_act


    def forward(self, hidden_states: torch.Tensor, label_embedding: torch.Tensor,
                adapter_res:bool=True, residual: torch.Tensor=None, adapter_layernorm_option:str=None):
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(hidden_states.size(-1))
        assert len(label_embedding.size()) == 3, f'label embedding should have dimension of 3'
        if adapter_layernorm_option == 'in':
            hidden_states = self.adapter_layer_norm_before(hidden_states)
        ## essentially a ''down mapping'' and an ''up mapping'' process
        down_projected = self.down_project(hidden_states)
        infused_projected = torch.cat([label_embedding, down_projected], -1)
        activated = self.activation(infused_projected)
        up_projected = self.up_project(activated)
        up_projected *= self.scale

        if adapter_layernorm_option == 'out':
            up_projected = self.adapter_layer_norm_before(up_projected)

        if residual is not None:
            up_projected += residual
        if adapter_res:
            out = hidden_states + up_projected
        else:
            out = up_projected
        return out

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


################################# Attention Blocks #########################################
####################### auxiliary attention blocks w/o Adapter BEGIN #######################
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

class MAM_Unmasked_Attention(Attention):
    """
    parallel adapter with prefix-tuning and LoRA component
    """
    def __init__(self, nx, n_ctx, config, AdapterConfig, scale=False):
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
        self.config = config
        self.attn_mode = AdapterConfig.attn_mode
        self.attn_option = AdapterConfig.attn_option

        # self.c_attn = Conv1D(n_state * 3, nx)
        ## use linear layer (which is approximately equivelent to Conv1D) to parameterize q, k, v
        if AdapterConfig.attn_mode == "lora":
            self.q_proj = LoRALinear(nx, n_state, r=config.attn_bn, lora_alpha=AdapterConfig.lora_alpha,
                                 lora_dropout=config.lora_dropout)
            self.v_proj = LoRALinear(nx, n_state, r=config.attn_bn, lora_alpha=AdapterConfig.lora_alpha,
                                 lora_dropout=config.lora_dropout)
        else:
            self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

        if 'prefix' in self.attn_mode:
            if self.attn_option == 'cross_attn' or self.attn_option == 'cross_attn_relu':
                self.ef_transform_layer_norm = nn.LayerNorm(config.hidden_size)

        elif self.attn_mode == 'adapter':
            self.ef_attn_adapter = GPT2Adapter(AdapterConfig)

        elif self.attn_mode != 'none':
            raise ValueError("att_mode not supported")

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        """
        unmasked attention layer for encoder, re-define _atten function
        """
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

    def forward(self, x, z=None,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                output_attentions=False,
                prefix_state=None,
                ):
        bsz = x.size(0)
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

        if prefix_state is not None and "prefix" in self.attn_mode:
            # legacy
            prefix_key = prefix_state['prev_key']  # bsz x nhead, attn_bn, head_dim
            prefix_value = prefix_state['prev_value']
            prefix_mask = prefix_state['prev_key_padding_mask']  # bsz, attn_bn: zeros

            # (bsz, nhead, attn_bn, head_im)
            prefix_key = prefix_key.view(bsz, self.config.n_head, *prefix_key.size()[-2:]).permute(0, 1, 3, 2)
            prefix_value = prefix_value.view(bsz, self.config.n_head, *prefix_value.size()[-2:])

            # import pdb; pdb.set_trace()
            # original lisa prefix-tuning
            # if self.cache_key == "self":
            #     key_states = torch.cat([key_states[:, 0, :].unsqueeze(1), prefix_key, key_states[:, 1:, :]], dim=1)
            #     value_states = torch.cat([value_states[:, 0, :].unsqueeze(1), prefix_value, value_states[:, 1:, :]], dim=1)
            # else:
            key = torch.cat([prefix_key, key], dim=3)
            value = torch.cat([prefix_value, value], dim=2)

            # import pdb; pdb.set_trace()
            if attention_mask is not None:
                # import pdb; pdb.set_trace()
                expanded_prefix_mask = prefix_mask[:, None, None, :].expand(bsz, 1, attention_mask.size(2),
                                                                            prefix_mask.size(1)).to(attention_mask.dtype)
                attention_mask = torch.cat([expanded_prefix_mask, attention_mask], dim=-1)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)

class Unmasked_Block(Block):
    """
    base block of Encoder, unmasked/bi-directional structure in the encoder
    to allow full information scope.
    Optimus uses BERT (bi-directional) to achieve this goal
    """
    def __init__(self, n_ctx, config, AdapterConfig, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Unmasked_Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

## Additive attention block for method 2 in the paper
class Cond_Masked_Block(Block):
    def __init__(self, n_ctx, config, AdapterConfig, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Cond_Attention(nx, n_ctx, config, AdapterConfig, scale)
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

## Additive attention block for method 2 in the paper
class Masked_Block(Block):
    def __init__(self, n_ctx, config, AdapterConfig, add_attn=True, add_mem=False, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = MAM_Attention(nx, n_ctx, config, AdapterConfig, add_attn, add_mem, scale=scale)
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
####################### auxiliary attention blocks w/o Adapter END #######################

####################### auxiliary attention blocks w/ Adapter BEGIN ######################
class Unmasked_AdapterBlock(Block):
    """
    base block of Encoder, unmasked/bi-directional structure in the encoder
    to allow full information scope.
    Optimus uses BERT (bi-directional) to achieve this goal
    """
    def __init__(self, n_ctx, config, AdapterConfig, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        # self.attn = Unmasked_Attention(nx, n_ctx, config, scale)
        self.attn = MAM_Unmasked_Attention(nx, n_ctx, config, AdapterConfig, scale=scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        if AdapterConfig.ffn_option == "pfeiffer":
            self.ln_3 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(nx, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.Adaconfig = AdapterConfig
        self.adapter = GPT2Adapter(AdapterConfig)

    def forward(
            self, x, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, use_cache=False, output_attentions=False, prefix_state=None,
    ):
        output_attn = self.attn(self.ln_1(x),
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                                prefix_state=prefix_state,)
        a = output_attn[0]  # output_attn: a, present, (attentions)
        outputs = output_attn[1:]

        if self.Adaconfig.ffn_option == "sequential":
            x = self.adapter(x)

        ## residual connection: intermidia layer, intermediate
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
            a = a + attn_output
            outputs = outputs + cross_attn_outputs[1:]  # add cross attentions if we output attention weights

        ## hidden states
        m = self.mlp(self.ln_2(x))
        if self.Adaconfig.ffn_option == "parallel_attn":
            a = self.adapter(a, False)
            x = x + a
        elif self.Adaconfig.ffn_option == "parallel_ffn":
            x = self.adapter(x)
        elif self.Adaconfig.ffn_option == "sequential":
            m = self.adapter(m) ## a = a + adapter_change(a)
        x = x + m
        if self.Adaconfig.ffn_option == "pfeiffer":
            x = self.ln_3(self.adapter(x, adapter_res=False, residual=m, adapter_layernorm_option="in") + a)

        outputs = [x] + outputs
        return outputs  # x, present, (attentions)

## Additive attention block for method 2 in the paper
class AdapterBlock(Block):
    """
    to infuse latent variable z to attention layers
    adapter-tuning essentially add down/up projection to the output
    """
    def __init__(self, n_ctx, config, AdapterConfig, add_attn=True, add_mem=False, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        # self.attn = Cond_Attention(nx, n_ctx, config, AdapterConfig, scale)
        self.attn = MAM_Attention(nx, n_ctx, config, AdapterConfig, add_attn, add_mem, scale=scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        if AdapterConfig.ffn_option == "pfeiffer":
            self.ln_3 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(nx, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.Adaconfig = AdapterConfig
        self.adapter = GPT2Adapter(AdapterConfig)


    def forward(self, x, z,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                output_attentions=False,
                prefix_state=None,):
        output_attn = self.attn(
            self.ln_1(x), z,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                prefix_state=prefix_state,
        )
        ## [bs, max_len, hidden size]
        a = output_attn[0]  # output_attn: a, present, (attentions)
        outputs = output_attn[1:]

        if self.Adaconfig.ffn_option == "sequential":
            # label_emb = label_emb.unsqueeze(1).repeat(1, x.size(1), 1)
            x = self.adapter(x)
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

        ## hidden states
        m = self.mlp(self.ln_2(x))
        if self.Adaconfig.ffn_option == "parallel_attn":
            # label_emb = label_emb.unsqueeze(1).repeat(1, a.size(1), 1)
            a = self.adapter(a, False)
            x = x + a
        elif self.Adaconfig.ffn_option == "parallel_ffn":
            # label_emb = label_emb.unsqueeze(1).repeat(1, x.size(1), 1)
            x = self.adapter(x)
        elif self.Adaconfig.ffn_option == "sequential":
            # label_emb = label_emb.unsqueeze(1).repeat(1, m.size(1), 1)
            m = self.adapter(m)  ## a = a + adapter_change(a)
        x = x + m
        if self.Adaconfig.ffn_option == "pfeiffer":
            # label_emb = label_emb.unsqueeze(1).repeat(1, x.size(1), 1)
            self.ln_3(self.adapter(x, adapter_res=False, residual=m, adapter_layernorm_option="in") + a)
        outputs = [x] + outputs
        return outputs  # x, present, (attentions)
####################### auxiliary attention blocks w/ Adapter END ########################


####################### transformer-based vae #######################
class Encoder(GPT2Model):
    def __init__(self, config, AdapterConfig):
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
        self.latent_type = AdapterConfig.latent_gen
        self.attn_mode = AdapterConfig.attn_mode
        self.tune_enc = AdapterConfig.tune_enc

        # manually modify number of layers in encoder to accommodate GPU memory
        n = AdapterConfig.encoder_n_layer
        self.h = nn.ModuleList([Unmasked_Block(config.n_ctx, config, AdapterConfig, scale=True) for _ in range(n)]) if self.tune_enc \
            else nn.ModuleList([Unmasked_AdapterBlock(config.n_ctx, config, AdapterConfig, scale=True) for _ in range(n)])
        ## Fine-tuning encoder block
        # self.h = nn.ModuleList([Unmasked_Block(config.n_ctx, config, scale=True) for _ in range(n)])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon) ##  The epsilon to use in the layer normalization layers

        self.init_weights()

        # added code here
        nx = config.n_embd
        nz = AdapterConfig.latent_size
        if self.latent_type == "averaged_attn":
            self.averageSelfAttention = AverageSelfAttention(nx, AdapterConfig)
        elif self.latent_type == "linear":
            self.z_linear = nn.Linear(nx, nx)
        else:
            raise NotImplementedError("Not implemented !")
        self.mean = Conv1D(nz, nx)
        self.logvar = Conv1D(nz, nx)
        if self.attn_mode == "prefix":
            self.prompt_model = Prefix(AdapterConfig, config)

    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        prefix_state = None
        if self.attn_mode == "prefix":
            prefix_state = self.prompt_model(input_ids.size(0), device=input_ids.device)
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
                hidden_states, layer_past=layer_past, attention_mask=attention_mask,
                head_mask=head_mask[i]) if self.tune_enc else\
                block(hidden_states, layer_past=layer_past, attention_mask=attention_mask,
                head_mask=head_mask[i], prefix_state=prefix_state[i] if isinstance(prefix_state, list) else prefix_state)

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
        if self.latent_type == "averaged_attn":
            representations, _ = self.averageSelfAttention(hidden_states, attention_mask.squeeze(1).squeeze(1))
        elif self.latent_type == "linear":
            representations = self.z_linear(hidden_states[:, 0]) ## following optimus. We "pool" the model by simply taking the hidden state correspondin to the first token.
        else:
            raise NotImplementedError("not implemented !")
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
    def __init__(self, config, AdapterConfig, add_input=False, add_attn=False, add_mem=False, attn_proj_vary=False):
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
        self.add_mem = add_mem
        self.attn_proj_vary = attn_proj_vary

        # self.output_hidden_states = config.output_hidden_states
        # self.output_attentions = config.output_attentions
        # self.output_past = config.output_past
        self.output_hidden_states = False
        self.output_attentions = False  ## True is return hidden_states
        self.output_past = True
        self.tune_dec = AdapterConfig.tune_dec

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.attn_mode = AdapterConfig.attn_mode

        ## choose different conditional generation methods (word embedding/attention bolck/softmax decoding)
        if self.add_input:
            nz = AdapterConfig.latent_size
            nx = config.n_embd
            self.input_proj = nn.Linear(nz, nx, bias=False)

        if self.add_attn or self.add_mem:
            nz = AdapterConfig.latent_size
            nx = config.n_embd
            n = AdapterConfig.decoder_n_layer

            if self.attn_proj_vary:
                self.attn_proj = nn.Linear(nz, nx * n, bias=False)
            else:
                self.attn_proj = nn.Linear(nz, nx, bias=False)
            self.h = nn.ModuleList([Masked_Block(config.n_ctx, config, AdapterConfig, add_attn, add_mem,
                                                 scale=True) for _ in range(n)]) if self.tune_dec else \
                nn.ModuleList([AdapterBlock(config.n_ctx, config, AdapterConfig, add_attn, add_mem,
                                            scale=True) for _ in range(n)])
            ## Fine-tuing decoder block
            # self.h = nn.ModuleList([Cond_Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        else:
            n = AdapterConfig.decoder_n_layer
            self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(n)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

        if self.attn_mode == "prefix":
            self.prompt_model = Prefix(AdapterConfig, config)

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
    ):
        prefix_state = None
        if self.attn_mode == "prefix":
            prefix_state = self.prompt_model(input_ids.size(0), device=input_ids.device)
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
            # # different latent vectors for each layer
            # past_split = torch.split(past.unsqueeze(1), self.config.hidden_size, dim=2)
            # past = list(zip(past_split, past_split))
            #
            # # past = past.view(batch_size,len(self.h),-1)
            # # past = [[past[:,i,:].unsqueeze(-2), past[:,i,:].unsqueeze(-2) ] for i in range(len(self.h))]
            # past_length = 1  # past[0][0].size(-2)
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
            # representations = torch.cat([representations, label_emb], dim=-1)
            input_proj = self.input_proj(representations).unsqueeze(1)
            hidden_states = hidden_states + input_proj

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        # add code here
        ## method 2 in the paper: add to attention layers
        if self.add_attn or self.add_mem:
            assert (representations is not None)
            ## add condition to latent representation via concatenation
            # representations = torch.cat([representations, label_emb], dim=-1)
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

            if self.add_attn or self.add_mem:
                if self.attn_proj_vary:
                    z = attn_proj[i]
                else:
                    z = attn_proj
                ## add label embedding to decoder adapter
                if self.tune_dec:
                    outputs = block(
                        hidden_states, z, layer_past=layer_past, attention_mask=attention_mask,
                        head_mask=head_mask[i]
                    )
                else:
                    outputs = block(
                        hidden_states, z, layer_past=layer_past, attention_mask=attention_mask,
                        head_mask=head_mask[i], prefix_state=prefix_state[i] if isinstance(prefix_state, list) else prefix_state
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


## For VQ-VAE like big AE
class CodeBook(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        """
        Code Book for VQ-VAE
        :param num_embeddings:
        :param embedding_dim:
        :param commitment_cost:
        """
        super(CodeBook, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Calculate distances
        distances = (torch.sum(inputs ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).cuda()
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings


class VAEModel(GPT2LMHeadModel):
    def __init__(self, config, AdapterConfig, add_input=False, add_attn=False, add_softmax=False,
                 attn_proj_vary=False, learn_prior=False):
        super(GPT2LMHeadModel, self).__init__(config)

        # add code here
        self.add_input = add_input
        self.add_attn = add_attn
        self.add_softmax = add_softmax
        self.attn_proj_vary = attn_proj_vary
        self.learn_prior = learn_prior

        self.transformer = Decoder(config, AdapterConfig, add_input, add_attn, False, attn_proj_vary)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.encoder = Encoder(config, AdapterConfig)
        if self.learn_prior:
            self.encoder_prior = Encoder(config, AdapterConfig)

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
    def __init__(self, config, AdapterConfig, add_input=False, add_attn=False, add_softmax=False, add_mem=False,
                 attn_proj_vary=False, learn_prior=False, reg_loss="kld"):
        super(GPT2LMHeadModel, self).__init__(config)

        # add code here
        self.add_input = add_input
        self.add_attn = add_attn
        self.add_softmax = add_softmax
        self.add_mem = add_mem
        self.attn_proj_vary = attn_proj_vary
        self.learn_prior = learn_prior
        self.reg_loss = reg_loss
        self.AdapterConfig = AdapterConfig
        self.encoder = Encoder(config, AdapterConfig)
        self.transformer = Decoder(config, AdapterConfig, add_input, add_attn,
                                   add_mem, attn_proj_vary)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if self.learn_prior:
            self.encoder_prior = Encoder(config, AdapterConfig)

        if self.add_softmax:
            nz = config.n_embd
            self.lm_head_rep = Conv1D(config.vocab_size, nz)
            # self.lm_head_rep = LM_head_rep(nz, config.vocab_size)
        if self.reg_loss == "adversarial":
            self.discriminator = nn.Sequential(nn.Linear(AdapterConfig.latent_size, AdapterConfig.dis_emb),
                                               nn.ReLU(),
                                               nn.Linear(AdapterConfig.dis_emb, 1),
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
        return [loss_d, loss_g]

    def symlog_loss(self, mean, logvar):
        z0 = self.reparameterize(mean, logvar)
        z1 = self.reparameterize(mean, logvar)
        log_p = lambda x: -0.5 * torch.sum(torch.pow(x, 2), 1)
        return torch.abs(torch.sum(log_p(z0) - log_p(z1)))


    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        from_prior=False,
        from_mean=False,
    ):
        # latent representation
        ## mean, logvar, last hidden state, (presents), (all hidden_states), (attentions)
        posterior_mean, posterior_logvar = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[:2]

        prior_mean = prior_logvar = torch.zeros([input_ids.size(0), self.AdapterConfig.latent_size], device=input_ids.device)
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

        if self.reg_loss == "adversarial":
            regularization_loss = self.adv_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar)
            regularization_loss.append(self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar).unsqueeze(0))
        elif self.reg_loss == "kld":
            # kl_loss
            regularization_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar).unsqueeze(0)
        elif self.reg_loss == "symlog":
            regularization_loss = self.symlog_loss(posterior_mean, posterior_logvar)
        else:
            raise TypeError("No such regularization loss implemented !")
        outputs = outputs + (regularization_loss, posterior_mean, posterior_logvar)

        return outputs  # lm_logits, presents, (all hidden_states), (attentions), (regularization_loss), (posterior_mean), (posterior_logvar)
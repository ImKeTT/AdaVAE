#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: utils.py
@author: ImKe at 2021/12/28
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import random, re, os
# from data.prompt_dataset import *
# from data.plot_dataset import *
# from data.arxiv_dataset import *
# from data.yelp_dataset import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import functools
from rake_nltk import Rake
import urllib, sys
import urllib.request
import json, re
import numpy as np
import copy
import math
from tqdm import tqdm
from scipy.spatial.distance import cdist
from tqdm import trange
from random import shuffle

#############################
######## model utils ########
#############################
def safe_log(z):
    return torch.log(z + 1e-7)

def log_sum_exp(value: torch.Tensor, dim: int=None, keepdim: bool=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


################################
######## training utils ########
################################
def top_k_top_p_filtering(logits, top_k=100, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits

def sample_sequence(model, length, z=None, batch_size=None,
                    temperature=1, top_k=100, top_p=0.95, device='cuda',
                    sample=True, eos_token=None, model_type='cvae'):
    with torch.no_grad():
        # if model_type == 'cvae':
        if z is None:
            z = torch.randn([batch_size, model.AdapterConfig.latent_size], device=device)
        assert z.size() == torch.Size([batch_size, model.AdapterConfig.latent_size]), "get latent code with wrong size"
        #     try:
        #         prior_mean, prior_logvar = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
        #     except:
        #         prior_mean = prior_logvar = torch.zeros([batch_size, model.config.n_embd], device=device)
        #     latent_mean, latent_logvar = prior_mean, prior_logvar
        #     z = model.reparameterize(latent_mean, latent_logvar)
        #     assert not torch.isnan(z).any(), 'training get nan z'
        # else:
        #     posterior_mean, posterior_logvar = model.encoder(input_ids=x_tokens, attention_mask=x_mask)[:2]
        #     latent_mean, latent_logvar = posterior_mean, posterior_logvar
        #     z = latent_mean
        #     assert not torch.isnan(z).any(), 'training get nan z'

        mem = None
        prev = torch.tensor([[eos_token]] * batch_size, dtype=torch.long, device=device)

        output = prev
        probability = torch.tensor([], dtype=torch.float, device=device)
        if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)
        for i in range(length): #trange
            last_hidden, mem = model.transformer(input_ids=prev, past=mem, representations=z)

            logits = model.lm_head(last_hidden)
            if model.add_softmax:
                logits_rep = model.lm_head_rep(z)
                logits = logits + logits_rep.unsqueeze(dim=1)

            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = next_token

            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all(): break
    return output, probability

def tokenize(texts, tokenizer, device, args):
    tokenizer.pad_token = tokenizer.eos_token
    x_tokenized = tokenizer(texts, padding=True,
                                 truncation=True,
                            return_tensors='pt', max_length=args.max_length)
    input_ids = x_tokenized['input_ids'][:, :-1].to(device)
    attention_mask = x_tokenized['attention_mask'][:, 1:].to(device)
    x_ids = x_tokenized['input_ids'][:, 1:].contiguous().to(device)
    ## target, input tokens, mask
    return x_ids, input_ids, attention_mask

def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

def freeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def init_para_frompretrained(m, pm, share_para=False):
    m.wte.weight = pm.wte.weight
    m.wpe.weight = pm.wpe.weight

    for i in range(min(len(m.h), len(pm.h))):
        m.h[i].ln_1.weight = pm.h[i].ln_1.weight if share_para else copy.copy(pm.h[i].ln_1.weight)
        m.h[i].ln_1.bias = pm.h[i].ln_1.bias if share_para else copy.copy(pm.h[i].ln_1.bias)
        m.h[i].attn.c_attn.weight = pm.h[i].attn.c_attn.weight if share_para else copy.copy(pm.h[i].attn.c_attn.weight)
        m.h[i].attn.c_attn.bias = pm.h[i].attn.c_attn.bias if share_para else copy.copy(pm.h[i].attn.c_attn.bias)
        m.h[i].attn.c_proj.weight = pm.h[i].attn.c_proj.weight if share_para else copy.copy(pm.h[i].attn.c_proj.weight)
        m.h[i].attn.c_proj.bias = pm.h[i].attn.c_proj.bias if share_para else copy.copy(pm.h[i].attn.c_proj.bias)
        m.h[i].ln_2.weight = pm.h[i].ln_2.weight if share_para else copy.copy(pm.h[i].ln_2.weight)
        m.h[i].ln_2.bias = pm.h[i].ln_2.bias if share_para else copy.copy(pm.h[i].ln_2.bias)
        m.h[i].mlp.c_fc.weight = pm.h[i].mlp.c_fc.weight if share_para else copy.copy(pm.h[i].mlp.c_fc.weight)
        m.h[i].mlp.c_fc.bias = pm.h[i].mlp.c_fc.bias if share_para else copy.copy(pm.h[i].mlp.c_fc.bias)
        m.h[i].mlp.c_proj.weight = pm.h[i].mlp.c_proj.weight if share_para else copy.copy(pm.h[i].mlp.c_proj.weight)
        m.h[i].mlp.c_proj.bias = pm.h[i].mlp.c_proj.bias if share_para else copy.copy(pm.h[i].mlp.c_proj.bias)

    m.ln_f.weight = pm.ln_f.weight if share_para else copy.copy(pm.ln_f.weight)
    m.ln_f.bias = pm.ln_f.bias if share_para else copy.copy(pm.ln_f.bias)

def unfreeze_GPT2_adapters(GPT2_model: nn.Module, Adapters: list) -> nn.Module:
    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in GPT2_model.named_modules():
        for adapter in Adapters:
            if isinstance(sub_module, (adapter, nn.LayerNorm)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True
    return GPT2_model

def switch_schedule(schedule, mult, switch):
    """ Apply LR multiplier before iteration "switch" """

    def f(e):
        s = schedule(e)
        if e < switch:
            return s * mult
        return s

    return f

def linear_schedule(args):
    def f(e):
        if e <= args.warmup:
            return e / args.warmup
        return max((e - args.iterations) / (args.warmup - args.iterations), 0)

    return f


def compare_date(date1: str, date2: str)->bool:
    """
    whether date2 is no earlier than date1
    :param date1:
    :param date2:
    :return:
    """
    date1 = date1.split(".")
    date2 = date2.split(".")
    if date1[0] == date2[0]:
        if int(date1[1]) <= int(date2[1]):
            return True
        else:
            return False
    else:
        cal_day = lambda x: int(x[0]) * 30 + int(x[1])
        if cal_day(date1) <= cal_day(date2):
            return True
        else:
            return False



###############################
######## metrics utils ########
###############################
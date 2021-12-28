#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: adaVAE.py
@author: ImKe at 2021/12/26
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
from src.adapters import *
import numpy as np
import pytorch_lightning as pl
from typing import Optional
import torch
from src.adapters.vae import *
import math
import fire
import os

def compute_loss(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta, use_adv_loss):
    """

    :param device:
    :param model: AdaVAEModel or VAEModel
    :param x_mask:
    :param x_tokens:
    :param y_mask:
    :param y_tokens:
    :param input_tokens:
    :param target_tokens:
    :param mask:
    :param loss_fn:
    :param beta:
    :param use_adv_loss:
    :return:
    """
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, x_mask=x_mask, x_tokens=x_tokens, y_mask=y_mask,
                    y_tokens=y_tokens)
    logits = outputs[0]
    regularization_loss = outputs[-1]
    if use_adv_loss:
        d_loss, g_loss = regularization_loss[0], regularization_loss[1]
    else:
        kl_loss = regularization_loss
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    if use_adv_loss:
        loss = ce_loss.mean() + beta * g_loss + d_loss
    else:
        kl_loss = kl_loss.mean()
        loss = ce_loss.mean() + beta * kl_loss

    return loss, ce_loss, regularization_loss


def train(
        num_epochs: int = 5,
        n_workers: int = 4,
        gpus: int = 1,
        precision: int = 32,
        patience: int = 5,
        adapter_size: Optional[int] = 64,
        lr: float = 2e-05,
        model_name: str = 'bert-base-multilingual-cased',
        train_file: str = './data/rusentiment/rusentiment_random_posts.csv',
        test_file: str = './data/rusentiment/rusentiment_test.csv',
        output_dir: str = './output_dir'
):
    pass
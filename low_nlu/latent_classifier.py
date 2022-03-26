#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: latent_classifier.py
@author: ImKe at 2022/3/20
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import torch
import torch.nn as nn
import sys, os
sys.path.append('../')
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from src.logger import Logger
from src.adapters.vae import *
from src.utils import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D


class AdaVAEforLatentClassification(nn.Module):
    def __init__(self, args, config, encoder, use_mean=False):
        super(AdaVAEforLatentClassification, self).__init__()
        self.encoder = encoder
        self.encoder.latent_representations = True

        self.num_labels = args.label_size

        self.classifier = nn.Linear(config.n_embd, self.num_labels)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.use_mean = use_mean

    def forward(self,
                input_ids=None,
                labels=None,
                past=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,):

        mean, _, _, representations = self.encoder(input_ids=input_ids,
                                                   past=past,
                                                   attention_mask=attention_mask,
                                                   token_type_ids=token_type_ids,
                                                   position_ids=position_ids,
                                                   head_mask=head_mask,
                                                   inputs_embeds=inputs_embeds)
        if self.use_mean:
            representations = mean
        representations = self.dropout(representations)

        logits = self.classifier(representations)

        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, representations) + outputs
        return outputs

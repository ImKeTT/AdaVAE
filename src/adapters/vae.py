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
from transformers import BertModel
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfOutput

from adapters.common import AdapterConfig


logging.basicConfig(level=logging.INFO)


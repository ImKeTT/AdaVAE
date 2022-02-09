#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: spacefusion.py
based on Optimus
"""
import sys
sys.path.append('../')
from src.adapters.vae import *
import numpy as np
import torch, copy, pdb
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary

from torch import nn


def set_trainable(module, value):
    for param in module.parameters():
        param.requires_grad = value


class SpaceFusion(AdaVAEModel):
    def __init__(self, args, config, AdapterConfig, sep_id, pad_id, add_input=False, add_attn=False, add_softmax=False,
                 attn_proj_vary=False, learn_prior=False, reg_loss="kld"):
        super(SpaceFusion, self).__init__(args, config, AdapterConfig, sep_id,
                                          add_input, add_attn, add_softmax, attn_proj_vary, learn_prior, reg_loss)

        self.transformer = Decoder(config, AdapterConfig, add_input, add_attn,
                                   attn_proj_vary)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.encoder = Encoder(config, AdapterConfig)

        # children = [v for v in self.encoder.layer.children()]  # list of 12 BertLayer

        # self.num_s2s_bert_layer = args.num_s2s_bert_layer
        # self.S2S_layers = nn.ModuleList(
        #     [copy.deepcopy(c) for c in children[-args.num_s2s_bert_layer:]])  # the last layer of encoder
        self.ix_turn_sep = sep_id
        # if args.freeze_bert:
        #     print('@' * 20 + f' freezing BERT {args.num_frozen_bert_layer} layers')
        #     for child in children[:args.num_frozen_bert_layer]:
        #         set_trainable(child, False)

        # add code here
        self.add_input = add_input
        self.add_attn = add_attn
        self.add_softmax = add_softmax
        self.attn_proj_vary = attn_proj_vary
        self.learn_prior = learn_prior
        self.reg_loss = reg_loss
        self.AdapterConfig = AdapterConfig
        self.CELoss = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="none")

        if self.learn_prior:
            self.encoder_prior = Encoder(config, AdapterConfig)

        if self.add_softmax:
            nz = config.n_embd
            self.lm_head_rep = Conv1D(config.vocab_size, nz)
            # self.lm_head_rep = LM_head_rep(nz, config.vocab_size)
        if self.reg_loss == "adversarial":
            self.discriminator = nn.Sequential(nn.Linear(config.n_embd, config.dis_emb),
                                               nn.ReLU(),
                                               nn.Linear(config.dis_emb, 1),
                                               nn.Softmax())

    def ids2speaker(self, ids):
        # 0 for speaker A, 1 for speaker B
        N, T = ids.shape
        speaker = np.zeros((N, T))
        sep = ids == self.ix_turn_sep
        for i in range(N):
            is_B = False  # start with speaker A
            for t in range(T):
                speaker[i, t] = int(is_B)
                if sep[i, t].item():
                    is_B = not is_B

        # make sure the final speaker is speaker B (so response is always speaker A)
        if not is_B:
            speaker = 1 - speaker

        return torch.LongTensor(speaker).to(ids.device)

    def forward(self,
        inputs_src=None,
        inputs_tgt=None,
        labels_tgt=None,
        past=None,
        src_attention_mask=None,
        tgt_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        from_prior=False,
        from_mean=False,
        return_vec=False,):  # [batch, time]
        # toggle config to get desired encoder output
        self.encoder.encoder.output_attentions = False
        self.encoder.encoder.output_hidden_states = True

        # AE encoder
        mask = (inputs_tgt > 0).float().to(inputs_src.device)
        posterior_mean, posterior_logvar = self.encoder(inputs_tgt, attention_mask=tgt_attention_mask)[:2]
        z_AE = self.reparameterize(posterior_mean, posterior_logvar)
        z_AE = z_AE.squeeze(1)

        # S2S encoder
        # mask = (inputs_src > 0).float()
        # speaker = self.ids2speaker(inputs_src)
        # outputs = self.encoder(inputs_src, attention_mask=src_attention_mask, token_type_ids=speaker)
        # _, _, all_layer_attn = outputs  # last_layer_attn, pooled, all_layer_attn = outputs
        # seq_z_prev = all_layer_attn[-self.num_s2s_bert_layer - 1]  # seq of z at layer 11 ()
        #
        # for s2s in self.S2S_layers:
        #     layer_outputs = s2s(seq_z_prev, attention_mask=src_attention_mask.unsqueeze(1).unsqueeze(1))
        #     seq_z_prev = layer_outputs[0]
        #
        # z_S2S = self.encoder.pooler(layer_outputs[0])
        # z_S2S, _ = self.connect(z_S2S)
        # z_S2S = z_S2S.squeeze(1)

        # S2S encoder
        mask = (inputs_src > 0).float()
        speaker = self.ids2speaker(inputs_src)
        posterior_mean, posterior_logvar = self.encoder(inputs_src, attention_mask=src_attention_mask, token_type_ids=speaker)[:2]
        z_S2S = self.reparameterize(posterior_mean, posterior_logvar)
        z_S2S = z_S2S.squeeze(1)

        if return_vec:
            return z_AE, z_S2S

        # interpolation/smoothness
        u = torch.FloatTensor(np.random.random((z_AE.shape[0], 1))).to(inputs_tgt.device)
        z_interp = u * z_AE + (1 - u) * z_S2S
        std = 0.1
        noise = torch.FloatTensor(np.random.normal(size=z_interp.shape) * std).to(z_interp.device)
        z_interp = z_interp + noise

        loss_rec = 0
        z_idx = 0
        for z in [z_AE, z_S2S, z_interp]:
            # pdb.set_trace()
            # past = z  # past = self.decoder.linear(z)
            transformer_outputs = self.transformer(labels_tgt,
                                       past=past,
                                       attention_mask=None,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds,
                                       representations=z)
            # outputs = self.decoder(input_ids=labels_tgt, past=past, labels=labels_tgt, label_ignore=self.pad_token_id)
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)

            # Perform masking
            if tgt_attention_mask is not None:
                att_mask = tgt_attention_mask.type(torch.bool)
                lm_logits = lm_logits.masked_select(att_mask.unsqueeze(-1))
                labels_tgt = labels_tgt.masked_select(att_mask)

            loss_rec_ = self.CELoss(lm_logits.view(-1, lm_logits.size(-1), labels_tgt))

            if z_idx == 1:
                loss_rec = loss_rec + 1.0 * loss_rec_
            else:
                loss_rec = loss_rec + loss_rec_
            z_idx += 1
        loss_rec = loss_rec / 3

        # fusion/regularization
        L_pull = self.dist_pair(z_AE, z_S2S)
        L_push = torch.stack([self.dist_batch(z) for z in [z_AE, z_S2S]]).min()
        loss_reg = (L_pull - L_push * 2) / np.sqrt(z.shape[-1])

        return loss_rec, loss_reg

    # def sent2latent(self, inputs_src, attn_mask):
    #     # toggle config to get desired encoder output
    #     self.encoder.encoder.output_attentions = False
    #     self.encoder.encoder.output_hidden_states = True
    #
    #     # S2S encoder
    #     mask = (inputs_src > 0).float()
    #     speaker = self.ids2speaker(inputs_src)
    #     outputs = self.encoder(inputs_src, attention_mask=attn_mask, token_type_ids=speaker)
    #
    #     _, _, all_layer_attn = outputs  # last_layer_attn, pooled, all_layer_attn = outputs
    #     # seq_z_prev = all_layer_attn[-2]     # seq of z at layer 11 ()
    #     # layer_outputs = self.S2S_layer(seq_z_prev, attention_mask=mask.unsqueeze(1).unsqueeze(1))
    #
    #     seq_z_prev = all_layer_attn[-self.num_s2s_bert_layer - 1]  # seq of z at layer 11 ()
    #     for s2s in self.S2S_layers:
    #         layer_outputs = s2s(seq_z_prev, attention_mask=mask.unsqueeze(1).unsqueeze(1))
    #         seq_z_prev = layer_outputs[0]
    #
    #     z_S2S = self.encoder.pooler(layer_outputs[0])
    #     z_S2S, _ = self.connect(z_S2S)
    #     z_S2S = z_S2S.squeeze(1)
    #
    #     return z_S2S

    def dist_pair(self, a, b):
        return F.pairwise_distance(a, b).mean()

    def dist_batch(self, vec):
        n = vec.shape[0]
        dmin = []
        for i in range(n):
            dd = F.pairwise_distance(vec[i:i + 1, :].repeat(n, 1), vec)
            dmin.append(dd.min())
        return torch.stack(dmin).mean()
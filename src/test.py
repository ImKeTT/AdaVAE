#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: test.py
@author: ImKe at 2022/1/28
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import numpy as np
import torch, math, time, os, argparse, copy, re
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from adapters.vae import *
from adaVAE import compute_loss
from utils import *
from collections import defaultdict
from adapters.common import AdapterConfig
from data import ConditionalGenerationDataset
import datetime

from torch.utils.data import Dataset, DataLoader
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()

# Default parameters are set based on single GPU training
parser.add_argument("--seed", type=int, default=42)

# parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae'])
parser.add_argument('--dataset', type=str, default='yelp_polarity', choices=['yelp_polarity, imdb_polariry'],
                    help="Dataset to use for training")

## mode options
parser.add_argument('--adapter_size', type=int, default=128,
                    help="Hidden size of GPT2 encoder/decoder adapter")
parser.add_argument('--latent_size', type=int, default=36,
                    help="Hidden size of latent code")
parser.add_argument('--encoder_n_layer', type=int, default=8,
                    help="attention layer number of GPT-2 encoder")
parser.add_argument('--decoder_n_layer', type=int, default=12,
                    help="attention layer number of GPT-2 decoder")
parser.add_argument('--class_num', type=int, default=2,
                    help="class number for controllable generation")
parser.add_argument('--label_emb_size', type=int, default=8,
                    help="label embedding size")
parser.add_argument('--adapter_scalar', type=str, default="1.0",
                    help="adapter scalar")
parser.add_argument('--ffn_option', type=str, default="parallel_ffn",
                    choices=['sequential', 'parallel_attn', 'parallel_ffn', 'pfeiffer'],
                    help="adapter type option")
parser.add_argument('--attn_mode', type=str, default="none",
                    choices=['prefix', 'adapter', 'lora', 'none'],
                    help="attention transfer type")
parser.add_argument('--reg_loss', type=str, default="kld",
                    choices=['kld', 'adversarial', 'symlog'],
                    help="regularization loss for latent space")

## testing paramters
parser.add_argument("--mode", type=str, help="Test mode", default="generate",
                    choices=['generate', 'interpolate', 'reconstruct', 'analogy', 'cal_interpolate'])
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size per GPU. Lists the schedule.')
parser.add_argument('--max_length', type=int, default=30,
                    help='max length of every input sentence')
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--out-dir', type=str, default='out')
parser.add_argument('--experiment', type=str, help="ckpt dirctory", default='out')
parser.add_argument('--adapter_init', type=str, default='bert', choices=['lora', 'bert', 'lisa', 'other'],
                    help="parameter initialization method for adapter layers.")
parser.add_argument('--workers', default=2, type=int, metavar='N',  help='number of data loading workers')
parser.add_argument("--total_sents", default=10, type=int, help="Total sentences to test recontruction/generation.")
parser.add_argument("--max_test_batch", default=10, type=int, help="Total sentence pairs to test interpolation/analogy.")
parser.add_argument("--num_interpolation_step", default=10, type=int)
parser.add_argument("--degree_to_target", type=float, default="1.0")
parser.add_argument("--max_val_batches", type=int, help="Max batch size number to test recontruction.", default=20000)
parser.add_argument("--latest_date", type=str, help="Latest date for model testing.", default="1.25")

## metrics
parser.add_argument('--au_delta', type=float, default=0.01,
                    help="threshold for activated unit calculation.")

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")


# KL cost annealing, increase beta from beta_0 to 1 in beta_warmup steps
parser.add_argument('--beta_0', default=1.00, type=float)
parser.add_argument('--beta_warmup', type=int, default=1000)
parser.add_argument('--kl_rate', type=float, default=0.0)

# cyc_vae parameters
parser.add_argument('--cycle', type=int, default=2000)

## trigger
parser.add_argument('--load', action="store_true")
parser.add_argument('--label_cond', action="store_true")
parser.add_argument('--save_all', action="store_true", help="save full parameters of the model")
parser.add_argument('--add_input', action="store_true")
parser.add_argument('--add_attn', action="store_true")
parser.add_argument('--add_softmax', action="store_true")
parser.add_argument('--attn_proj_vary', action="store_true")
parser.add_argument('--finetune_enc', action="store_true")
parser.add_argument('--test_model', action="store_true")

def generate(args, model, save_dir, label, bsz, tokenizer, device, parallel=False, topk=100, top_p=0.95):
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if parallel or bsz <= 1000:
        sents, _ = sample_sequence(model, args.max_length,
                                   torch.full([bsz, ], label).long().to(device),
                                   batch_size=bsz, top_k=topk, top_p=top_p,
                                   device=device, sample=True, eos_token=endoftext)
        sents = sents.tolist()
    else:
        sents = []
        assert bsz % 1000 == 0, "total sentence should be divided by 1000"
        partition = int(bsz/1000)
        for i in range(partition):
            sents_, _ = sample_sequence(model, args.max_length,
                                       torch.full([1000, ], label).long().to(device),
                                       batch_size=1000, top_k=topk, top_p=top_p,
                                       device=device, sample=True, eos_token=endoftext)
            sents.extend(sents_.tolist())
    # Sample sentences
    sentences_list = []
    for i in range(len(sents)):
        sent = sents[i]
        sent = sent[sent.index(endoftext) + 1:]

        if endoftext in sent:
            idx = sent.index(endoftext)
            sent = sent[:idx]

        sent = tokenizer.decode(sent).strip()
        sentences_list.append(sent)
    with open(f"{save_dir}/{bsz}-label{label}.txt", 'w') as f:
        for sentence in sentences_list:
            f.write(sentence + '\n')
    print(f"Finish generating {bsz} sentences with label {label}...")

def cal_interpolate(args, ada_config, model, tokenizer, device, eval_dataloader, num_interpolation_steps=10, top_k=100, top_p=0.5):
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    count = 0
    latent_codes = []
    sample_interval = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating interpolation"):
        with torch.no_grad():
            if sample_interval == 0 or sample_interval == args.total_sents:
                label = F.one_hot(torch.tensor(batch['y']),
                                  torch.tensor(ada_config.class_num)).float().to(device)
                x_ids, input_ids, attention_mask = tokenize(batch['x'], tokenizer, device, args)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, label_onehot=label, from_mean=True)
                latent_z = outputs[-2]
                latent_codes.append(latent_z)
                if sample_interval == 5:
                    latent_codes.append(latent_z)
                    sample_interval = 0
                    continue
            else:
                sample_interval += 1
                continue
        count += 1
        if count > args.total_sents:
            break
    result = defaultdict(str)
    num_steps = num_interpolation_steps
    for step in range(num_steps + 1):
        latent_z = latent_codes[0] + (latent_codes[1] - latent_codes[0]) * step * 1.0 / num_steps
        sents, _ = sample_sequence(model, args.max_length, latent_z.long().to(device),
                                       batch_size=1, top_k=top_k, top_p=top_p,
                                       device=device, sample=True, eos_token=endoftext)
        # Sample sentences
        sents = sents.tolist()
        sentences_list = []
        ## bsz == 1 only sample 1 sentence for each interpolation step
        sent = sents[0]
        sent = sent[sent.index(endoftext) + 1:]

        if endoftext in sent:
            idx = sent.index(endoftext)
            sent = sent[:idx]

        sent = tokenizer.decode(sent).strip()
        result[step] = sent

    return result



def interpolate(args, ada_config, model, tokenizer, device, batch_pair, num_interpolation_steps=10, top_k=100, top_p=0.5):
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    with torch.no_grad():
        label0 = F.one_hot(torch.tensor(batch_pair['y'][0]),
                          torch.tensor(ada_config.class_num)).float().to(device)
        x_ids, input_ids, attention_mask = tokenize(batch_pair['x'][0], tokenizer, device, args)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, label_onehot=label0, from_mean=True)
        latent_z1 = outputs[-2]

        label1 = F.one_hot(torch.tensor(batch_pair['y'][1]),
                           torch.tensor(ada_config.class_num)).float().to(device)
        x_ids, input_ids, attention_mask = tokenize(batch_pair['x'][1], tokenizer, device, args)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, label_onehot=label1, from_mean=True)
        latent_z2 = outputs[-2]
    result = defaultdict(str)
    num_steps = num_interpolation_steps + 1
    for step in range(num_steps + 1):
        latent_z = latent_z1 + (latent_z2 - latent_z1) * step * 1.0 / num_steps
        sents, _ = sample_sequence(model, args.max_length, latent_z.long().to(device),
                                       batch_size=1, top_k=top_k, top_p=top_p,
                                       device=device, sample=True, eos_token=endoftext)
        # Sample sentences
        sents = sents.tolist()
        sentences_list = []
        ## bsz == 1 only sample 1 sentence for each interpolation step
        sent = sents[0]
        sent = sent[sent.index(endoftext) + 1:]

        if endoftext in sent:
            idx = sent.index(endoftext)
            sent = sent[:idx]

        sent = tokenizer.decode(sent).strip()
        result[step] = sent
        print(sent)

    return result

def analogy(args, ada_config, model, tokenizer, device, batch_triple, top_k=100, top_p=0.5):
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    with torch.no_grad():
        label0 = F.one_hot(torch.tensor(batch_triple['y'][0]),
                           torch.tensor(ada_config.class_num)).float().to(device)
        x_ids, input_ids, attention_mask = tokenize(batch_triple['x'][0], tokenizer, device, args)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, label_onehot=label0, from_mean=True)
        latent_z1 = outputs[-2]

        label1 = F.one_hot(torch.tensor(batch_triple['y'][1]),
                           torch.tensor(ada_config.class_num)).float().to(device)
        x_ids, input_ids, attention_mask = tokenize(batch_triple['x'][1], tokenizer, device, args)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, label_onehot=label1, from_mean=True)
        latent_z2 = outputs[-2]

        label2 = F.one_hot(torch.tensor(batch_triple['y'][2]),
                           torch.tensor(ada_config.class_num)).float().to(device)
        x_ids, input_ids, attention_mask = tokenize(batch_triple['x'][2], tokenizer, device, args)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, label_onehot=label2, from_mean=True)
        latent_z3 = outputs[-2]

    result = defaultdict(str)
    latent_z = latent_z3 + args.degree_to_target * (latent_z2 - latent_z1)
    sents, _ = sample_sequence(model, args.max_length, latent_z.long().to(device),
                               batch_size=1, top_k=top_k, top_p=top_p,
                               device=device, sample=True, eos_token=endoftext)
    # Sample sentences
    sents = sents.tolist()
    sentences_list = []
    ## bsz == 1 only sample 1 sentence for each interpolation step
    sent = sents[0]
    sent = sent[sent.index(endoftext) + 1:]

    if endoftext in sent:
        idx = sent.index(endoftext)
        sent = sent[:idx]

    sent = tokenizer.decode(sent).strip()
    result[0] = sent
    print(sent)

    return result

def cal_rec(args, ada_config, model, tokenizer, device, eval_dataloader, save_dir=None, top_k=100, top_p=0.95):
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    rec_sents = []
    with tqdm(total=min(len(eval_dataloader), args.max_val_batches), desc="Evaluating Model") as pbar:
        for i, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                label = F.one_hot(torch.tensor(batch['y']),
                                  torch.tensor(ada_config.class_num)).float().to(device)
                x_ids, input_ids, attention_mask = tokenize(batch['x'], tokenizer, device, args)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, label_onehot=label, from_mean=True)
                latent_z = outputs[-2]
                sents, _ = sample_sequence(model, args.max_length, torch.tensor(batch['y']),
                                           z=latent_z,
                                           batch_size=args.batch_size, top_k=top_k, top_p=top_p,
                                           device=device, sample=False, eos_token=endoftext)
                # Sample sentences
                sents = sents.tolist()
                for i in range(len(sents)):
                    sent = sents[0]
                    sent = sent[sent.index(endoftext) + 1:]

                    if endoftext in sent:
                        idx = sent.index(endoftext)
                        sent = sent[:idx]

                    sent = tokenizer.decode(sent).strip()
                    rec_sents.append(sent)
            i += 1
            if i > args.max_val_batches:
                break
            pbar.update(1)
    if not save_dir is None:
        with open(f"{save_dir}/reconstruction.txt", 'w') as f:
            for sent in rec_sents:
                f.write(sent + "\n")
    else:
        return rec_sents

def val_step(args, model, val_loader, ada_config, tokenizer, device):
    model.eval()
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    n_words_bpe = 0
    n_words = 0
    n_examples = 0
    cnt_au = 0
    logp_sum = 0.0
    reg_loss_sum = 0.0

    mu_batch_list, logvar_batch_list = [], []
    neg_entropy = 0.
    max_val_batches = args.max_val_batches

    print("Validation loop.         Batches: %d" % len(val_loader))
    print("Validation loop. max_val_batches: %d" % max_val_batches)

    with tqdm(total=min(len(val_loader), max_val_batches), desc="Evaluating Model") as pbar:
        for i, val_data_dict in enumerate(val_loader):
            with torch.no_grad():
                val_x_ids, val_input_ids, val_attention_mask = tokenize(val_data_dict['x'], tokenizer, device, args)
                val_label_onehot = F.one_hot(torch.tensor(val_data_dict['y']),
                                         torch.tensor(ada_config.class_num)).float().to(device)

                val_loss, val_ce_loss, val_reg_loss, val_mu, val_lv = compute_loss(device, model, val_x_ids,
                                                                                   val_input_ids, val_attention_mask,
                                                                                   val_label_onehot, loss_fn, 1.0, 0.0, args.reg_loss)
                # else:
                #     loss, ce_loss, kl_loss = compute_loss_ae(device, AdaVAE, x_mask, x_tokens, y_mask, y_tokens,
                #                                              input_tokens, target_tokens, mask, loss_fn, 1.0)
            """
            calculate text perplexity
            """
            target_tokens = val_x_ids
            if len(target_tokens.size()) == 1:
                target_tokens = target_tokens.unsqueeze(0)
            n, l = target_tokens.size()

            text = target_tokens.tolist()
            tokens = [t[:t.index(endoftext) + 1] if endoftext in t else t for t in text]
            words_bpe = sum([len(t) for t in tokens])
            n_words_bpe += words_bpe
            logprob = val_ce_loss.mean()


            logp_sum += logprob * words_bpe

            n_words_bpe += len(text)

            ctext = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
            ctext = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in ctext]
            ctext = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in
                     ctext]
            words = sum([len(
                [t for t in re.split('("|\'|!|\?|\.|,|:| |\n|’|“|”|;|\(|\)|`)', s) if t != ' ' and t != '']) for
                s in ctext])
            n_words += words

            reg_loss_sum += val_reg_loss.item()

            """
            calculate mutual information (mi) Stage 1
            """
            n_examples += n
            nz = val_mu.size(1)
            # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
            neg_entropy += (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + val_lv).sum(-1)).sum().item()
            mu_batch_list += [val_mu.cpu()]
            logvar_batch_list += [val_lv.cpu()]

            """
            compute the number of active units (au) Stage 1
            """
            if cnt_au == 0:
                means_sum = val_mu.sum(dim=0, keepdim=True)
            else:
                means_sum = means_sum + val_mu.sum(dim=0, keepdim=True)
            cnt_au += val_mu.size(0)


            if i > max_val_batches:
                break
            pbar.update(1)

    neg_entropy = neg_entropy / n_examples
    mean_mean = means_sum / cnt_au
    loss_bpe = logp_sum / n_words_bpe
    ppl_bpe = round(math.exp(min(logp_sum / n_words_bpe, 100)), 3)
    ppl_word = round(math.exp(min(logp_sum / n_words, 100)), 3)
    reg = reg_loss_sum / len(val_loader)

    """
    calculate mi and au Stage 2
    """
    n_examples = 0
    log_qz = 0.
    for i in tqdm(range(len(mu_batch_list)), desc="Evaluating MI, Stage 2"):
        ###############
        # get z_samples
        ###############
        mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()

        # [z_batch, 1, nz]
        with torch.no_grad():
            z_samples = model.reparameterize(mu, logvar).unsqueeze(1)

        z_samples = z_samples.view(-1, 1, nz)
        n_examples += z_samples.size(0)

        ###############
        # compute density
        ###############
        # [1, x_batch, nz]
        # mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        # indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
        indices = np.arange(len(mu_batch_list))
        mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
        logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
        x_batch, nz = mu.size()

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)


    log_qz /= n_examples
    mi = (neg_entropy - log_qz).item()

    """
    calculate au Stage 2
    """
    cnt_au = 0
    with tqdm(total=min(len(val_loader), max_val_batches), desc="Evaluating AU, Stage 2") as pbar:
        for i, val_data_dict in enumerate(val_loader):
            with torch.no_grad():
                val_x_ids, val_input_ids, val_attention_mask = tokenize(val_data_dict['x'], tokenizer, device, args)
                val_label_onehot = F.one_hot(torch.tensor(val_data_dict['y']),
                                         torch.tensor(ada_config.class_num)).float().to(device)

                val_loss, val_ce_loss, val_reg_loss, val_mu, val_lv = compute_loss(device, model, val_x_ids,
                                                                                   val_input_ids, val_attention_mask,
                                                                                   val_label_onehot, loss_fn, 1.0, 0.0, args.reg_loss)
            if cnt_au == 0:
                var_sum = ((val_mu - mean_mean) ** 2).sum(dim=0)
            else:
                var_sum = var_sum + ((val_mu - mean_mean) ** 2).sum(dim=0)
            cnt_au += val_mu.size(0)

            if i > max_val_batches:
                break
            pbar.update(1)

    # (nz)
    au_var = var_sum / (cnt_au - 1)
    n_au = (au_var >= args.au_delta).sum().item()

    print('val loss    : %.4f' % loss_bpe)
    print('val ppl_bpe : %.4f' % ppl_bpe)
    print('val ppl_word: %.4f' % ppl_word)
    print('val reg_loss: %.4f' % reg)
    print('val MI      : %.4f' % mi)
    print('val AU      : %.4f' % n_au)


def test(args):
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    device = torch.device(args.gpu if gpu else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='/home/tuhq/.cache/torch/transformers')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='/home/tuhq/.cache/torch/transformers')
    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    config = GPT2Config()
    ada_config = AdapterConfig(hidden_size=768,
                               adapter_size=args.adapter_size,
                               adapter_act='relu',
                               adapter_initializer_range=1e-2,
                               label_emb_size=args.label_emb_size,
                               latent_size=args.latent_size,
                               class_num=args.class_num,
                               encoder_n_layer=args.encoder_n_layer,
                               decoder_n_layer=args.decoder_n_layer,
                               init=args.adapter_init,
                               adapter_scalar=args.adapter_scalar,
                               ffn_option=args.ffn_option,
                               attn_mode=args.attn_mode,
                               attn_option='none',
                               mid_dim=30,
                               attn_bn=25,
                               prefix_dropout=0.1,
                               tune_enc=args.finetune_enc)
    AdaVAE = AdaVAEModel(config, ada_config, add_input=args.add_input, add_attn=args.add_attn,
                         add_softmax=args.add_softmax,
                         attn_proj_vary=args.attn_proj_vary, learn_prior=False, reg_loss=args.reg_loss,
                         label_cond=args.label_cond)
    ## load pre-trained weights
    init_para_frompretrained(AdaVAE.transformer, gpt2_model.transformer, share_para=False)
    init_para_frompretrained(AdaVAE.encoder, gpt2_model.transformer, share_para=False)
    AdaVAE.lm_head.weight = gpt2_model.lm_head.weight

    AdaVAE.eval()
    ## load ckpt
    # experiment = args.experiment
    latest_date = args.latest_date
    for experiment in os.listdir(args.out_dir):
        if compare_date(latest_date, experiment.split("_")[-1]):
            save_folder = os.path.join(args.out_dir, experiment)
            if not os.path.exists(os.path.join(save_folder, 'model_latest.pt')):
                continue
            print('Loading model weights...')
            state = torch.load(os.path.join(save_folder, 'model_latest.pt'))  # , map_location='cpu' model_latest.pt
            if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
                state_copy = copy.copy(state)
                keys = state_copy.keys()
                for k in keys:
                    state[k.replace('module.', '')] = state.pop(k)
            ## load trained parameters
            if not args.save_all:
                model_dict = AdaVAE.state_dict()
                additional_dict = {k: v for k, v in state.items() if k in model_dict}
                model_dict.update(additional_dict)
                AdaVAE.load_state_dict(model_dict)
            else:
                AdaVAE.load_state_dict(state)
            AdaVAE = AdaVAE.to(device)
            mode = args.mode
            assert mode in ['generate', 'interpolate', 'reconstruct', 'analogy', 'cal_interpolate'], "get invalid test mode.."

            args.dataset = '_'.join(experiment.split("_")[:2])
            test_loader = DataLoader(
                ConditionalGenerationDataset.from_file(f"../data/{args.dataset}/test.txt"),
                batch_size=args.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=args.workers)
            val_loader = DataLoader(
                ConditionalGenerationDataset.from_file(f"../data/{args.dataset}/valid.txt"),
                batch_size=args.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=args.workers)

            if args.test_model:
                print("test set")
                val_step(args, AdaVAE, test_loader, ada_config, tokenizer, device)
                print("valid set")
                val_step(args, AdaVAE, val_loader, ada_config, tokenizer, device)

            save_dir = os.path.join(save_folder, "test_texts")
            os.makedirs(save_dir, exist_ok=True)
            if mode == "generate":
                for label in range(args.class_num):
                    generate(args, AdaVAE, save_dir, label, args.total_sents, tokenizer, device, topk=100, top_p=0.95)
                print(f"Done generating {args.total_sents} for {args.class_num} class(es).")

            elif mode == "interpolate":
                test_loader = DataLoader(
                    ConditionalGenerationDataset.from_file(f"../data/{args.dataset}/test.txt"),
                    batch_size=2,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=args.workers)
                result = []
                with tqdm(total=min(len(test_loader), args.max_test_batch),
                          desc=f"Interpolation for {args.max_test_batch} pairs") as pbar:
                    for i, batch in enumerate(test_loader):
                        result_ = interpolate(args, ada_config, AdaVAE, tokenizer, device, batch,
                                             num_interpolation_steps=args.num_interpolation_step)
                        result.append(result_)
                        if i >args.max_test_batch:
                            break
                        pbar.update(1)
                return result

            elif mode == "reconstruct":
                cal_rec(args, ada_config, AdaVAE, tokenizer, device, test_loader, save_dir)
                print(f"Done reconstructing for test data set.")

            elif mode == "analogy":
                test_loader = DataLoader(
                    ConditionalGenerationDataset.from_file(f"../data/{args.dataset}/test.txt"),
                    batch_size=3,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=args.workers)
                result = []
                with tqdm(total=min(len(test_loader), args.max_test_batch),
                          desc=f"Analogy for {args.max_test_batch} pairs") as pbar:
                    for i, batch in enumerate(test_loader):
                        result_ = analogy(args, ada_config, AdaVAE, tokenizer, device, batch)
                        result.append(result_)
                        if i > args.max_test_batch:
                            break
                        pbar.update(1)
                return result

            elif mode == "cal_interpolate":
                pass

if __name__=="__main__":
    # args = parser.parse_args()
    args = parser.parse_args('--mode reconstruct '
                             '--out-dir out --label_cond --add_attn --total_sents 5000 --max_length 50 --batch_size 128'.split())
    test(args)
#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: test.py
@author: ImKe at 2022/1/28
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import numpy as np
import torch, math, time, os, argparse, copy
from tqdm import tqdm
import torch.nn.functional as F
from adapters.vae import *
from utils import *
from collections import defaultdict
from adapters.common import AdapterConfig
from data import ConditionalGenerationDataset
import datetime

from torch.utils.data import Dataset, DataLoader
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D


parser = argparse.ArgumentParser()

# Default parameters are set based on single GPU training
parser.add_argument("--seed", type=int, default=42)

# parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae'])
parser.add_argument('--dataset', type=str, default='imdb_polarity', choices=['yelp_polarity, imdb_polariry'],
                    help="Dataset to use for training")

## mode options
parser.add_argument('--adapter_size', type=int, default=256,
                    help="Hidden size of GPT2 encoder/decoder adapter")
parser.add_argument('--latent_size', type=int, default=768,
                    help="Hidden size of latent code")
parser.add_argument('--encoder_n_layer', type=int, default=6,
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
parser.add_argument('--attn_mode', type=str, default="prefix",
                    choices=['prefix', 'adapter', 'lora', 'none'],
                    help="attention transfer type")
parser.add_argument('--reg_loss', type=str, default="kld",
                    choices=['kld', 'adversarial', 'symlog'],
                    help="regularization loss for latent space")

## testing paramters
parser.add_argument("--mode", type=str, help="Testing mode", default="generate",
                    choices=['generate', 'interpolate', 'reconstruct', 'analogy', 'cal_interpolate'])
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size per GPU. Lists the schedule.')
parser.add_argument('--seq-lens', nargs='+', type=int, default=[30],
                    help='seq length per sample. Lists the schedule.')
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
parser.add_argument("--max_val_batches", type=int, help="Max batch size number to test recontruction.", default=2000)

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

def generate(args, model, save_dir, label, bsz, tokenizer, device, topk=100, top_p=0.5):
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    sents, _ = sample_sequence(model, args.max_length,
                               torch.full([bsz, ], label).long().to(device),
                               batch_size=bsz, top_k=topk, top_p=top_p,
                               device=device, sample=True, eos_token=endoftext)
    # Sample sentences
    sents = sents.tolist()
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

def cal_rec(args, ada_config, model, tokenizer, device, eval_dataloader, save_dir=None, top_k=100, top_p=0.5):
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

                sents, _ = sample_sequence(model, args.max_length, latent_z.long().to(device),
                                           batch_size=1, top_k=top_k, top_p=top_p,
                                           device=device, sample=True, eos_token=endoftext)
                # Sample sentences
                sents = sents.tolist()
                sentences_list = []
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
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
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
                         attn_proj_vary=args.attn_proj_vary, learn_prior=args.learn_prior, reg_loss=args.reg_loss,
                         label_cond=args.label_cond)
    init_para_frompretrained(AdaVAE.transformer, gpt2_model.transformer, share_para=True)
    init_para_frompretrained(AdaVAE.encoder, gpt2_model.transformer, share_para=True)

    ## load ckpt
    # experiment = args.experiment
    latest_date = 24
    for experiment in os.listdir(args.out_dir):
        if int(experiment.split(".")[-1])>=latest_date:
            save_folder = os.path.join(args.out_dir, experiment)
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
                del model_dict
            else:
                AdaVAE.load_state_dict(state)
                del state
            mode = args.mode
            assert mode in ['generate', 'interpolate', 'reconstruct', 'analogy', 'cal_interpolate'], "get invalid test mode.."

            if mode == "generate":
                save_dir = os.path.join(save_folder, "test_texts")
                for label in range(args.class_num):
                    generate(args, AdaVAE, save_dir, label, args.total_sents, tokenizer, device, topk=100, top_p=0.5)
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
                test_loader = DataLoader(
                    ConditionalGenerationDataset.from_file(f"../data/{args.dataset}/test.txt"),
                    batch_size=args.batch_size,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=args.workers)
                cal_rec(args, ada_config, AdaVAE, tokenizer, device, test_loader)
                print(f"Done reconstructing {args.total_sents} for test data set.")

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
    args = parser.parse_args()
    test(args)
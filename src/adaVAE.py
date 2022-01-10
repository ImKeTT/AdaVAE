#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: adaVAE.py
@author: ImKe at 2021/12/26
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
from adapters import *
import numpy as np
import collections
import torch, math, time, os, argparse, re, copy
from logger import Logger
from tensorboardX import SummaryWriter
from tqdm import tqdm
from typing import Optional
import torch.nn.functional as F
from adapters.vae import *
from utils import *
from adapters.common import AdapterConfig
from data import ConditionalGenerationDataset

from torch.utils.data import Dataset, DataLoader
from apex.optimizers import FusedAdam
from apex import amp
from apex.fp16_utils import FP16_Optimizer
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
# from transformers.configuration_gpt2 import GPT2Config
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D


devices = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = devices

parser = argparse.ArgumentParser()
parser.add_argument('experiment', type=str)

# Default parameters are set based on single GPU training
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument("--seed", type=int, default=42)

# parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae'])
parser.add_argument('--iterations', type=int, default=30000 * 3)
parser.add_argument('--dataset', type=str, default='yelp_polarity', choices=['yelp_polarity, imdb_polariry'],
                    help="Dataset to use for training")
parser.add_argument('--warmup', type=int, default=10000,
                    help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")

parser.add_argument('--adapter_size', type=int, default=256,
                    help="Hidden size of GPT2 encoder/decoder adapter")
parser.add_argument('--class_num', type=int, default=2,
                    help="class number for controllable generation")
parser.add_argument('--label_emb_size', type=int, default=8,
                    help="label embedding size")
parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1],
                    help='batch size per GPU. Lists the schedule.')
parser.add_argument('--seq-lens', nargs='+', type=int, default=[30],
                    help='seq length per sample. Lists the schedule.')
parser.add_argument('--max_length', type=int, default=25,
                    help='max length of every input sentence')
parser.add_argument('--switch-time', type=float, default=0,
                    help="Percentage of iterations to spend on short sequence training.")
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--out-dir', type=str, default='out')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers')
# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")

parser.add_argument('--fp16', action='store_true', help="Train using FP16?")
parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)

# KL cost annealing, increase beta from beta_0 to 1 in beta_warmup steps
parser.add_argument('--beta_0', default=1.00, type=float)
parser.add_argument('--beta_warmup', type=int, default=10000)
# cyc_vae parameters
parser.add_argument('--cycle', type=int, default=30000)
## trigger
parser.add_argument('--load', action="store_true")
parser.add_argument('--label_cond', action="store_true")
parser.add_argument('--save_all', action="store_true", help="save full parameters of the model")
parser.add_argument('--add_input', action="store_true")
parser.add_argument('--add_attn', action="store_true")
parser.add_argument('--add_softmax', action="store_true")
parser.add_argument('--attn_proj_vary', action="store_true")
parser.add_argument('--adv_loss', action="store_true")
parser.add_argument('--learn_prior', action="store_true")

# args = parser.parse_args('test --batch-sizes 1 --seq-lens 1024 '
#                          '--add_input --learn_prior --fp16'.split()) # wi.12.proj_vary_beta_cvae

def compute_loss(device, model, x_tokens, input_tokens, att_mask, label_onehot, loss_fn, beta, use_adv_loss):
    """

    :param device:
    :param model:
    :param input_tokens: input word ids
    :param mask: input mask
    :param x_tokens: target sequence
    :param loss_fn:
    :param beta: weight of regularization loss
    :param use_adv_loss: use adversarial loss for WAE
    :return:
    """
    input_tokens = input_tokens.to(device)
    att_mask = att_mask.to(device)
    x_tokens = x_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=att_mask, label_onehot=label_onehot, from_mean=True)
    logits = outputs[0]
    regularization_loss = outputs[-1]
    if use_adv_loss:
        d_loss, g_loss = regularization_loss[0], regularization_loss[1]
    else:
        kl_loss = regularization_loss
    num_logits = logits.size(-1)

    # Perform masking
    if att_mask is not None:
        att_mask = att_mask.type(torch.bool)
        att_mask = att_mask.to(device)
        logits = logits.masked_select(att_mask.unsqueeze(-1))
        x_tokens = x_tokens.masked_select(att_mask)

    ## x_token is target tokens
    ce_loss = loss_fn(logits.view(-1, num_logits), x_tokens.view(-1))
    if use_adv_loss:
        loss = ce_loss.mean() + beta * g_loss + d_loss
    else:
        kl_loss = kl_loss.mean()
        loss = ce_loss.mean() + beta * kl_loss

    return loss, ce_loss, regularization_loss


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

def train_step(device, model, optimizer, x_tokens, input_tokens, att_mask, label_onehot, loss_fn, beta, use_adv_loss, model_type):
    # output = []
    # if model_type == 'ae_vae_fusion':
    #     optimizer.zero_grad()
    #     loss, ce_loss, kl_loss = compute_loss_ae(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
    #                                           target_tokens, mask, loss_fn, beta)
    #     with amp.scale_loss(loss, optimizer) as scaled_loss:
    #         scaled_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)  # max_grad_norm=1.0
    #     # loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
    #     optimizer.step()
    #     output.append((loss.item(), ce_loss.mean().item(), kl_loss.item()))

    optimizer.zero_grad()
    loss, ce_loss, reg_loss = compute_loss(device, model, x_tokens, input_tokens, att_mask, label_onehot, loss_fn,
                                          beta, use_adv_loss)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)  # max_grad_norm=1.0
    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
    optimizer.step()
    # output.append((loss.item(), ce_loss.mean().item(), reg_loss.item()))

    return loss.item(), ce_loss.mean().item(), reg_loss.item()

def train(args):
    # if args.model_type == 'cvae':
    #     args.learn_prior = True
    # else:
    #     args.learn_prior = False

    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        # print('Setting GPUs {}'.format(args.device))
        print('Using GPU devices {}'.format(devices))
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # logging
    save_folder = os.path.join(args.out_dir, args.experiment)
    os.makedirs(os.path.join(save_folder, 'ckpt/model'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'ckpt/opt'), exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
    # importlib.reload(logging)
    logging = Logger(os.path.join(save_folder, 'train.log'))
    # logging.basicConfig(filename=os.path.join(save_folder, 'train.log'),
    #                     level=logging.INFO, format='%(asctime)s--- %(message)s', filemode='w')
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))

    print('Loading models...')
    # cache_dir = os.path.join(args.out_dir, 'model_cache')
    # os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Hack to allow tokenizing longer sequences.
    # tokenizer.max_len = int(1e12)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    print('gpt2_params:', num_params(gpt2_model))  # gpt2: 124439808

    ## GPT2 config and adapter config
    config = GPT2Config()
    """
    GPT2Config {
      "activation_function": "gelu_new",
      "attn_pdrop": 0.1,
      "bos_token_id": 50256,
      "embd_pdrop": 0.1,
      "eos_token_id": 50256,
      "initializer_range": 0.02,
      "layer_norm_epsilon": 1e-05,
      "model_type": "gpt2",
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_inner": null,
      "n_layer": 12,
      "n_positions": 1024,
      "reorder_and_upcast_attn": false,
      "resid_pdrop": 0.1,
      "scale_attn_by_inverse_layer_idx": false,
      "scale_attn_weights": true,
      "summary_activation": null,
      "summary_first_dropout": 0.1,
      "summary_proj_to_labels": true,
      "summary_type": "cls_index",
      "summary_use_proj": true,
      "transformers_version": "4.12.0",
      "use_cache": true,
      "vocab_size": 50257
    }
    """
    ada_config = AdapterConfig(hidden_size=768,
                               adapter_size=args.adapter_size,
                               adapter_act='relu',
                               adapter_initializer_range=1e-2,
                               label_emb_size=args.label_emb_size,
                               class_num=args.class_num)

    AdaVAE = AdaVAEModel(config, ada_config, add_input=args.add_input, add_attn=args.add_attn, add_softmax=args.add_softmax,
                   attn_proj_vary=args.attn_proj_vary, learn_prior=args.learn_prior, adv_loss=args.adv_loss, label_cond=args.label_cond)
    init_para_frompretrained(AdaVAE.transformer, gpt2_model.transformer, share_para=True)
    init_para_frompretrained(AdaVAE.encoder, gpt2_model.transformer, share_para=False)

    ## freeze all prarameters expect the ones in adapters
    # AdaVAE = freeze_all_parameters(AdaVAE)
    # AdaVAE.transformer = unfreeze_GPT2_adapters(AdaVAE.transformer, Cond_GPT2Adapter)
    # AdaVAE.encoder = unfreeze_GPT2_adapters(AdaVAE.encoder, Cond_GPT2Adapter)

    if args.learn_prior:
        init_para_frompretrained(AdaVAE.encoder_prior, AdaVAE.encoder, share_para=True)
        AdaVAE.encoder_prior.averageSelfAttention.attention_weights = AdaVAE.encoder.averageSelfAttention.attention_weights
    AdaVAE.lm_head.weight = gpt2_model.lm_head.weight
    if AdaVAE.add_softmax:
        AdaVAE.lm_head_rep = Conv1D(*gpt2_model.lm_head.weight.size())
        # AdaVAE.lm_head_rep = LM_head_rep(*gpt2_model.lm_head.weight.size()[::-1])
    print('AdaVAE params:', num_params(AdaVAE))  #

    # fix pre-trained parameters before certain iterations
    tuning_all_after_iters = 6000
    tuning_all = False
    for name, parameter in AdaVAE.named_parameters():
        new_pars = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj', 'attn_proj', 'Nu_fc1', 'Nu_fc2',
                    'lm_head_rep']
        if args.adv_loss:
            new_pars.append('discriminator')
        if args.label_cond:
            new_pars.append('label_embedding')

        if not any([True if n in name else False for n in new_pars]):
            parameter.requires_grad = False
        # print((name, parameter.requires_grad))
    print('AdaVAE params with gradients:', num_params(AdaVAE))

    print('Setup data...')
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens)
    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedule'
    args.switch_time = 0
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0
    print('Batch schedule', batch_schedule)
    train_loader = DataLoader(
        ConditionalGenerationDataset.from_file(f"../data/{args.dataset}/train.txt"),
        batch_size=batch_schedule[cur_b_schedule][0],
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers)
    test_loader = DataLoader(
        ConditionalGenerationDataset.from_file(f"../data/{args.dataset}/test.txt"),
        batch_size=batch_schedule[-1][0],
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers)
    print('Done.')

    ###
    val_loader = test_loader
    ###

    print('Wrapping models and optimizers...')
    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                  int(args.iterations * args.switch_time))
    AdaVAE = AdaVAE.to(device)
    AdaVAE.train()

    optimizer = AdamW(AdaVAE.parameters(), lr=args.lr, correct_bias=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    AdaVAE, optimizer = amp.initialize(AdaVAE, optimizer, opt_level=args.fp16_opt_level)

    ## load ckpt
    if args.load:
        print('Loading model weights...')
        state = torch.load(os.path.join(save_folder,'ckpt/model',
                                        'model_0000048.pt'))  # , map_location='cpu' model_latest.pt
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
        # optimizer.load_state_dict(torch.load(os.path.join(save_folder, 'ckpt/opt',
        #                                                   'optimizer_0000048.pt')))
        # gc.collect()
    print('Done.')

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    print('Done.')

    print('Begin training iterations')
    logging.info("Begin training iterations")
    max_val_batches = 20000  # max num. of val batches
    logging.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    optimizer.zero_grad()
    beta = args.beta_0
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def val_step(val_loader):
        AdaVAE.eval()

        n_words_bpe = 0
        n_words = 0
        logp_sum = 0.0
        reg_loss_sum = 0.0

        logging.info("Validation loop.         Batches: %d" % len(val_loader))
        logging.info("Validation loop. max_val_batches: %d" % max_val_batches)

        with tqdm(total=min(len(val_loader), max_val_batches)) as pbar:
            for i, val_data_dict in enumerate(val_loader):
                with torch.no_grad():
                    val_x_ids, val_input_ids, val_attention_mask = tokenize(val_data_dict['x'], tokenizer, device, args)
                    val_label_onehot = F.one_hot(torch.tensor(val_data_dict['y']),
                                             torch.tensor(ada_config.class_num)).float().to(device)

                    val_loss, val_ce_loss, val_reg_loss = compute_loss(device, AdaVAE, val_x_ids, val_input_ids, val_attention_mask,
                                                           val_label_onehot, loss_fn, 1.0, args.adv_loss)
                    # else:
                    #     loss, ce_loss, kl_loss = compute_loss_ae(device, AdaVAE, x_mask, x_tokens, y_mask, y_tokens,
                    #                                              input_tokens, target_tokens, mask, loss_fn, 1.0)
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

                if i > max_val_batches:
                    break
                pbar.update(1)

        loss_bpe = logp_sum / n_words_bpe
        ppl_bpe = round(math.exp(min(logp_sum / n_words_bpe, 100)), 3)
        ppl_word = round(math.exp(min(logp_sum / n_words, 100)), 3)
        reg = reg_loss_sum / len(val_loader)

        v_writer.add_scalar('loss', loss_bpe, num_iters)
        v_writer.add_scalar('ppl_bpe', ppl_bpe, num_iters)
        v_writer.add_scalar('ppl_word', ppl_word, num_iters)
        v_writer.add_scalar('reg_loss', reg, num_iters)
        logging.info('val loss    : %.4f' % loss_bpe)
        logging.info('val ppl_bpe : %.4f' % ppl_bpe)
        logging.info('val ppl_word: %.4f' % ppl_word)
        logging.info('val reg_loss: %.4f' % reg)
        for cl in range(ada_config.class_num):
            bsz = 5
            sents, _ = sample_sequence(AdaVAE, args.max_length,
                                    torch.full([bsz,], cl).long().to(device),
                                    batch_size=bsz, top_k=100, top_p=0.95,
                                    device=device, sample=True, eos_token=endoftext)
            # Sample sentences
            print("-" * 50)
            print(f"Sentences with label {cl}:")
            sents = sents.tolist()
            for i in range(len(sents)):
                sent = sents[i]
                sent = sent[sent.index(endoftext) + 1:]

                if endoftext in sent:
                    idx = sent.index(endoftext)
                    sent = sent[:idx]

                sent = tokenizer.decode(sent).strip()
                print(sent)

        AdaVAE.train()

    # todo: add parameter saving & testing
    while num_iters < args.iterations:
        # Run epoch
        st = time.time()

        # Training
        print('Training loop. Batches:', len(train_loader))
        logging.info('\n----------------------------------------------------------------------')
        logging.info("Training loop.       Batches: %d" % len(train_loader))

        # train_iter = iter(train_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(train_iter)
        with tqdm(total=len(train_loader)) as pbar:
            for i, data_dict in enumerate(train_loader):
                x_ids, input_ids, attention_mask = tokenize(data_dict['x'], tokenizer, device, args)
                label_onehot = F.one_hot(torch.tensor(data_dict['y']),
                                         torch.tensor(ada_config.class_num)).float().to(device)

                if num_iters % args.cycle >= args.cycle - args.beta_warmup:
                    beta = min(1.0, beta + (1. - args.beta_0) / args.beta_warmup)

                if not tuning_all and num_iters >= tuning_all_after_iters:
                    AdaVAE.transformer = unfreeze_GPT2_adapters(AdaVAE.transformer, Cond_GPT2Adapter)
                    AdaVAE.encoder = unfreeze_GPT2_adapters(AdaVAE.encoder, Cond_GPT2Adapter)
                    for name, parameter in AdaVAE.named_parameters():
                        print((name, parameter.requires_grad))
                    #     parameter.requires_grad = True
                    print('AdaVAE params with gradients:', num_params(AdaVAE))
                    tuning_all = True

                loss, ce_loss, reg_loss = train_step(device, AdaVAE, optimizer, x_ids, input_ids, attention_mask,
                                    label_onehot, loss_fn, beta, args.adv_loss, args.model_type)

                lr = scheduler.get_last_lr()[0]
                # Log to Tensorboard
                t_writer.add_scalar('loss', loss, num_iters)
                t_writer.add_scalar('ppl', math.exp(min(ce_loss, 10)), num_iters)
                t_writer.add_scalar('lr', lr, num_iters)
                t_writer.add_scalar('iter_time', time.time() - st, num_iters)
                t_writer.add_scalar('kl', reg_loss, num_iters)
                t_writer.add_scalar('beta', beta, num_iters)

                # if args.model_type == 'ae_vae_fusion':
                #     loss, ce_loss, kl_loss = output[0]
                #     # Log to Tensorboard
                #     t_writer.add_scalar('ae_loss', loss, num_iters)
                #     t_writer.add_scalar('ae_kl', kl_loss, num_iters)

                st = time.time()
                end = num_iters >= args.iterations

                if args.warmup != -1:
                    scheduler.step()

                if end:
                    break
                num_iters += 1
                pbar.update(1)

                if num_iters % args.cycle == 0:
                    beta = args.beta_0
                    logging.info('KL annealing restart')

                if num_iters % 10000 == 0:
                    val_step(test_loader)

                if (num_iters + 1) % 30000 == 0:
                    print('Saving model...')
                    logging.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
                    logging.info("Saving model...")
                    logging.info('\n------------------------------------------------------')

                    if args.save_all:
                        save_orderdict = AdaVAE.state_dict()
                    else:
                        save_orderdict = collections.OrderedDict()
                        for name, parameter in AdaVAE.named_parameters():
                            if parameter.requires_grad:
                                save_orderdict[name] = parameter
                    torch.save(save_orderdict,
                               os.path.join(save_folder, 'ckpt/model',
                                            'model_' + '{:07d}'.format(num_iters) + '.pt'))
                    # torch.save(optimizer.state_dict(),
                    #            os.path.join(save_folder, 'ckpt/opt',
                    #                         'optimizer_' + '{:07d}'.format(num_iters) + '.pt'))

                # if args.switch_time > 0 and num_iters == int(args.iterations * args.switch_time):
                #     print('Switch to long sequence training')
                #     logging.info("Switch to long sequence training")
                #     cur_b_schedule += 1
                #     train_loader, val_loader, test_loader = prepare_dataset(
                #         args.data_dir, args.dataset, tokenizer,
                #         batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
                #         batch_schedule[-1][0], batch_schedule[-1][1],
                #         batch_schedule[-1][0], batch_schedule[-1][1],
                #         make_test=True,
                #         num_workers=args.workers, data_type=args.data_type
                #     )
        if not end:
            e += 1
            logging.info("Training loop. The ith epoch completed: %d" % e)

    torch.save(AdaVAE.state_dict(), os.path.join(save_folder, 'model_latest.pt'))
    print('Training complete.')
    logging.info("Training complete.")

if __name__=="__main__":
    args = parser.parse_args('ex0110_as64 --batch-sizes 128 --max_length 25 --add_attn --label_cond --adapter_size 64'.split())
    train(args)

# class AdaGPT2VAE(pl.LightningModule):
#     """
#     This class is used to fine-tune a pre-trained model from Transformers using PyTorch-Lightning.
#     It is optimized towards models with SentencePiece tokenizers to be converted into LibTorch + SentencePiece
#     for C++ deployments.
#
#     :arg model_name: str, pre-trained model name for a model from Transformers
#     :arg n_classes: int, number of classes in the classification problem
#     :arg max_length: int, maximum length of tokens that the model uses
#     :arg lr: float, learning rate for fine-tuning
#     :arg eps: float, epsilon parameter for Adam optimizer
#     """
#     def __init__(self,
#                  model_name: str,
#                  n_classes: int,
#                  max_length: int = 100,
#                  lr: float = 2e-05,
#                  eps: float = 1e-08):
#         super(AdaGPT2VAE, self).__init__()
#         self.max_length = max_length
#         self.lr = lr
#         self.eps = eps
#
#         config = GPT2Config.from_pretrained(model_name,
#                                             num_labels=n_classes,
#                                             output_attentions=False,
#                                             output_hidden_states=False,
#                                             torchscript=True)
#         self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_fast=False)
#         self.model = AdaVAEModel(config)
#         self.criteria = nn.CrossEntropyLoss()
#         gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
#         print('gpt2_params:', num_params(gpt2_model))  # gpt2: 124439808
#         config = GPT2Config()
#         init_para_frompretrained(self.model.transformer, gpt2_model.transformer, share_para=True)
#         init_para_frompretrained(self.model.encoder, gpt2_model.transformer, share_para=False)
#
#     def forward(self, x_ids, x_mask, input_ids, attention_mask=None, label_emb=None):
#         output = self.model(
#         x_mask=x_mask,
#         x_tokens=x_ids,
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         label_emb=label_emb
#         )
#         ## lm_logits, presents, (all hidden_states), (attentions), (regularization_loss)
#         return output
#
#     def tokenize(self, texts):
#         x_tokenized = self.tokenizer(texts, padding=True,
#                                      truncation=True, return_tensors='pt')
#         input_ids = x_tokenized['input_ids'].to(self.device)
#         attention_mask = x_tokenized['attention_mask'].to(self.device)
#         x_ids = x_tokenized['input_ids'][:, 1:].to(self.device)
#         x_mask = x_tokenized['attention_mask'][:, 1:].to(self.device)
#         return x_ids, x_mask, input_ids, attention_mask
#
#     def compute(self, batch, cn: int):
#         """
#
#         :param batch:
#         :param cn: class number
#         :return:
#         """
#         x = batch['x']
#         y = batch['y']
#         y = F.one_hot(y, cn)
#
#         x_ids, x_mask, input_ids, attention_mask = self.tokenize(x)
#         output = self(x_ids, x_mask, input_ids, attention_mask, y)
#         lm_logits = output[0]
#         regularization_loss = output[-1]
#         return lm_logits, regularization_loss
#
#     def training_step(self, batch, batch_nb):
#         loss, logits = self.compute(batch)
#         return {'loss': loss}
#
#     def validation_step(self, batch, batch_nb):
#         loss, logits = self.compute(batch)
#
#         y = batch['y']
#         a, y_hat = torch.max(logits, dim=1)
#
#         return {'val_loss': loss, 'val_acc': torch.tensor(val_acc)}
#
#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
#
#         self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
#         self.log('avg_val_acc', avg_val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
#
#     def test_step(self, batch, batch_nb):
#         loss, logits = self.compute(batch)
#
#         y = batch['y']
#         a, y_hat = torch.max(logits, dim=1)
#         test_acc = accuracy_score(y_hat.cpu(), y.cpu())
#
#         return {'test_acc': torch.tensor(test_acc)}
#
#     def test_epoch_end(self, outputs):
#         avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
#
#         self.log('avg_test_acc', avg_test_acc, on_epoch=True, prog_bar=True, sync_dist=True)
#
#     def configure_optimizers(self):
#         return torch.optim.Adam([p for p in self.parameters() if p.requires_grad],
#                                 lr=self.lr, eps=self.eps)
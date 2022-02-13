#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: run_vae_ctrl_gen.py
@author: ImKe at 2022/2/6
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""

from ctrl_gen import CARA, Ctrl_AdaVAE
import datetime, os, copy, math, time, collections, argparse, nltk, json, sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
from src.logger import Logger
from src.adapters.vae import *
from src.utils import *
from apex import amp
from src.adapters.common import AdapterConfig
from src.data import ConditionalGenerationDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()

# Default parameters are set based on single GPU training
parser.add_argument('--lr', type=float, default=5e-7)
parser.add_argument("--seed", type=int, default=42)

# parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae'])
parser.add_argument('--iterations', type=int, default=10000 * 3)
parser.add_argument('--dataset', type=str, default='yelp_polarity', choices=['yelp_polarity', 'imdb_polarity'],
                    help="Dataset to use for training")
parser.add_argument('--warmup', type=int, default=1000,
                    help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")

## mode options
parser.add_argument('--adapter_size', type=int, default=128,
                    help="Hidden size of GPT2 encoder/decoder adapter")
parser.add_argument('--latent_size', type=int, default=32,
                    help="Hidden size of latent code")
parser.add_argument('--encoder_n_layer', type=int, default=6,
                    help="attention layer number of GPT-2 encoder")
parser.add_argument('--decoder_n_layer', type=int, default=12,
                    help="attention layer number of GPT-2 decoder")
parser.add_argument('--label_size', type=int, default=2,
                    help="class number for controllable generation")
# parser.add_argument('--label_emb_size', type=int, default=8,
#                     help="label embedding size")
parser.add_argument('--adapter_scalar', type=str, default="1.0",
                    help="adapter scalar")
parser.add_argument('--ffn_option', type=str, default="parallel_ffn",
                    choices=['sequential', 'parallel_attn', 'parallel_ffn', 'pfeiffer'],
                    help="adapter type option")
parser.add_argument('--attn_mode', type=str, default="none",
                    choices=['prefix', 'adapter', 'lora', 'none'],
                    help="attention transfer type")

## training paramters
parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1],
                    help='batch size per GPU. Lists the schedule.')
parser.add_argument('--eval_batch_size', type=int, default=100,
                    help='batch size per GPU. Lists the schedule.')
parser.add_argument('--seq-lens', nargs='+', type=int, default=[30],
                    help='seq length per sample. Lists the schedule.')
parser.add_argument('--max_length', type=int, default=25,
                    help='max length of every input sentence')
parser.add_argument('--block_size', type=int, default=50,
                    help='max length of generated sentences')
parser.add_argument('--switch-time', type=float, default=0,
                    help="Percentage of iterations to spend on short sequence training.")
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--out-dir', type=str, default='train_out')
parser.add_argument('--load_dir', type=str, default='../src/out')
parser.add_argument('--experiment', type=str,
                    default=None)
parser.add_argument('--eval_output_dir', type=str, default='eval_out')
parser.add_argument('--adapter_init', type=str, default='lora',
                    choices=['lora', 'bert', 'lisa', 'other'],
                    help="parameter initialization method for adapter layers.")
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers')

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")

parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)

# loss weights
parser.add_argument('--beta_cls', default=1.00, type=float)
parser.add_argument('--beta_latent', default=1.00, type=float)
parser.add_argument('--beta_warmup', type=int, default=1000)

## generation
parser.add_argument('--top_k', default=5, type=int)
parser.add_argument('--top_p', default=0.0, type=float)
parser.add_argument('--temperature', default=0.5, type=float)

## trigger
parser.add_argument('--load', action="store_true")
parser.add_argument('--do_train', action="store_true")
parser.add_argument('--add_input', action="store_true")
parser.add_argument('--add_attn', action="store_true")
parser.add_argument('--add_mem', action="store_true")
parser.add_argument('--add_softmax', action="store_true")
parser.add_argument('--finetune_enc', action="store_true")
parser.add_argument('--finetune_dec', action="store_true")
parser.add_argument('--attn_proj_vary', action="store_true")
parser.add_argument('--save_all', action="store_true", help="save full parameters of the model")



def compute_loss(device, model, x_tokens, input_tokens, att_mask, cond_labels):
    input_tokens = input_tokens.to(device)
    att_mask = att_mask.to(device)
    x_tokens = x_tokens.to(device)

    loss_dict, acc_dict = model(input_ids=input_tokens, tgt_seq_ids=x_tokens,
                      cond_labels=cond_labels, attention_mask=att_mask)
    return loss_dict, acc_dict

def train_step(device, model, optimizer, x_tokens, input_tokens, att_mask, cond_labels):
    optimizer.zero_grad()
    loss_dict, acc_dict = compute_loss(device, model, x_tokens, input_tokens, att_mask, cond_labels)

    loss = loss_dict['loss'].mean()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)  # max_grad_norm=1.0
    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
    optimizer.step()

    return loss_dict, acc_dict


def evaluate(args, model, tokenizer, logging, eval_dataloader, max_val_batches, eval_output_dir, iter, device):
    os.makedirs(eval_output_dir, exist_ok=True)
    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d" % len(eval_dataloader))
    logging.info("  Batch size = %d" % args.eval_batch_size)
    logging.info("  Num steps = %d" % (min(max_val_batches, len(eval_dataloader) // args.eval_batch_size)))

    outputs = {
        'sampled_cond_labels': None,
        'cond_labels': None,
        'tgt_seq_ids': None,
        'generated': None,
        'at_generated': None,
        'cg_generated': None,
        'pred_cls': None,
        'pred_ge_cls': None,
        'pred_at_cls': None,
        'pred_cg_cls': None,
    }
    with tqdm(total=min(len(eval_dataloader), max_val_batches), desc="Evaluating Model") as pbar:
        for bi, batch in enumerate(eval_dataloader):

            # Data
            # input_seq_ids, tgt_seq_ids, tokenized_text_lengths, cond_labels = batch
            # max_len_values, _ = tokenized_text_lengths.max(0)
            # input_seq_ids = input_seq_ids[:, :max_len_values[0]]
            # tgt_seq_ids = tgt_seq_ids[:, :max_len_values[1]]
            # input_seq_ids = input_seq_ids.to(args.device)
            # tgt_seq_ids = tgt_seq_ids.to(args.device)
            # cond_labels = cond_labels.to(args.device)
            # input_mask = torch.where(
            #     torch.arange(max_len_values[0].item()).unsqueeze(0).repeat(input_seq_ids.size(0), 1).type_as(
            #         tokenized_text_lengths).to(args.device)
            #     < tokenized_text_lengths[:, 0].unsqueeze(1).to(args.device), torch.ones_like(input_seq_ids),
            #     torch.zeros_like(input_seq_ids))

            ## Data
            tgt_seq_ids, input_seq_ids, input_mask = tokenize(batch['x'], tokenizer, device, args)
            cond_labels = torch.tensor(batch['y']).to(device)

            # Model
            with torch.no_grad():
                result = model(input_ids=input_seq_ids, tgt_seq_ids=tgt_seq_ids, cond_labels=cond_labels,
                                   attention_mask=input_mask)
            if bi == 0:
                for key in outputs.keys():
                    outputs[key] = result[key].cpu().tolist()
            else:
                for key in outputs.keys():
                    outputs[key].extend(result[key].cpu().tolist())

            if bi > max_val_batches:
                break
            pbar.update(1)

        # compute accuracies and store in results
    acc = np.mean(np.array(np.array(outputs['pred_cls']) == np.array(outputs['cond_labels']), dtype=np.float))
    acc_ge = np.mean(np.array(np.array(outputs['pred_ge_cls']) == np.array(outputs['cond_labels']), dtype=np.float))
    acc_at = np.mean(
        np.array(np.array(outputs['pred_at_cls']) == np.array(outputs['sampled_cond_labels']), dtype=np.float))
    acc_cg = np.mean(
        np.array(np.array(outputs['pred_cg_cls']) == np.array(outputs['sampled_cond_labels']), dtype=np.float))
    metrics = {'acc': acc, 'acc_ge': acc_ge, 'acc_at': acc_at, 'acc_cg': acc_cg}

    # dump generated outputs to file.
    json.dump(outputs, open(
        os.path.join(eval_output_dir, "outputs_{}.json".format(iter) if iter is not None else "outputs.json"), 'w'))

    # compute BLEU
    bos_token_id = eos_token_id = pad_token_id = tokenizer.encode('<|endoftext|>')[0]

    generated_ids = []
    generated_text = []
    for g in outputs['generated']:
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        g = g[:g.index(eos_token_id)] if eos_token_id in g else g
        g = g[:g.index(pad_token_id)] if pad_token_id in g else g
        g_text = tokenizer.decode(g, clean_up_tokenization_spaces=True)
        generated_ids.append(g)
        generated_text.append(g_text)

    tgt_seq_ids = []
    tgt_seq_text = []
    for g in outputs['tgt_seq_ids']:
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        g = g[:g.index(eos_token_id)] if eos_token_id in g else g
        g = g[:g.index(pad_token_id)] if pad_token_id in g else g
        g_text = tokenizer.decode(g, clean_up_tokenization_spaces=True)
        tgt_seq_ids.append(g)
        tgt_seq_text.append(g_text)

    at_generated_ids = []
    at_generated_text = []
    for g in outputs['at_generated']:
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        g = g[:g.index(eos_token_id)] if eos_token_id in g else g
        g = g[:g.index(pad_token_id)] if pad_token_id in g else g
        g_text = tokenizer.decode(g, clean_up_tokenization_spaces=True)
        at_generated_ids.append(g)
        at_generated_text.append(g_text)

    cg_generated_ids = []
    cg_generated_text = []
    for g in outputs['cg_generated']:
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        if g and g[0] in [eos_token_id, bos_token_id]:
            g = g[1:]
        g = g[:g.index(eos_token_id)] if eos_token_id in g else g
        g = g[:g.index(pad_token_id)] if pad_token_id in g else g
        g_text = tokenizer.decode(g, clean_up_tokenization_spaces=True)
        cg_generated_ids.append(g)
        cg_generated_text.append(g_text)

    f = open(os.path.join(eval_output_dir, "reconstruction{}.txt".format(('_' + str(iter)) if iter is not None else '')), 'w')
    f.write('\n'.join([g + '\n' + t for g, t in zip(generated_text, tgt_seq_text)]))
    fat = open(os.path.join(eval_output_dir,
                            "attribute_transfer{}.txt".format(('_' + str(iter)) if iter is not None else '')), 'w')
    fat.write('\n'.join([g + '\n' + t for g, t in zip(at_generated_text, tgt_seq_text)]))
    fcg = open(os.path.join(eval_output_dir,
                            "conditional_generation{}.txt".format(('_' + str(iter)) if iter is not None else '')), 'w')
    fcg.write('\n'.join(cg_generated_text))

    rec_bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references=[[nltk.word_tokenize(t)] for t in tgt_seq_text],
                                                     hypotheses=[nltk.word_tokenize(g) for g in generated_text])

    at_bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references=[[nltk.word_tokenize(t)] for t in tgt_seq_text],
                                                    hypotheses=[nltk.word_tokenize(g) for g in at_generated_text])

    cg_generated_text_subset = cg_generated_text[:500]  # use a subset, otherwise it takes a long time to compute.
    cg_bleu = nltk.translate.bleu_score.corpus_bleu(
        list_of_references=[[nltk.word_tokenize(t) for t in tgt_seq_text] for _ in
                            range(len(cg_generated_text_subset))],
        hypotheses=[nltk.word_tokenize(g) for g in cg_generated_text_subset])

    cg_self_bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references=[
        [nltk.word_tokenize(t) for t in cg_generated_text_subset[:i] + cg_generated_text_subset[i + 1:]]
        for i in range(len(cg_generated_text_subset))],
                                                         hypotheses=[nltk.word_tokenize(g) for g in
                                                                     cg_generated_text_subset])

    metrics['rec_bleu'] = rec_bleu
    metrics['at_bleu'] = at_bleu
    metrics['cg_bleu'] = cg_bleu
    metrics['cg_self_bleu'] = cg_self_bleu

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    writer = open(output_eval_file, "w")
    logging.info("***** Eval results, global steps*****")
    for key, value in metrics.items():
        logging.info("  %s = %s" % (key, str(value)))
        writer.write("%s = %s\n"  % (key, str(value)))

    return metrics

def main(args):
    now = datetime.datetime.now()
    date = f"{now.month}.{now.day}"
    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        # print('Setting GPUs {}'.format(args.device))
        # print('Using GPU devices {}'.format(devices))
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)


    config = GPT2Config()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='/home/tuhq/.cache/torch/transformers')
    tokenizer.pad_token = tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='/home/tuhq/.cache/torch/transformers')
    ada_config = AdapterConfig(hidden_size=768,
                               adapter_size=args.adapter_size,
                               adapter_act='relu',
                               adapter_initializer_range=1e-2,
                               latent_size=args.latent_size,
                               class_num=args.label_size,
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
                               tune_enc=False,
                               tune_dec=False) ## two-stage training, should employ plain GPT-2 decoder/encoder + adapters

    AdaVae_encoder = Encoder(config, ada_config)
    AdaVae_decoder = Decoder(config, ada_config, args.add_input, args.add_attn, args.add_mem, attn_proj_vary=False)
    # AdaVae_average_attn = AverageSelfAttention(config.n_embd, ada_config)
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")


    model = Ctrl_AdaVAE(args, AdaVae_encoder, AdaVae_decoder, endoftext, config, ada_config, add_attn=args.add_attn)

    ## load pre-trained weights
    init_para_frompretrained(model.transformer, gpt2_model.transformer, share_para=True)
    init_para_frompretrained(model.encoder, gpt2_model.transformer, share_para=True)
    model.lm_head.weight = gpt2_model.lm_head.weight

    ## load ckpt
    print('Loading model weights...')
    experiment = args.experiment
    load_folder = os.path.join(args.load_dir, experiment)
    if args.add_attn and args.add_mem:
        final_folder = f"{args.dataset}_{date}_add_attn_mem"
    elif args.add_attn:
        final_folder = f"{args.dataset}_{date}_add_attn"
    elif args.add_mem:
        final_folder = f"{args.dataset}_{date}_add_mem"
    save_folder = os.path.join(args.out_dir, final_folder)
    os.makedirs(save_folder, exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
    logging_file = f"{args.dataset}_CtrlGen.log"
    logging = Logger(os.path.join(save_folder, logging_file))
    state = torch.load(os.path.join(load_folder, 'model_latest.pt'))  # , map_location='cpu' model_latest.pt
    if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
        state_copy = copy.copy(state)
        keys = state_copy.keys()
        for k in keys:
            state[k.replace('module.', '')] = state.pop(k)


    ## load trained parameters
    if not args.save_all:
        model_dict = model.state_dict()
        additional_dict = {k: v for k, v in state.items() if k in model_dict}
        model_dict.update(additional_dict)
        model.load_state_dict(model_dict) ## only loads adapters and latent connectors from ckpt
    else:
        model.load_state_dict(state)
    model = model.to(device)

    model_params = num_params(model)
    logging.info(f'model params: {model_params}')

    # fix pre-trained parameters before certain iterations
    args.warmup = args.beta_warmup = int(args.iterations / 6)
    for name, parameter in model.named_parameters():
        new_pars = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj', 'attn_proj', 'Nu_fc1', 'Nu_fc2',
                    'lm_head_rep']
        pars_ctrl = ['label_embedding', 'latent_generator', 'linear', 'latent_classifier', 'latent_discriminator',
                     'conv1', 'classifier']
        new_pars.extend(pars_ctrl)


        if not any([True if n in name else False for n in new_pars]):
            parameter.requires_grad = False

    decoder_unfreeze_modules = [GPT2Adapter]
    encoder_unfreeze_modules = [GPT2Adapter]
    if ada_config.attn_mode == "prefix":
        pass

    ## deal with encoder/decoder parameter settings
    if args.finetune_enc or args.finetune_dec:
        if args.finetune_enc and args.finetune_dec:
            for _, parameter in model.encoder.named_parameters():
                parameter.requires_grad = True
            for _, parameter in model.transformer.named_parameters():
                parameter.requires_grad = True
        elif args.finetune_enc:
            for _, parameter in model.encoder.named_parameters():
                parameter.requires_grad = True
            model.transformer = unfreeze_GPT2_adapters(model.transformer, decoder_unfreeze_modules)
        elif args.finetune_dec:
            for _, parameter in model.transformer.named_parameters():
                parameter.requires_grad = True
            model.encoder = unfreeze_GPT2_adapters(model.encoder, encoder_unfreeze_modules)
    else:
        model.encoder = unfreeze_GPT2_adapters(model.encoder, encoder_unfreeze_modules)
        model.transformer = unfreeze_GPT2_adapters(model.transformer, decoder_unfreeze_modules)

    for name, parameter in model.named_parameters():
        print((name, parameter.requires_grad))
    model_params_with_gradients = num_params(model)
    logging.info(f'model params with gradients:{model_params_with_gradients}')
    if args.finetune_enc or args.finetune_dec:
        logging.info('Trainable parameters %d / %d= %.4f' % (model_params_with_gradients, model_params,
                                                             model_params_with_gradients / model_params))
    else:
        logging.info('Additional parameters %d / %d = %.4f' % (model_params_with_gradients, model_params,
                                                               model_params_with_gradients / (
                                                                       model_params - model_params_with_gradients)))

    logging.info('Setup data...')
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens)
    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedules'
    args.switch_time = 0
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0
    logging.info('Batch schedule')
    logging.info(batch_schedule)
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
    val_loader = DataLoader(
        ConditionalGenerationDataset.from_file(f"../data/{args.dataset}/valid.txt"),
        batch_size=batch_schedule[-1][0],
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers)
    logging.info('Done.')

    logging.info('Wrapping models and optimizers...')
    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                  int(args.iterations * args.switch_time))
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    ## load ckpt
    if args.load:
        logging.info('Loading model weights...')
        state = torch.load(os.path.join(save_folder, 'model_latest.pt'))  # , map_location='cpu' model_latest.pt
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        ## load trained parameters
        if not args.save_all:
            model_dict = model.state_dict()
            additional_dict = {k: v for k, v in state.items() if k in model_dict}
            model_dict.update(additional_dict)
            model.load_state_dict(model_dict)
            del model_dict
        else:
            model.load_state_dict(state)
            del state

    logging.info('Done.')

    logging.info("Begin training iterations")
    max_val_batches = 5  # max num. of val batches
    logging.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    optimizer.zero_grad()

    def val_step(val_loader):
        model.eval()
        """
        n_words_bpe = 0
        n_words = 0
        val_loss = []
        val_acc_enc_dis = []
        val_acc_gen = []
        val_acc_enc_cls = []
        val_acc_cls = []
        logp_sum = 0.0

        logging.info("Validation loop.         Batches: %d" % len(val_loader))
        logging.info("Validation loop. max_val_batches: %d" % max_val_batches)

        with tqdm(total=min(len(val_loader), max_val_batches), desc="Evaluating Model") as pbar:
            for i, val_data_dict in enumerate(val_loader):
                with torch.no_grad():
                    val_x_ids, val_input_ids, val_attention_mask = tokenize(val_data_dict['x'], tokenizer, device, args)
                    val_labels = torch.tensor(val_data_dict['y']).to(device)

                    val_loss_dict, val_acc_dict = compute_loss(device, model, val_x_ids,
                                                               val_input_ids,
                                                               val_attention_mask, val_labels)
                    # else:
                    #     loss, ce_loss, kl_loss = compute_loss_ae(device, model, x_mask, x_tokens, y_mask, y_tokens,
                    #                                              input_tokens, target_tokens, mask, loss_fn, 1.0)

                val_ce_loss = val_loss_dict['loss_rec'].mean().item()
                val_loss.append(val_loss_dict['loss'].mean().item())
                val_acc_enc_dis.append(val_acc_dict['acc_encode_z_dis'].mean().item())
                val_acc_gen.append(val_acc_dict['acc_gen_z_dis'].mean().item())
                val_acc_enc_cls.append(val_acc_dict['acc_encode_z_cls'].mean().item())
                val_acc_cls.append(val_acc_dict['acc_cls'].mean().item())


                ## calculate text perplexity
                
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


                if i > max_val_batches:
                    break
                pbar.update(1)

        loss_bpe = logp_sum / n_words_bpe
        ppl_bpe = round(math.exp(min(logp_sum / n_words_bpe, 100)), 3)
        ppl_word = round(math.exp(min(logp_sum / n_words, 100)), 3)


        v_writer.add_scalar('loss', np.mean(val_loss), num_iters)
        v_writer.add_scalar('ce_loss', loss_bpe, num_iters)
        v_writer.add_scalar('ppl_bpe', ppl_bpe, num_iters)
        v_writer.add_scalar('ppl_word', ppl_word, num_iters)
        v_writer.add_scalar('acc_enc_dis', np.mean(val_acc_enc_dis))
        v_writer.add_scalar('acc_enc_cls', np.mean(val_acc_enc_cls))
        v_writer.add_scalar('acc_gen', np.mean(val_acc_gen))
        v_writer.add_scalar('acc_cls', np.mean(val_acc_cls))

        logging.info('val loss       : %.4f' % np.mean(val_loss))
        logging.info('val ce_loss    : %.4f' % loss_bpe)
        logging.info('val ppl_bpe    : %.4f' % ppl_bpe)
        logging.info('val ppl_word   : %.4f' % ppl_word)
        logging.info('val acc_enc_dis: %.4f' % np.mean(val_acc_enc_dis))
        logging.info('val acc_enc_cls: %.4f' % np.mean(val_acc_enc_cls))
        logging.info('val acc_gen    : %.4f' % np.mean(val_acc_gen))
        logging.info('val acc_cls    : %.4f' % np.mean(val_acc_cls))
        """
        evaluate(args, model, tokenizer, logging, val_loader, max_val_batches,
                 os.path.join(args.eval_output_dir, final_folder), num_iters, device)

        model.train()

    if args.do_train:
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
                    cond_labels = torch.tensor(data_dict['y']).to(device)

                    if args.warmup != -1:
                        scheduler.step()

                    loss_dict, acc_dict = train_step(device, model, optimizer, x_ids, input_ids, attention_mask, cond_labels)

                    lr = scheduler.get_last_lr()[0]

                    loss = loss_dict['loss'].mean().item()
                    ce_loss = loss_dict['loss_rec'].mean().item()
                    loss_enc = loss_dict['loss_encoder'].mean().item()
                    loss_lsc = loss_dict['loss_lsc'].mean().item()
                    loss_lsd = loss_dict['loss_lsd'].mean().item()
                    loss_lsg = loss_dict['loss_lsg'].mean().item()
                    loss_cls = loss_dict['loss_cls'].mean().item()
                    acc_enc_z_dis = acc_dict['acc_encode_z_dis'].mean().item()
                    acc_gen_z_dis = acc_dict['acc_gen_z_dis'].mean().item()
                    acc_enc_z_cls = acc_dict['acc_encode_z_cls'].mean().item()
                    acc_cls = acc_dict['acc_cls'].mean().item()

                    # Log to Tensorboard
                    t_writer.add_scalar('loss', loss, num_iters)
                    t_writer.add_scalar('loss_rec', ce_loss, num_iters)
                    t_writer.add_scalar('ppl', math.exp(min(ce_loss, 10)), num_iters)
                    t_writer.add_scalar('loss_enc', loss_enc, num_iters)
                    t_writer.add_scalar('loss_lsc', loss_lsc, num_iters)
                    t_writer.add_scalar('loss_lsd', loss_lsd, num_iters)
                    t_writer.add_scalar('loss_lsg', loss_lsg, num_iters)
                    t_writer.add_scalar('loss_cls', loss_cls, num_iters)
                    t_writer.add_scalar('acc_enc_z_dis', acc_enc_z_dis, num_iters)
                    t_writer.add_scalar('acc_gen_z_dis', acc_gen_z_dis, num_iters)
                    t_writer.add_scalar('acc_enc_z_cls', acc_enc_z_cls, num_iters)
                    t_writer.add_scalar('acc_cls', acc_cls, num_iters)
                    t_writer.add_scalar('lr', lr, num_iters)
                    t_writer.add_scalar('iter_time', time.time() - st, num_iters)

                    st = time.time()
                    end = num_iters >= args.iterations

                    if end:
                        break
                    num_iters += 1
                    pbar.update(1)

                    if num_iters % 5000 == 0:
                        logging.info("test set")
                        val_step(test_loader)
                        logging.info("validation set")
                        val_step(val_loader)

                    if (num_iters + 1) % 6000 == 0:
                        logging.info('Saving model...')
                        logging.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
                        logging.info("Saving model...")
                        logging.info('\n------------------------------------------------------')

                        if args.save_all:
                            save_orderdict = model.state_dict()
                        else:
                            save_orderdict = collections.OrderedDict()
                            for name, parameter in model.named_parameters():
                                if parameter.requires_grad:
                                    save_orderdict[name] = parameter
                        torch.save(save_orderdict,
                                   os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + '.pt'))
            if not end:
                e += 1
                logging.info("Training loop. The ith epoch completed: %d" % e)

        if args.save_all:
            save_orderdict = model.state_dict()
        else:
            save_orderdict = collections.OrderedDict()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    save_orderdict[name] = parameter
        torch.save(save_orderdict, os.path.join(save_folder, 'model_latest.pt'))
        logging.info('Training complete.')

    ## evaluate: generate; evaluate...
    else:
        model.eval()
        evaluate(args, model, tokenizer, logging, val_loader, max_val_batches,
                 os.path.join(args.eval_output_dir, f"{args.dataset}_{date}"), 0, device)

if __name__=="__main__":
    args = parser.parse_args()
    # args = parser.parse_args('--batch-sizes 50 --max_length 64 --add_mem --do_train --adapter_size 128 --latent_size 32 --experiment '
    #                          'yelp_polarity_iter6000_as128_scalar1.0_add_mem_beta1.0_reg-kld_attn_mode-none_ffn_option-parallel_ffn_enc_layer-8_dec_layer-12_zdim-32_zrate-10.0_sd-42_2.12'.split())
    main(args)
#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: run_spacefusion_gen.py
@author: ImKe at 2022/2/9
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import numpy as np
import collections
import torch, math, time, os, argparse, re, copy, sys
from logger import Logger
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from spacefusion import SpaceFusion
sys.path.append('../')
from src.adapters.vae import *
from src.utils import *
from src.adapters.common import AdapterConfig
from src.data import DialogGenerationDataset
import datetime

from torch.utils.data import Dataset, DataLoader
from apex.optimizers import FusedAdam
from apex import amp
from apex.fp16_utils import FP16_Optimizer
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D


## tokenize for dialog generation task
def tokenize(context, response, tokenizer, device, args):

    response_tokenized = tokenizer(response, padding=True, truncation=True, return_tensors='pt',
                                   max_length=args.max_length)
    context_tokenized = tokenizer(context, padding=True, truncation=True, return_tensors='pt')

    inputs_src = context_tokenized['input_ids'].to(device)
    src_attention_mask = context_tokenized['attention_mask'].to(device)
    labels_tgt = response_tokenized['input_ids'].to(device)
    inputs_tgt = labels_tgt
    tgt_attention_mask =  response_tokenized['attention_mask'].to(device)

    return inputs_src, inputs_tgt, labels_tgt, src_attention_mask, tgt_attention_mask


def compute_loss(device, model, inputs_src, inputs_tgt, labels_tgt, src_attention_mask, tgt_attention_mask, beta, kl_rate):
    inputs_src, inputs_tgt, labels_tgt, src_attention_mask, tgt_attention_mask = \
        inputs_src.to(device), inputs_tgt.to(device), labels_tgt.to(device), src_attention_mask.to(device), \
        tgt_attention_mask.to(device)

    loss_rec, loss_reg = model(inputs_src=inputs_src, inputs_tgt=inputs_tgt, labels_tgt=labels_tgt,
                    src_attention_mask=src_attention_mask, tgt_attention_mask=tgt_attention_mask, from_mean=True)

    loss = loss_rec.mean() + beta * max(loss_reg, kl_rate)

    return loss, loss_rec, loss_reg

def train_step(device, model, optimizer, inputs_src, inputs_tgt, labels_tgt, src_attention_mask, tgt_attention_mask, beta, kl_rate):
    optimizer.zero_grad()
    loss, loss_rec, loss_reg = compute_loss(device, model, inputs_src, inputs_tgt, labels_tgt, src_attention_mask,
                                            tgt_attention_mask, beta, kl_rate)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)  # max_grad_norm=1.0
    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
    optimizer.step()


def evaluate(args):
    pass

def main(args):
    now = datetime.datetime.now()

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

    fusion_type = "add_attn" if args.add_attn else "add_input"
    # logging
    experiment = f"{args.dataset}_iter{args.iterations}_as{args.adapter_size}_scalar{args.adapter_scalar}_{fusion_type}_beta{args.beta_0}" \
                 f"_reg-{args.reg_loss}_attn_mode-{args.attn_mode}_ffn_option-{args.ffn_option}_enc_layer-{args.encoder_n_layer}_" \
                 f"dec_layer-{args.decoder_n_layer}_zdim-{args.latent_size}_zrate-{args.kl_rate}_sd-{args.seed}_{now.month}.{now.day}"
    save_folder = os.path.join(args.out_dir, experiment)
    os.makedirs(os.path.join(save_folder, 'ckpt/model'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'ckpt/opt'), exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
    # importlib.reload(logging)
    logging_file = f"{args.dataset}_init-{args.adapter_init}_ada-scalar{args.adapter_scalar}_as{args.adapter_size}_" \
                   f"{fusion_type}_beta{args.beta_0}_reg-{args.reg_loss}_attn_mode-{args.attn_mode}_ffn_option-{args.ffn_option}" \
                   f"beta{args.beta_0}_enc_layer-{args.encoder_n_layer}_dec_layer-{args.decoder_n_layer}_" \
                   f"zdim-{args.latent_size}_zrate-{args.kl_rate}_sd-{args.seed}_{now.month}.{now.day}.log"
    logging = Logger(os.path.join(save_folder, logging_file))
    # logging.basicConfig(filename=os.path.join(save_folder, 'train.log'),
    #                     level=logging.INFO, format='%(asctime)s--- %(message)s', filemode='w')
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))

    logging.info('Loading models...')
    # cache_dir = os.path.join(args.out_dir, 'model_cache')
    # os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='/home/tuhq/.cache/torch/transformers')
    special_tokens_dict = {'sep_token': '<separate>', 'pad_token': '<pad>'}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Hack to allow tokenizing longer sequences.
    # tokenizer.max_len = int(1e12)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='/home/tuhq/.cache/torch/transformers')
    logging.info(f'gpt2_params:{num_params(gpt2_model)}')  # gpt2: 124439808
    logging.info(f'gpt2_transformer_params:{num_params(gpt2_model.transformer)}')

    ## GPT2 config and adapter config
    config = GPT2Config()
    ada_config = AdapterConfig(hidden_size=768,
                               adapter_size=args.adapter_size,
                               adapter_act='relu',
                               adapter_initializer_range=1e-2,
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
                               tune_enc=args.finetune_enc,
                               tune_dec=args.finetune_dec)
    assert ada_config.ffn_option in ['sequential', 'parallel_attn', 'parallel_ffn',
                                     'pfeiffer'], 'expect proper ffn_option'
    ## latent (z) size is n_embd = 768
    model_sf = SpaceFusion(args, config, AdapterConfig, tokenizer.sep_token, tokenizer.pad_token, add_input=args.add_input,
                           add_attn=args.add_attn, add_softmax=args.add_softmax, attn_proj_vary=args.attn_proj_vary,
                           learn_prior=args.learn_prior, reg_loss=args.reg_loss)
    init_para_frompretrained(model_sf.transformer, gpt2_model.transformer, share_para=True)
    init_para_frompretrained(model_sf.encoder, gpt2_model.transformer, share_para=True)

    ## freeze all prarameters expect the ones in adapters
    # model_sf = freeze_all_parameters(model_sf)
    # model_sf.transformer = unfreeze_GPT2_adapters(model_sf.transformer, Cond_GPT2Adapter)
    # model_sf.encoder = unfreeze_GPT2_adapters(model_sf.encoder, Cond_GPT2Adapter)

    if args.learn_prior:
        init_para_frompretrained(model_sf.encoder_prior, model_sf.encoder, share_para=True)
        model_sf.encoder_prior.averageSelfAttention.attention_weights = model_sf.encoder.averageSelfAttention.attention_weights
    model_sf.lm_head.weight = gpt2_model.lm_head.weight

    if model_sf.add_softmax:
        model_sf.lm_head_rep = Conv1D(*gpt2_model.lm_head.weight.size())
        # model_sf.lm_head_rep = LM_head_rep(*gpt2_model.lm_head.weight.size()[::-1])
    adavae_params = num_params(model_sf)
    logging.info(f'model_sf params: {adavae_params}')

    # fix pre-trained parameters before certain iterations
    tuning_all_after_iters = int(args.iterations / 6)
    args.warmup = args.beta_warmup = int(args.iterations / 6)
    args.cycle = int(args.iterations / 3)
    tuning_all = False
    for name, parameter in model_sf.named_parameters():
        new_pars = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj', 'attn_proj', 'Nu_fc1', 'Nu_fc2',
                    'lm_head_rep']
        if args.reg_loss == "adversarial":
            new_pars.append('discriminator')

        if not any([True if n in name else False for n in new_pars]):
            parameter.requires_grad = False
        # print((name, parameter.requires_grad))
    logging.info(f'model_sf params with gradients: {num_params(model_sf)}')

    logging.info('Setup data...')
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens)
    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedule'
    args.switch_time = 0
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0
    logging.info('Batch schedule')
    logging.info(batch_schedule)
    GDataset = DialogGenerationDataset
    prefix_path = '../data'
    train_loader = DataLoader(
        GDataset.from_file(os.path.join(prefix_path, args.dataset, "train.txt")),
        batch_size=batch_schedule[cur_b_schedule][0],
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers)
    test_loader = DataLoader(
        GDataset.from_file(os.path.join(prefix_path, args.dataset, "test.txt")),
        batch_size=batch_schedule[-1][0],
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers)
    val_loader = DataLoader(
        GDataset.from_file(os.path.join(prefix_path, args.dataset, "valid.txt")),
        batch_size=batch_schedule[-1][0],
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers)
    logging.info('Done.')

    logging.info('Wrapping models and optimizers...')
    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                  int(args.iterations * args.switch_time))
    model_sf = model_sf.to(device)
    model_sf.train()

    optimizer = AdamW(model_sf.parameters(), lr=args.lr, correct_bias=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    model_sf, optimizer = amp.initialize(model_sf, optimizer, opt_level=args.fp16_opt_level)

    ## load ckpt
    if args.load:
        logging.info('Loading model weights...')
        state = torch.load(os.path.join(save_folder, 'ckpt/model', 'model_latest.pt'))  # , map_location='cpu' model_latest.pt
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        ## load trained parameters
        if not args.save_all:
            model_dict = model_sf.state_dict()
            additional_dict = {k: v for k, v in state.items() if k in model_dict}
            model_dict.update(additional_dict)
            model_sf.load_state_dict(model_dict)
            del model_dict
        else:
            model_sf.load_state_dict(state)
            del state
        # optimizer.load_state_dict(torch.load(os.path.join(save_folder, 'ckpt/opt',
        #                                                   'optimizer_0000048.pt')))
        # gc.collect()
    logging.info('Done.')

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    logging.info('Done.')

    logging.info('Begin training iterations')
    logging.info("Begin training iterations")
    max_val_batches = 200  # max num. of val batches
    logging.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    optimizer.zero_grad()
    beta = args.beta_0
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def val_step(val_loader):
        model_sf.eval()

        n_words_bpe = 0
        n_words = 0
        n_examples = 0
        cnt_au = 0
        logp_sum = 0.0
        reg_loss_sum = 0.0

        mu_batch_list, logvar_batch_list = [], []
        neg_entropy = 0.

        logging.info("Validation loop.         Batches: %d" % len(val_loader))
        logging.info("Validation loop. max_val_batches: %d" % max_val_batches)

        with tqdm(total=min(len(val_loader), max_val_batches), desc="Evaluating Model") as pbar:
            for i, val_data_dict in enumerate(val_loader):
                with torch.no_grad():
                    val_x_ids, val_input_ids, val_attention_mask = tokenize(val_data_dict['x'], tokenizer, device, args)
                    # val_label_onehot = F.one_hot(torch.tensor(val_data_dict['y']),
                    #                          torch.tensor(ada_config.class_num)).float().to(device)

                    val_loss, val_ce_loss, val_reg_loss = model_sf(device, model_sf, val_x_ids, val_input_ids,
                                                     val_attention_mask, loss_fn, 1.0, 0.0, args.reg_loss)
                    # else:
                    #     loss, ce_loss, kl_loss = compute_loss_ae(device, model_sf, x_mask, x_tokens, y_mask, y_tokens,
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
        bsz = 5
        sents, _ = sample_sequence(model_sf, args.max_length,
                                   batch_size=bsz, top_k=100, top_p=0.95,
                                   device=device, sample=True, eos_token=endoftext)
        # Sample sentences
        logging.info("-" * 50)
        sents = sents.tolist()
        for i in range(len(sents)):
            sent = sents[i]
            sent = sent[sent.index(endoftext) + 1:]

            if endoftext in sent:
                idx = sent.index(endoftext)
                sent = sent[:idx]

            sent = tokenizer.decode(sent).strip()
            logging.info(sent)

        model_sf.train()

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

                if num_iters % args.cycle >= args.cycle - args.beta_warmup:
                    beta = min(1.0, beta + (1. - args.beta_0) / args.beta_warmup)

                if not tuning_all and num_iters >= tuning_all_after_iters:
                    decoder_unfreeze_modules = [GPT2Adapter]
                    encoder_unfreeze_modules = [GPT2Adapter]
                    if ada_config.attn_mode == "prefix":
                        decoder_unfreeze_modules.append(Prefix)
                        encoder_unfreeze_modules.append(Prefix)
                    model_sf.encoder = unfreeze_GPT2_adapters(model_sf.encoder, encoder_unfreeze_modules)
                    model_sf.transformer = unfreeze_GPT2_adapters(model_sf.transformer, decoder_unfreeze_modules)
                    if args.finetune_enc or args.finetune_dec:
                        if args.finetune_enc:
                            for _, parameter in model_sf.encoder.named_parameters():
                                parameter.requires_grad = True
                        if args.finetune_dec:
                            for _, parameter in model_sf.transformer.named_parameters():
                                parameter.requires_grad = True
                    for name, parameter in model_sf.named_parameters():
                        print((name, parameter.requires_grad))
                    adavae_params_with_gradients = num_params(model_sf)
                    logging.info(f'model_sf params with gradients:{adavae_params_with_gradients}')
                    if args.finetune_enc or args.finetune_dec:
                        logging.info(
                            'Trainable parameters %d / %d= %.4f' % (adavae_params_with_gradients, adavae_params,
                                                                    adavae_params_with_gradients / adavae_params))
                    else:
                        logging.info(
                            'Additional parameters %d / %d = %.4f' % (adavae_params_with_gradients, adavae_params,
                                                                      adavae_params_with_gradients / (
                                                                                  adavae_params - adavae_params_with_gradients)))
                    tuning_all = True

                if args.warmup != -1:
                    scheduler.step()

                loss, ce_loss, reg_loss = train_step(device, model_sf, optimizer, inputs_src, inputs_tgt,
                                                     labels_tgt, src_attention_mask, tgt_attention_mask, beta, kl_rate)

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

                if end:
                    break
                num_iters += 1
                pbar.update(1)

                if num_iters % args.cycle == 0:
                    beta = args.beta_0
                    logging.info('KL annealing restart')

                if num_iters % 500 == 0:
                    logging.info("test set")
                    val_step(test_loader)
                    logging.info("validation set")
                    val_step(val_loader)

                if (num_iters + 1) % 3000 == 0:
                    logging.info('Saving model...')
                    logging.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
                    logging.info("Saving model...")
                    logging.info('\n------------------------------------------------------')

                    if args.save_all:
                        save_orderdict = model_sf.state_dict()
                    else:
                        save_orderdict = collections.OrderedDict()
                        for name, parameter in model_sf.named_parameters():
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

    if args.save_all:
        save_orderdict = model_sf.state_dict()
    else:
        save_orderdict = collections.OrderedDict()
        for name, parameter in model_sf.named_parameters():
            if parameter.requires_grad:
                save_orderdict[name] = parameter
    torch.save(save_orderdict, os.path.join(save_folder, 'model_latest.pt'))
    logging.info('Training complete.')



if __name__=="__main__":
    pass
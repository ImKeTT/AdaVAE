#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: oracle_cls.py
@author: ImKe at 2022/2/23
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import torch.nn as nn
import torch
import datetime, os, copy, math, time, collections, argparse, nltk, json, sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# from tensorboardX import SummaryWriter
from src.logger import Logger
from src.data import ConditionalGenerationDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup


parser = argparse.ArgumentParser()

# Default parameters are set based on single GPU training
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument("--seed", type=int, default=42)

# parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--max_length', type=int, default=30)
parser.add_argument('--iterations', type=int, default=15000 * 3)
parser.add_argument('--dataset', type=str, default='yelp_polarity', choices=['yelp_polarity', 'imdb_polarity'],
                    help="Dataset to use for training")
parser.add_argument('--out_dir', type=str, default='cls_train_out')

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers')



def tokenize(texts, tokenizer, device, args):
    # tokenizer.pad_token = tokenizer.eos_token
    x_tokenized = tokenizer(texts, padding=True,
                                 truncation=True,
                            return_tensors='pt', max_length=args.max_length)
    input_ids = x_tokenized['input_ids'][:, :-1].to(device)
    attention_mask = x_tokenized['attention_mask'][:, 1:].to(device)
    x_ids = x_tokenized['input_ids'][:, 1:].contiguous().to(device)
    ## target, input tokens, mask
    return x_ids, input_ids, attention_mask

class Oracle_Classifier(nn.Module):
    def __init__(self, config, class_num, wte):
        super(Oracle_Classifier, self).__init__()
        self.class_num = class_num
        self.gpt_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.gpt_embeddings.weight.data = wte.weight.data

        self.conv1 = nn.Conv1d(config.hidden_size, config.hidden_size, 3)
        self.classifier = nn.Linear(config.hidden_size, 1 if self.class_num <= 2 else self.class_num)
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    def step(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def forward(self, sentences, cond_labels):
        ft = self.gpt_embeddings(sentences)
        ft = self.conv1(ft.transpose(1, 2))
        ft = torch.mean(ft, dim=-1)
        ft = self.classifier(ft)
        prob_cls = ft.squeeze(1)
        loss_cls = self.BCEWithLogitsLoss(prob_cls, cond_labels.float())
        pred_cls = (prob_cls >= 0).to(dtype=torch.long)
        acc_cls = (pred_cls == cond_labels).float()

        return loss_cls, acc_cls


def train(args):
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

    save_folder = os.path.join(args.out_dir, "oracle_cls")
    os.makedirs(save_folder, exist_ok=True)
    logging_file = "oracle_cls.log"
    logging = Logger(os.path.join(args.out_dir, logging_file))
    # t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)

    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))
    logging.info('Loading models...')

    config = GPT2Config()
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='/home/tuhq/.cache/torch/transformers')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='/home/tuhq/.cache/torch/transformers')
    tokenizer.pad_token = tokenizer.eos_token

    model = Oracle_Classifier(config, args.class_num, wte=gpt2_model.transformer.wte)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    model = model.to(device)
    model.train()

    logging.info('Setup data...')
    train_loader = DataLoader(
        ConditionalGenerationDataset.from_file(f"../data/{args.dataset}/train.txt"),
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers)
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
    logging.info('Done.')


    def val_step(val_loader):
        model.eval()
        val_loss_list, val_acc_list = [], []
        with tqdm(total=min(len(val_loader), max_val_batches), desc="Evaluating Model") as pbar:
            for i, val_data_dict in enumerate(val_loader):
                with torch.no_grad():
                    val_x_ids, val_input_ids, val_attention_mask = tokenize(val_data_dict['x'], tokenizer, device, args)
                    val_labels = torch.tensor(val_data_dict['y']).to(device)

                    val_loss_cls, val_acc_cls = model(val_input_ids, val_labels)
                    val_loss_list.append(val_loss_cls.item())
                    val_acc_list.append(val_acc_cls.mean().item())
        val_loss = np.mean(val_loss_list)
        val_acc = np.mean(val_acc_list)
        val_loss_std = np.std(val_loss_list)
        val_acc_std = np.std(val_acc_list)

        logging.info("val loss: %.4f + %.4f" % (val_loss, val_loss_std))
        logging.info("val acc : %.4f + %.4f" % (val_acc, val_acc_std))
        model.train()
        return val_acc


    best_acc = 0.0
    logging.info("Begin training iterations")
    max_val_batches = 200  # max num. of val batches
    logging.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    et = 0
    while num_iters < args.iterations:
    # Run epoch
        # Training
        print('Training loop. Batches:', len(train_loader))
        logging.info('\n----------------------------------------------------------------------')
        logging.info("Training loop.       Batches: %d" % len(train_loader))

        # train_iter = iter(train_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(train_iter)
        with tqdm(total=len(train_loader)) as pbar:
            for i, data_dict in enumerate(train_loader):
                x_ids, input_ids, attention_mask = tokenize(data_dict['x'], tokenizer, device, args)
                cond_labels = torch.tensor(data_dict['y']).to(device)

                loss_cls, acc_cls = model(input_ids, cond_labels)
                loss = model.step(optimizer, loss_cls)
                acc_cls = acc_cls.mean()

                # t_writer.add_scalar('loss', loss, num_iters)
                # t_writer.add_scalar('acc', acc_cls, num_iters)

                end = num_iters >= args.iterations

                if end:
                    break
                num_iters += 1
                pbar.update(1)

                if (num_iters + 1) % 2000 == 0:
                    logging.info("Test dataset")
                    _ = val_step(test_loader)
                    logging.info("Valid dataset")
                    val_acc = val_step(val_loader)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        save_orderdict = model.state_dict()
                        torch.save(save_orderdict, os.path.join(save_folder, 'oracle_cls_best.pt'))
                    else:
                        et += 1
        if et >= 5:
            logging.info("Early Stopping..")
            break

        if not end:
            e += 1
            logging.info("Training loop. The ith epoch completed: %d" % e)

    # save_orderdict = model.state_dict()
    # torch.save(save_orderdict, os.path.join(save_folder, 'oracle_cls_latest.pt'))

    logging.info("Test dataset")
    val_step(test_loader)
    logging.info("Valid dataset")
    val_step(val_loader)
    logging.info("-" * 50)
    logging.info("best acc: {:.4f}".format(best_acc))


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
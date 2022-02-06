#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: cls_ft.py
@author: ImKe at 2022/1/10
@email: tuisaac163@gmail.com
@feature: # fine-tuned transformer for classification
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.getcwd()))
import torch.nn as nn
import numpy as np
from src.logger import Logger
from src.data import ConditionalGenerationDataset
from src.utils import *
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
import torch, logging, math, time, os, argparse, re, copy
from apex import amp
from tqdm import tqdm
import random
import datetime
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup


devices = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = devices

parser = argparse.ArgumentParser()

# Default parameters are set based on single GPU training
parser.add_argument('--lr', type=float, default = 2e-5)
parser.add_argument("--seed", type=int, default=42)

# parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
parser.add_argument('--iterations', type=int, default=50000)  # wp 850001  wi 300001 ax 300001 yp 800001
parser.add_argument('--dataset', type=str, default='imdb_polarity', choices=['yelp_polarity, imdb_polariry'],
                    help="Dataset to use for training")

parser.add_argument('--class_num', type=int, default=2,
                    help="class number for controllable generation")
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size per GPU.')
parser.add_argument('--seq-lens', nargs='+', type=int, default=[50],
                    help='seq length per sample. Lists the schedule.')
parser.add_argument('--max_length', type=int, default=50,
                    help='max length of every input sentence')
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--out_dir', type=str, default='out')
parser.add_argument('--cls_dir', type=str, default='../src/out')
parser.add_argument('--load', type=str, help='path to load model from') # , default='out/test/'
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument("--latest_date", type=str, help="Latest date for model testing.", default="1.25")

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")

parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)




class AutoModelClassificationFinetuner(nn.Module):
    def __init__(self,
                 model_name: str,
                 n_classes: int,
                 max_length: int = 100,
                 lr: float = 2e-05,
                 eps: float = 1e-08,
                 device: str = 'cuda'):
        super(AutoModelClassificationFinetuner, self).__init__()
        self.max_length = max_length
        self.lr = lr
        self.eps = eps
        self.device = device

        config = AutoConfig.from_pretrained(model_name,
                                            num_labels=n_classes,
                                            output_attentions=False,
                                            output_hidden_states=False,
                                            torchscript=True,
                                            cache_dir="/data/zhangsy/_cache/torch/transformers/bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       use_fast=False,
                                                       cache_dir="/data/zhangsy/_cache/torch/transformers/bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                            config=config,
                                                                            cache_dir="/data/zhangsy/_cache/torch/transformers/bert-base-uncased")

    def tokenize(self, texts):
        x_tokenized = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=self.max_length,
                                     return_tensors='pt')
        input_ids = x_tokenized['input_ids'].to(self.device)
        attention_mask = x_tokenized['attention_mask'].to(self.device)
        return input_ids, attention_mask

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def compute(self, batch):
        x = batch['x']
        y = batch['y'].to(self.device)

        loss, logits = self(*self.tokenize(x), labels=y)
        a, y_train = torch.max(logits, dim=1)
        train_acc = accuracy_score(y.cpu(), y_train.cpu())
        return loss, logits, train_acc

    def test_step(self, batch):
        loss, logits, _ = self.compute(batch)

        y = batch['y']
        a, y_hat = torch.max(logits, dim=1)
        test_acc = accuracy_score(y.cpu(), y_hat.cpu())
        test_recall = recall_score(y.cpu(), y_hat.cpu(), average='macro')
        test_precision = precision_score(y.cpu(), y_hat.cpu(), average='macro')
        test_f1 = f1_score(y.cpu(), y_hat.cpu(), average='macro')


        return test_acc, test_recall, test_precision, test_f1, loss.cpu()


def train(args):
    now = datetime.datetime.now()
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
    if gpu:
        torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    args.experiment = f"cls_train_{args.dataset}_{now.month}.{now.day}"
    # logging
    save_folder = os.path.join(args.out_dir, args.experiment)
    os.makedirs(os.path.join(save_folder, 'ckpt/model'), exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'test'), flush_secs=5)
    # importlib.reload(logging)
    logging = Logger(os.path.join(save_folder, f'cls_train_{args.dataset}.log'))
    # logging.basicConfig(filename=os.path.join(save_folder, 'train.log'),
    #                     level=logging.INFO, format='%(asctime)s--- %(message)s', filemode='w')
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))

    print('Loading models...')
    model_name = "bert-base-uncased"

    model = AutoModelClassificationFinetuner(model_name, n_classes=args.class_num, lr=args.lr)


    print('Setup data...')
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
    print('Done.')

    ###
    val_loader = test_loader
    ###

    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=args.iterations
    )
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    num_iters=0
    e=0
    def val_step(val_loader):
        max_val_batches = 10
        model.eval()
        val_acc_list = []
        val_loss_list = []
        with tqdm(total=min(len(val_loader), max_val_batches), desc="Evaluating Model") as pbar:
            for i, val_data_dict in enumerate(val_loader):
                with torch.no_grad():
                    val_acc, val_recall, val_precision, val_f1, val_loss = model.test_step(val_data_dict)
                    val_acc_list.append(val_acc)
                    val_loss_list.append(val_loss)
                if i > max_val_batches:
                    break
                pbar.update(1)
        val_loss = np.mean(val_loss_list)
        val_acc = np.mean(val_acc_list)
        v_writer.add_scalar('val_loss', val_loss, num_iters)
        v_writer.add_scalar('val_acc', val_acc, num_iters)
        logging.info('val loss    : %.4f' % val_loss)
        logging.info('val acc     : %.4f' % val_acc)
        model.train()
    # args.iterations = len(train_loader)*5 ## 5 epochs
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
                optimizer.zero_grad()
                loss, logits, acc = model.compute(data_dict)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)  # max_grad_norm=1.0
                optimizer.step()
                scheduler.step()
                lr = scheduler.get_last_lr()[0]

                # Log to Tensorboard
                t_writer.add_scalar('loss', loss, num_iters)
                t_writer.add_scalar('acc', acc, num_iters)
                t_writer.add_scalar('lr', lr, num_iters)

                st = time.time()
                # logging.info("time %.4f" % st)
                end = num_iters >= args.iterations

                if end:
                    break
                num_iters += 1
                pbar.update(1)

                if num_iters % 3000 == 0:
                    val_step(test_loader)

                if (num_iters + 1) % 30000 == 0:
                    print('Saving model...')
                    logging.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
                    logging.info("Saving model...")
                    logging.info('\n------------------------------------------------------')

                    save_orderdict = model.state_dict()
                    torch.save(save_orderdict,
                               os.path.join(save_folder, 'ckpt/model',
                                            'model_' + '{:07d}'.format(num_iters) + '.pt'))

        if not end:
            e += 1
            logging.info("Training loop. The ith epoch completed: %d" % e)
    logging.info("Training Finished. Saving the last model weights")
    save_orderdict = model.state_dict()
    torch.save(save_orderdict,
               os.path.join(save_folder, 'ckpt/model',
                            'model_latest'.format(num_iters) + '.pt'))

def process_generated_texts(args, path):
    sentences_all = []
    for i in range(args.class_num):
        with open(os.path.join(path, f"5000-label{i}.txt"), 'r') as f:
            sentences = f.readlines()
        sentences = [f"{int(i)}\t"+ins for ins in sentences]
        sentences_all.extend(sentences)
    random.shuffle(sentences_all)
    return sentences_all

def test(args):
    now = datetime.datetime.now()
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
    if gpu:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print('Loading models...')
    model_name = "bert-base-uncased"
    ckpt = os.path.join(args.out_dir, f"cls_train_{args.dataset}_2.4/ckpt/model/model_latest.pt")
    model = AutoModelClassificationFinetuner(model_name, n_classes=args.class_num)
    model.load_state_dict(torch.load(ckpt))
    print('Done')

    def val_step(val_loader, logging):
        max_val_batches = 99999
        val_acc_list = []
        val_recall_list = []
        val_precision_list = []
        val_f1_list = []
        val_loss_list = []
        with tqdm(total=min(len(val_loader), max_val_batches), desc="Evaluating Model") as pbar:
            for i, val_data_dict in enumerate(val_loader):
                with torch.no_grad():
                    val_acc, val_recall, val_precision, val_f1, val_loss = model.test_step(val_data_dict)
                    val_acc_list.append(val_acc)
                    val_recall_list.append(val_recall)
                    val_precision_list.append(val_precision)
                    val_f1_list.append(val_f1)
                    val_loss_list.append(val_loss)
                if i > max_val_batches:
                    break
                pbar.update(1)
        val_loss = np.mean(val_loss_list)

        val_acc = np.mean(val_acc_list)
        val_acc_std = np.std(val_acc_list)
        val_recall = np.mean(val_recall_list)
        val_recall_std = np.std(val_recall_list)
        val_precision = np.mean(val_precision_list)
        val_precision_std = np.std(val_precision_list)
        val_f1 = np.mean(val_f1_list)
        val_f1_std = np.std(val_f1_list)
        logging.info('val loss      : %.4f' % val_loss)
        logging.info('val acc       : %.4f + %.4f' % (val_acc, val_acc_std))
        logging.info('val recall    : %.4f + %.4f' % (val_recall, val_recall_std))
        logging.info('val precision : %.4f + %.4f' % (val_precision, val_precision_std))
        logging.info('val f1        : %.4f + %.4f' % (val_f1, val_f1_std))

    latest_date = args.latest_date
    for experiment in os.listdir(args.cls_dir):
        if compare_date(latest_date, experiment.split("_")[-1]) and "_".join(experiment.split("_")[:2]) == args.dataset:
            # logging
            logging = Logger(os.path.join(args.cls_dir, experiment, f'cls_test.log'))
            logging.info('\n*******************************************************************************\n')
            logging.info("the configuration:")
            logging.info(str(args).replace(',', '\n'))

            eval_list = process_generated_texts(args, os.path.join(args.cls_dir, experiment, "test_texts"))
            dataloader = DataLoader(
                ConditionalGenerationDataset(eval_list),
                batch_size=args.batch_size,
                pin_memory=True,
                drop_last=True,
                num_workers=args.workers)

            model = model.to(device)
            model.eval()

            val_step(dataloader, logging)

            logging.info("Testing Finished..")

if __name__=="__main__":
    args = parser.parse_args()
    # train(args)
    test(args)


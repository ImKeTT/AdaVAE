#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: preprocess_dailydialog.py
@author: ImKe at 2022/2/9
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()

## training paramters
parser.add_argument('--max_length', type=int, default=40,
                    help='max length of every input sentence')
parser.add_argument('--window_size', type=int, default=10,
                    help="dialogue window size.")
parser.add_argument('--sep_word', type=str, default=";")
parser.add_argument('--data_dir', type=str, default='../data/dailydialog')
parser.add_argument('--out_dir', type=str, default='train_out')

def preprocess(args, datalist):
    processed = []
    for line in datalist:
        line = line.split("__eou__")[:-1] ## exclude \n
        if len(line)>1:
            line = [l.replace(";", ",") for l in line]
            line = line[:args.window_size + 1]
            context = [' '.join(text.split()[:args.max_length]) for text in line[:-1]]
            context = f' {args.sep_word} '.join(context)
            response = line[-1]
            processed.append(context + "\t" + response)

    return processed


def main(args):
    process_file = ['train', 'test', 'validation']
    for file in process_file:
        with open(os.path.join(args.data_dir, file, f"dialogues_{file}.txt"), 'r') as f:
            datalist = f.readlines()
        processed = preprocess(args, datalist)
        with open(os.path.join(args.data_dir, f"{file[:5]}.txt"), 'w') as f:
            for line in processed:
                f.write(line + "\n")
        print(f"Finish processing {file} file.")

if __name__=="__main__":
    args = parser.parse_args()
    main(args)



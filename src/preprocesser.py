#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: preprocesser.py
@author: ImKe at 2021/12/27
@email: tuisaac163@gmail.com
@feature: #preprocess yelp and imdb dataset to unified format
"""
import pandas as pd
import numpy as np
import os
import random

dataroot = "/data/tuhq/PTMs/bert_adapter/data"
def process_csv(path):
    data = f"{dataroot}/{path}"
    aa = pd.read_csv(data, header=None)
    texts = []
    for index in range(len(aa)):
        if aa[0][index]==1:
            texts.append('0\t'+aa[1][index])
        else:
            texts.append('1\t'+aa[1][index])
    with open(f"{dataroot}/{path}.txt", 'w') as f:
        for i in texts:
            f.write(i + '\n')
    print("Finished..")

def process_imdb(mode):
    data_dir = f"{dataroot}/imdb_polarity/{mode}/pos"

    files = os.listdir(data_dir)
    pos_list = []
    for file in files:
        with open(f"{data_dir}/{file}", 'r') as f:
            inst = f.readlines()
        pos_list.append('1\t' + inst[0])
    data_dir = f"{dataroot}/imdb_polarity/{mode}/neg"

    files = os.listdir(data_dir)
    neg_list = []
    for file in files:
        with open(f"{data_dir}/{file}", 'r') as f:
            inst = f.readlines()
        neg_list.append('0\t' + inst[0])
    pos_list.extend(neg_list)
    random.shuffle(pos_list)
    with open(f"{dataroot}/imdb_polarity/{mode}.txt", 'w') as f:
        for i in pos_list:
            f.write(i + '\n')
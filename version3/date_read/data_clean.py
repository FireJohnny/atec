#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: data_clean.py
@time: 2018/6/3 17:07
"""

import csv
import codecs
import jieba as jb
import random
import sys
# sys.setdefaultencoding("utf-8")
import pandas as pd
import numpy as np





def read_data(file_dir):
    jb.add_word("花呗")
    jb.add_word("借呗")
    _index = []
    text_1 = []
    text_2 = []
    label  = []
    with codecs.open(file_dir,encoding="utf-8") as f:
        file_text = f.readlines()
        for i in file_text:
            _in,t_1,t_2,_l = i.replace("\ufeff","").strip().split("\t")
            _index.append(_in)

            text_1.append(cut_word(t_1))
            text_2.append(cut_word(t_2))
            if int(_l) ==0:
                label.append([0,1]) #[0,1] : label = 0
            else :
                label.append([1,0]) #[1, 0 ] : label = 1
    max_len = max([len(t) for t in (text_1+text_2)])
    text_1 = [" ".join(t) for t in text_1 ]
    text_2 = [" ".join(t) for t in text_2 ]
    return _index,text_1,text_2,np.array(label),max_len
    pass

def read_out_file(file_dir):
    jb.add_word("花呗")
    jb.add_word("借呗")
    _index = []
    text_1 = []
    text_2 = []

    with codecs.open(file_dir,encoding="utf-8") as f:
        file_text = f.readlines()
        for i in file_text:
            _in,t_1,t_2 = i.replace("\ufeff","").strip().split("\t")
            _index.append(_in)

            text_1.append(cut_word(t_1))
            text_2.append(cut_word(t_2))

    text_1 = [" ".join(t) for t in text_1 ]
    text_2 = [" ".join(t) for t in text_2 ]
    return _index,text_1,text_2,




def cut_word(_text):
    t = list(jb.cut(_text))
    return t


def creat_batch(text1,text2,labels,batch_size = 64,random_data = True):
    data_len  = len(text1)
    num_batch_per_epoch = int((data_len-1)/batch_size)+1
    if random_data:
        shuffle_indices = np.random.permutation(np.arange(data_len))
        shuffle_text1 = np.array(text1)[shuffle_indices]
        shuffle_text2 = np.array(text2)[shuffle_indices]
        shuffle_lablels = labels[shuffle_indices]
    for batch in range(num_batch_per_epoch):
        start_index = batch*batch_size
        end_index = min((batch+1)*batch_size,data_len)
        yield shuffle_text1[start_index:end_index],shuffle_text2[start_index:end_index],shuffle_lablels[start_index:end_index]
        pass
    pass
if __name__ == '__main__':
    f_dir = "F:/game/atec/atec_nlp_sim_train.csv"
    f_dir_2 = "F:/game/atec/atec_nlp_sim_train_add.csv"
    read_data(f_dir)
    pass



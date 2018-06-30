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
    # label  = []
    labels = []
    with codecs.open(file_dir,encoding="utf-8") as f:
        file_text = f.readlines()
        for i in file_text:
            _in,t_1,t_2, _l = i.replace("\ufeff","").strip().split("\t")
            _index.append(_in)

            text_1.append(cut_word(t_1))
            text_2.append(cut_word(t_2))
            labels.append( _l)
            # if int(_l) ==0:
            #     label.append([0,1]) #[0,1] : label = 0
            # else :
            #     label.append([1,0]) #[1, 0 ] : label = 1

    max_len = max([len(t) for t in (text_1+text_2)])
    text_1 = [" ".join(t) for t in text_1 ]
    text_2 = [" ".join(t) for t in text_2 ]
    return _index,text_1,text_2,max_len,labels
    pass

def read_data_2(file_dir):
    jb.add_word("花呗")
    jb.add_word("借呗")
    _index = []
    text_1 = []
    # text_2 = []
    # label  = []
    labels = []
    with codecs.open(file_dir,encoding="utf-8") as f:
        file_text = f.readlines()
        for i in file_text:
            _in,t_1,t_2, _l = i.replace("\ufeff","").strip().split("\t")
            _index.append(_in)

            text_1.append(cut_word(t_1))
            text_1.append(cut_word(t_2))
            labels.append( _l)
            # if int(_l) ==0:
            #     label.append([0,1]) #[0,1] : label = 0
            # else :
            #     label.append([1,0]) #[1, 0 ] : label = 1

    max_len = max([len(t) for t in (text_1)])
    text_1 = [" ".join(t) for t in text_1 ]
    # text_2 = [" ".join(t) for t in text_2 ]
    return _index,text_1,max_len,labels


def balance_data(t1,t2,label_1, _use=True):
    label_1_ind = []
    label_0_ind = []
    text1 = []
    text2 = []
    # label = label.tolist()
    for i in range(len(label_1)):
        if int(label_1[i]) == 0:
            label_0_ind.append(i)
        else:
            label_1_ind.append(i)
    #label_0_ind_new = random.sample(label_0_ind,25000)
    for i in label_1_ind:
        text1.append(t1[i])
        text2.append(t2[i])
    if _use:
        label_0_ind_new = random.sample(label_0_ind,50000)
        for i in label_0_ind_new:
            text1.append(t1[i])
            text2.append(t2[i])
    else:
        label_0_ind_new = label_0_ind
        for i in label_0_ind_new:
            text1.append(t1[i])
            text2.append(t2[i])
    label1 = [[0, 1] for _ in range(len(label_1_ind)) ]
    label0 = [[1, 0] for _ in range(len(label_0_ind_new))]
    label_a = label1 + label0
    return text1,text2,label_a

def new_data_process(t1,t2,labels):
    
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



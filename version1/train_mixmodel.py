#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: train_mixmodel.py
@time: 2018/6/5 20:37
"""

import codecs
import Graph_model.mix_model as m
from date_read.data_clean import *
from tensorflow.contrib.data import Dataset
from tensorflow.contrib import learn
import tensorflow as tf
import time
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,"

def train():
    train_file = "./atec_nlp_sim_train.csv"
    train_file2 = "./atec_nlp_sim_train_add.csv"
    L2 = 0.02
    _index,t_1,t_2,labels,max_len = read_data(train_file)

    out_index,out_t1,out_t2,labels2,max_len = read_data(train_file2)
    labels = np.concatenate((labels,labels2),axis=0)
    max_len = 20
    vocab_process = learn.preprocessing.VocabularyProcessor(max_len)
    vocab_process.fit(t_1+t_2+out_t1+out_t2)
    t_1 = list(vocab_process.transform(t_1+out_t1))
    t_2 = list(vocab_process.transform(t_2+out_t2))
    # out_t1 = np.array(list(vocab_process.transform(out_t1)))
    # out_t2 = np.array(list(vocab_process.transform(out_t2)))
    global vocab
    vocab = len(vocab_process.vocabulary_)
    vocab_process.save("vocab")

    dev_sample_index = -1 * int(0.1 * float(len(labels)))

    train_1,dev_1 = t_1[:dev_sample_index],t_1[dev_sample_index:]
    train_2,dev_2 = t_2[:dev_sample_index],t_2[dev_sample_index:]
    train_labels,dev_labels = np.array(labels)[:dev_sample_index],np.array(labels)[dev_sample_index:]
    epoches = 20


    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = m.model(emb_dim=100, length=max_len, vocab_size=vocab, filter_size=2, conv_out=64, lstm_cell=32)
            model.build()
            sess.run(tf.global_variables_initializer())
            # Saver = tf.
            timenow = str(int(time.time()))
            if not os.path.exists("./log/"+timenow):
                os.mkdir("./log/"+timenow)
            if not os.path.exists("./log/"+timenow+"/train"):
                os.mkdir("./log/"+timenow+"/train")
            if not os.path.exists("./log/"+timenow+"/dev"):
                os.mkdir("./log/"+timenow+"/dev")
            trainWriter=tf.summary.FileWriter("./log/"+timenow+"/train",sess.graph)
            devWriter=tf.summary.FileWriter("./log/"+timenow+"/dev",sess.graph)
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=0)
            def train(train_1,train_2,_labels,step):

                for b_1,b_2,labels in creat_batch(train_1,train_2,_labels,batch_size=64):
                    feed_dict = {
                        model.x_1:b_1,
                        model.x_2:b_2,
                        model.labels:labels,
                        model.keep_prob:0.5
                    }

                    _,acc,summary = sess.run([model.Opt,model.acc,model.summary],feed_dict=feed_dict)
                    trainWriter.add_summary(summary,global_step=step)
                    print("step:{}, acc:{}".format(step,acc))
                    if step %100==0:
                        dev(dev_1,dev_2,dev_labels,step)
                    step+=1
                return step

            def dev(test_1,test_2,labels,step):
                feed_dict = {
                        model.x_1:test_1,
                        model.x_2:test_2,
                        model.labels:labels,
                        model.keep_prob:1.0
                }
                acc,summary = sess.run([model.acc,model.summary],feed_dict=feed_dict)
                devWriter.add_summary(summary,global_step=step)
                if not os.path.exists("./log/"+timenow+"/model"):
                    os.mkdir("./log/"+timenow+"/model")
                saver.save(sess,"./log/"+timenow+"/model/model",global_step=step)
                print("step:{}, acc:{}".format(step,acc))

            step =1
            for epoch in range(epoches):
                step = train(train_1,train_2,train_labels,step)
                print("\n\tepoch: {}".format(epoch+1))



def out(input_path,out_path):
    max_len = 20
    _index,t1,t2 = read_out_file(input_path)
    vocab_process = learn.preprocessing.VocabularyProcessor.restore("./vocab")
    # vocab_process.restore("./vocab")
    t_1 = list(vocab_process.transform(t1))
    t_2 = list(vocab_process.transform(t2))
    vocab = len(vocab_process.vocabulary_)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = m.model(emb_dim=100, length=max_len, vocab_size=vocab, filter_size=2, conv_out=64, lstm_cell=32)
            model.build()
            # sess.run(tf.global_variables_initializer())
            # ckpt = tf.train.load_checkpoint("./log/model/model-4000")#_checkpoint("./log/model/model-4000")
            saver = tf.train.Saver()
            saver.restore(sess,"./log/model/model-4000")
            def final(out_t1,out_t2, index):
                feed_dict={
                    model.x_1:out_t1,
                    model.x_2:out_t2,
                    model.keep_prob:1.0
                }
                pre = sess.run([model.pre],feed_dict=feed_dict)
                with codecs.open(out_path,mode="a+",encoding="utf-8") as out:
                    for label in enumerate(pre[0]):

                        if label==0:
                            w_str =str(index)+"\t"+str(1)+"\n"
                        else:
                            w_str = str(index)+"\t"+str(0)+"\n"
                        out.writelines(w_str)
                        index +=1
                        pass
                return index

            len_t = len(t_1)

            i = 3000
            index = 1
            for x in range(0,len_t,i):

                _t1, _t2 = t_1[x:i+x], t_2[x:i+x]
                index = final(_t1, _t2,index)


if __name__ == '__main__':
    # train()
    input_path = sys.argv[1]
    out_path = sys.argv[2]
    out(input_path,out_path)
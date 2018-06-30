#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: mix_model.py
@time: 2018/6/4 10:43
"""
import tensorflow as tf
from model.embeding import *
from model.attention import *
from model.cnn_model import *
from model.rnn_model import *
from model.fc_model import *
import numpy as np
class model():
    def __init__(self, emb_dim, length,vocab_size,filter_size,conv_out,lstm_cell ):
        self.emb_dim = emb_dim
        self.length = length
        self.vocab_size = vocab_size
        self.filter_size = filter_size
        self.conv_out = conv_out
        self.lstm_cell = lstm_cell

    def _creatplacehoder(self):
        self.x_1 = tf.placeholder(dtype=tf.int32, shape=[None, self.length])
        self.x_2 = tf.placeholder(dtype=tf.int32, shape=[None, self.length])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None,2])
        self.keep_prob = tf.placeholder(dtype=tf.float32)

    def creat_model(self):
        self.embed = embed(vocab_size=self.vocab_size,embed_size=self.emb_dim)
        self.cnn_x1 = cnn_model(name="cnn_x1",out_size=self.conv_out) #input tensor: [batch, length, embed_size] kernel_size[fiter_size,len_size]
        self.cnn_x2 = cnn_model(name="cnn_x2",out_size=self.conv_out)
        self.rnn_x1 = rnn_model(name="rnn_x1", cell_size=self.lstm_cell) #input tensor: [batch, length, embed_size]
        self.rnn_x2 = rnn_model(name="rnn_x2", cell_size=self.lstm_cell)
        self.birnn_x1 = birnn_model(name="birnn_x1")
        self.birnn_x2 = birnn_model(name="birnn_x2")
        self.Attention_1 = Attention_1(size=32)  #input tensor: [batch, length-h,embed_size] ,self attention
        self.Attention_2 = Attention_1(name = "Attention_2", size=32)
        self.att_mat = Attention_2(name = "att_mat")  #input tensor1: [batch, x_1] input tensor2: [batch, x_2], outside attention before conv
        self.Att_mat_2 = Attention_2(name="att_mat_2") #outside attention before full connect
        self.Att_mat_3 = Attention_2(name = "att_mat_3")
        self.fc_layer = FC_model(name= "fc_layer")

    def pad_(self,x):
        return tf.pad(x, np.array([[0, 0], [0, 0], [self.filter_size - 1, self.filter_size - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

    def CNN_layer(self,variable_scope, x1, x2):
        with tf.variable_scope(variable_scope) as scope:
            with tf.name_scope("cnn_att_mat"):
                aW = tf.get_variable(name = "aW", shape=(self.length, self.emb_dim))
                att_mat = self.att_mat(x1, x2)

                x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl",att_mat,aW)),-1)
                x2_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl",tf.transpose(att_mat,[0,2,1]),aW)),-1)

                x1 = tf.concat([x1, x1_a], axis = 3)
                x2 = tf.concat([x2, x2_a], axis = 3)
            x1 = self.pad_(x1)
            L_conv = self.cnn_x1(x1, kernel_size=(self.emb_dim, self.filter_size),reuse = False)
            R_conv = self.cnn_x1(self.pad_(x2), kernel_size=(self.emb_dim, self.filter_size),reuse = True)

            L_conv = tf.transpose(L_conv,[0,3,2,1])
            R_conv = tf.transpose(R_conv,[0,3,2,1])

            # L_attention, R_attention = None, None

            #ABCNN3
            att_mat = self.Att_mat_2(L_conv,R_conv)
            L_attention, R_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)
            #pool
            L_wp = w_pool(variable_scope="left", x=L_conv, attention=L_attention, model_type="ABCNN", length=self.length, filer_size=self.filter_size)
            L_ap = all_pool("left",L_conv,length=self.length,filter_size=self.filter_size,embed_szie=self.emb_dim,conv_out=64)
            R_wp = w_pool(variable_scope="right", x=R_conv, attention=R_attention, model_type="ABCNN", length=self.length, filer_size=self.filter_size)
            R_ap = all_pool("left",R_conv,length=self.length,filter_size=self.filter_size,embed_szie=self.emb_dim,conv_out=64)

            return L_wp,L_ap,R_wp,R_ap

    def squeeze_data(self,x):

        _input = tf.reshape(x,[-1, self.length,self.conv_out])
        input_x = [tf.squeeze(input, [1]) for input in tf.split(_input, self.length, 1)]
        return input_x

    def LSTM_layer(self,variables_scope,x1,x2):
        with tf.variable_scope(variables_scope) as scope:
            #reconstruct squeeze data
            x1 = self.squeeze_data(x1)
            x2 = self.squeeze_data(x2)

            #input data to birnn
            L_rnn = self.birnn_x1(x1)
            R_rnn = self.birnn_x1(x2,reuse =True)
            _length = len(L_rnn)
            L_rnn = tf.transpose(L_rnn, [1, 0, 2])
            R_rnn = tf.transpose(R_rnn, [1, 0, 2])
            expend_L_rnn = tf.expand_dims(L_rnn, -1)
            trans_L_rnn = tf.transpose(expend_L_rnn,[0,2,1,3])

            expend_R_rnn = tf.expand_dims(R_rnn, -1)
            trans_R_rnn = tf.transpose(expend_R_rnn,[0,2,1,3])

            #attention matrix
            with tf.name_scope("rnn_att_mat"):
                aW = tf.get_variable(name = "aW", shape=(_length, self.lstm_cell*2))
                att_mat = self.att_mat(trans_L_rnn, trans_R_rnn)
                x1_a = tf.transpose(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)),[0,2,1])
                x2_a = tf.transpose(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.transpose(att_mat, [0,2,1]),aW)),[0,2,1])
            # attention layer
            L_att = self.Attention_1(L_rnn,)
            R_att = self.Attention_1(R_rnn,reuse = True)
            with tf.variable_scope("att2att"):
                L_att_2 = self.Attention_2(x1_a)
                R_att_2 = self.Attention_2(x2_a,reuse = True)


            return L_att, R_att, L_att_2, R_att_2
        pass

    def build(self):
        self._creatplacehoder()
        self.creat_model()
        with tf.name_scope("embed"):
            input_x1 = self.embed(self.x_1)
            input_x2 = self.embed(self.x_2, reuse=True)

        with tf.name_scope("input_pool"):
            L_x1_0 = all_pool("input_x1", input_x1, self.length, self.filter_size, self.emb_dim, conv_out=64)
            R_x2_0 = all_pool("input_x2", input_x2, self.length, self.filter_size, self.emb_dim, conv_out=64)

        with tf.name_scope("CNN_layer"):
            input_x1 = tf.nn.dropout(input_x1,self.keep_prob)
            input_x2 = tf.nn.dropout(input_x2,self.keep_prob)
            L_x1_wp, L_x1_1, R_x2_wp, R_x2_1 = self.CNN_layer("cnn_layer", input_x1, input_x2)

            sims = [cos_sim(L_x1_0, R_x2_0), cos_sim(L_x1_1, R_x2_1)]

        with tf.name_scope("LSTM_layer"):
            L_x1_wp = tf.nn.dropout(L_x1_wp,self.keep_prob)
            R_x2_wp = tf.nn.dropout(R_x2_wp,self.keep_prob)
            x1 = tf.transpose(L_x1_wp, [0, 2, 1, 3])
            x2 = tf.transpose(R_x2_wp, [0, 2, 1, 3])
            L_att, R_att, L_att_2, R_att_2 =  self.LSTM_layer(variables_scope="LSTM_layer",x1=x1,x2=x2)
            sims.append(cos_sim(L_att, R_att))
            sims.append(cos_sim(L_att_2, R_att_2))
        x = tf.stack(sims, axis=1)
        x = tf.concat([x, L_att, R_att], axis=1)
        with tf.name_scope("FC_layer"):
            x = tf.nn.dropout(x,self.keep_prob)
            self.logits = self.fc_layer(x)

        self.optimzer()
        self.f1score()
        self.predict()
        self.merge()

        pass
    def loss(self):
        return tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.labels, logits=self.logits))
        pass
    def optimzer(self):
        self.losses = self.loss()
        self.Opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=self.losses)
        pass
    def predict(self):
        self.pre =  tf.argmax(self.logits,1)


    def merge(self):
        tf.summary.scalar(name = "loss", tensor=self.losses)
        # tf.summary.scalar(name = "f1score", tensor=self.)
        self.summary = tf.summary.merge_all()



    # def train(self,epoch = 30):
    #     init = tf.global_variables_initializer()
    #     with tf.Session as sess:
    #         sess.run(init)
    #         # for i in range(epoch):
    #
    #
    #
    #
    #     pass


if __name__ == '__main__':
    test=model(100,10,10000,4,64,256)
    test._creatplacehoder()
    test.build()
    pass



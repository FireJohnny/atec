#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: utils.py
@time: 2018/6/4 9:35
"""
import tensorflow as tf


def cos_sim(v1,v2):

    norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
    dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

    return dot_products / (norm1 * norm2)


def w_pool(variable_scope, x, attention,model_type,length,filer_size):
    # x: [batch, di, s+w-1, 1]
    # attention: [batch, s+w-1]
    with tf.variable_scope(variable_scope + "-w_pool"):
        if model_type == "ABCNN":
            pools = []
                    # [batch, s+w-1] => [batch, 1, s+w-1, 1]
            attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

            for i in range(length):
                        # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                pools.append(tf.reduce_sum(x[:, :, i:i + filer_size, :] * attention[:, :, i:i + filer_size, :],
                                                   axis=2,
                                                   keepdims=True))

                # [batch, di, s, 1]
                w_ap = tf.concat(pools, axis=2, name="w_ap")
        else:
            w_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, filer_size),
                        strides=1,
                        padding="VALID",
                        name="w_ap"
                    )
                    # [batch, di, s, 1]

        return w_ap

def all_pool(variable_scope, x,length,filter_size,embed_szie,conv_out):
    with tf.variable_scope(variable_scope + "-all_pool"):
        if variable_scope.startswith("input"):
            pool_width = length
            d = embed_szie
        else:
            pool_width = length + filter_size - 1
            d = conv_out

        all_ap = tf.layers.average_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=(1, pool_width),
                    strides=1,
                    padding="VALID",
                    name="all_ap"
                )
            # [batch, di, 1, 1]

            # [batch, di]
        all_ap_reshaped = tf.reshape(all_ap, [-1, d])
            #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

        return all_ap_reshaped

if __name__ == '__main__':
    pass



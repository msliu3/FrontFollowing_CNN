#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test.py    
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version
------------      -------    --------
2020/2/12 20:26   msliu      1.0      

@Description
------------
None
"""

# import lib
import tensorflow as tf
import numpy as np

# flat_size = 32 * 24 + 32
# feed_back = tf.get_variable('feed_back', shape=[1, 32], dtype=tf.float32)
# fc1_weights = tf.get_variable('fc7_weights', shape=[flat_size, 512], dtype=tf.float32)
# fc1_biases = tf.get_variable('f7_biases', shape=[512], dtype=tf.float32)
# input_data = tf.placeholder(tf.float32, shape=[1, 24, 32, 1], name="input")
# net = input_data
# net = tf.reshape(net, shape=[-1, flat_size-32])
# print(net.shape)
# net = tf.concat([net, feed_back], axis=1)
# print(net.shape)
# net = tf.add(tf.matmul(net, fc1_weights), fc1_biases)
c = []
a = np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]]]).reshape([2, 4, 2])
b = np.array([[[2, 4, 6, 7], [5, 6, 7, 8]], [[2, 4, 6, 7], [5, 6, 7, 8]]]).reshape([2, 4, 2])
c.append(a)
c.append(b)
print(np.array(c).reshape([2, 2, 4, 2]))
print("------------------------------------")
print(a)

print(a.shape)
# with tf.Session() as sess:
#     sess.run()

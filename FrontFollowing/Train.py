#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Train.py    
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version
------------      -------    --------
2020/2/10 18:21   msliu      1.0      

@Description
------------
None
"""

import tensorflow as tf

import numpy as np
import os
import re

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
data_path_0 = father_path + os.path.sep + "resource" + os.path.sep + "stop_move" + os.path.sep + "data_combined_00.txt"
data_path_1 = father_path + os.path.sep + "resource" + os.path.sep + "stop_move" + os.path.sep + "ir_data_1_straight.txt"
data_path_2 = father_path + os.path.sep + "resource" + os.path.sep + "stop_move" + os.path.sep + "ir_data_2_right.txt"
print(data_path_0)
print(data_path_1)
print(data_path_2)
num_classes = 2


def read_data(path, label, class_num):
    with open(path) as f:
        line = f.readlines()
        line_data = list(map(to_float, line))
        # print(len(line_data))
        # labels = np.array(np.zeros((len(line_data), class_num)))
        # one = np.array(np.ones((len(line_data), 1)))
        # labels[:, label] = 1
        labels = np.array(np.ones((len(line_data)))) * label
        tensor = np.array(line_data).reshape([len(line_data), 24, 32, 1])
        return tensor, labels


def to_float(item_str):
    temp = list(map(float, item_str.split(",")))
    temp.pop(0)
    return np.array(temp).reshape((24, 32))


input_data = tf.placeholder(tf.float32, shape=[None, 24, 32, 1], name="input")
labels = tf.placeholder(tf.int32, shape=[None], name="labels")

height, width, num_channels = 24, 32, 1

conv1_weights = tf.get_variable('conv1_weights', shape=[3, 3, num_channels, 32], dtype=tf.float32)
conv1_biases = tf.get_variable('conv1_biases', shape=[32], dtype=tf.float32)
conv2_weights = tf.get_variable('conv2_weights', shape=[3, 3, 32, 32], dtype=tf.float32)
conv2_biases = tf.get_variable('conv2_biases', shape=[32], dtype=tf.float32)
conv3_weights = tf.get_variable('conv3_weights', shape=[3, 3, 32, 64], dtype=tf.float32)
conv3_biases = tf.get_variable('conv3_biases', shape=[64], dtype=tf.float32)
conv4_weights = tf.get_variable('conv4_weights', shape=[3, 3, 64, 64], dtype=tf.float32)
conv4_biases = tf.get_variable('conv4_biases', shape=[64], dtype=tf.float32)
conv5_weights = tf.get_variable('conv5_weights', shape=[3, 3, 64, 128], dtype=tf.float32)
conv5_biases = tf.get_variable('conv5_biases', shape=[128], dtype=tf.float32)
conv6_weights = tf.get_variable('conv6_weights', shape=[3, 3, 128, 128], dtype=tf.float32)
conv6_biases = tf.get_variable('conv6_biases', shape=[128], dtype=tf.float32)

flat_height = height // 2
flat_width = width // 2
flat_size = flat_height * flat_width * 128

fc7_weights = tf.get_variable('fc7_weights', shape=[flat_size, 512], dtype=tf.float32)
fc7_biases = tf.get_variable('f7_biases', shape=[512], dtype=tf.float32)
fc8_weights = tf.get_variable('fc8_weights', shape=[512, 512], dtype=tf.float32)
fc8_biases = tf.get_variable('f8_biases', shape=[512], dtype=tf.float32)
fc9_weights = tf.get_variable('fc9_weights', shape=[512, num_classes], dtype=tf.float32)
fc9_biases = tf.get_variable('f9_biases', shape=[num_classes], dtype=tf.float32)

net = input_data
net = tf.nn.conv2d(net, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))

net = tf.nn.conv2d(net, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
net = tf.nn.relu(tf.nn.bias_add(net, conv2_biases))

net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

net = tf.nn.conv2d(net, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
net = tf.nn.relu(tf.nn.bias_add(net, conv3_biases))

net = tf.nn.conv2d(net, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
net = tf.nn.relu(tf.nn.bias_add(net, conv4_biases))

# net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

net = tf.nn.conv2d(net, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
net = tf.nn.relu(tf.nn.bias_add(net, conv5_biases))

net = tf.nn.conv2d(net, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
net = tf.nn.relu(tf.nn.bias_add(net, conv6_biases))

net = tf.reshape(net, shape=[-1, flat_size])
net = tf.nn.relu(tf.add(tf.matmul(net, fc7_weights), fc7_biases))
net = tf.nn.relu(tf.add(tf.matmul(net, fc8_weights), fc8_biases))
net = tf.add(tf.matmul(net, fc9_weights), fc9_biases)
result = net
logits = tf.nn.softmax(net)
classes = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)

global_step = tf.Variable(0, trainable=False)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels))
learning_rate = tf.train.exponential_decay(0.1, global_step, 150, 0.9)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = optimizer.minimize(loss, global_step)
acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ir_data_0, label_0 = read_data(data_path_0, 0, 2)
    ir_data_1, label_1 = read_data(data_path_1, 1, 2)
    test, test_label = read_data(data_path_2, 1, 2)
    train = np.r_[ir_data_0, ir_data_1]
    label = np.r_[label_0, label_1]
    print(label_0)
    print(label_1)
    # print(test_label)
    # print(train)

    train_dict = {input_data: train, labels: label}
    result = sess.run(result, feed_dict=train_dict)
    logits = sess.run(logits, feed_dict=train_dict)
    classes = sess.run(classes, feed_dict=train_dict)
    loss = sess.run(loss, feed_dict=train_dict)
    acc_ = sess.run(acc, feed_dict=train_dict)
    print("before", acc_)
    while acc_ < 0.9:
        sess.run(train_step, feed_dict=train_dict)
        acc_ = sess.run(acc, feed_dict=train_dict)
        print(acc_)
    print("after", acc_)
    test_dict = {input_data: test, labels: test_label}
    acc_ = sess.run(acc, feed_dict=test_dict)
    print("test:", acc_)

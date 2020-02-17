#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Train_2.py    
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version
------------      -------    --------
2020/2/11 19:48   msliu      1.0      

@Description
------------
None
"""

# import lib
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
import re
from sklearn.cluster import KMeans

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
data_path_0 = father_path + os.path.sep + "resource" + os.path.sep + "stop_move" + os.path.sep + "data_combined_00.txt"
data_path_1 = father_path + os.path.sep + "resource" + os.path.sep + "stop_move" + os.path.sep + "ir_data_1_straight.txt"
data_path_2 = father_path + os.path.sep + "resource" + os.path.sep + "stop_move" + os.path.sep + "ir_data_2_right.txt"
print(data_path_0)
print(data_path_1)
print(data_path_2)
num_classes = 2

def show_temperature(temperature):
    print()
    for i in range(24):
        t = temperature[i]
        for j in range(32):
            print("%.2f" % t[j], end=" ")
        print()
    # print(temperature[0])
    pass

def k_means_detect(temperature):
    """
    这个函数的目的是，使用聚类算法（k-means）去判断原始的温度矩阵中，哪些点属于人的身体，那些点属于环境
    已有的知识分类：“身体”点温度 > 环境点温度

    btw，注意这里传入的参数是list，并不是转成图片数据后的结果，如果想要打印结果需要转成np.reshape
    :param temperature:温度的list
    :return: 返回list ， list（reslut）， falg（身体的结果0 or 1）
    """
    temp_np = np.array(temperature).reshape((len(temperature), 1))
    # print(temp_np.shape)
    result = KMeans(n_clusters=2).fit_predict(temp_np)

    # 以图片方式print result
    # result1 = np.array(result).reshape(24, 32)
    # show_temperature(result1)

    # 脚的平均温度值应该比空白地方高
    temp0 = 0.0
    num0 = 0
    temp1 = 0.0
    num1 = 0
    env_max0 = 0.0
    env_max1 = 0.0
    for i in range(len(temperature)):
        if result[i] == 0:
            if env_max0 < temperature[i]:
                env_max0 = temperature[i]
            temp0 += temperature[i]
            num0 += 1
        else:
            if env_max1 < temperature[i]:
                env_max1 = temperature[i]
            temp1 += temperature[i]
            num1 += 1
    temp0 = temp0 / num0
    temp1 = temp1 / num1

    if temp0 > temp1:
        return result, 0, env_max1
    else:
        return result, 1, env_max0


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
        print(tensor)
        return tensor, labels


def to_float(item_str):
    temp = list(map(float, item_str.split(",")))
    temp.pop(0)
    max_temp = max(temp)
    min_temp = min(temp)
    for i in range(len(temp)):
        temp[i] = (temp[i] - min_temp) / (max_temp - min_temp)
    np_data = np.array(temp).reshape((24, 32))
    # result, flag, env_max0 = k_means_detect(temp)
    # show_temperature(np_data)
    return np_data


input_data = tf.placeholder(tf.float32, shape=[None, 24, 32, 1], name="input")
labels = tf.placeholder(tf.int32, shape=[None], name="labels")

height, width, num_channels = 24, 32, 1

# conv1_weights = tf.get_variable('conv1_weights', shape=[3, 3, num_channels, 32], dtype=tf.float32)
# conv1_biases = tf.get_variable('conv1_biases', shape=[32], dtype=tf.float32)
# conv2_weights = tf.get_variable('conv2_weights', shape=[3, 3, 32, 64], dtype=tf.float32)
# conv2_biases = tf.get_variable('conv2_biases', shape=[64], dtype=tf.float32)

flat_height = height
flat_width = width
flat_size = flat_height * flat_width

fc1_weights = tf.get_variable('fc7_weights', shape=[flat_size, 512], dtype=tf.float32)
fc1_biases = tf.get_variable('f7_biases', shape=[512], dtype=tf.float32)
fc2_weights = tf.get_variable('fc8_weights', shape=[512, 512], dtype=tf.float32)
fc2_biases = tf.get_variable('f8_biases', shape=[512], dtype=tf.float32)
fc3_weights = tf.get_variable('fc9_weights', shape=[512, num_classes], dtype=tf.float32)
fc3_biases = tf.get_variable('f9_biases', shape=[num_classes], dtype=tf.float32)

net = input_data
# net = tf.nn.conv2d(net,conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
# net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))
#
# net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# net = tf.nn.conv2d(net, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
# net = tf.nn.relu(tf.nn.bias_add(net, conv2_biases))

# net = tf.nn.conv2d(net, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
# net = tf.nn.relu(tf.nn.bias_add(net, conv3_biases))
#
# net = tf.nn.conv2d(net, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
# net = tf.nn.relu(tf.nn.bias_add(net, conv4_biases))
#
# net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# net = tf.nn.conv2d(net, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
# net = tf.nn.relu(tf.nn.bias_add(net, conv5_biases))

# net = tf.nn.conv2d(net, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
# net = tf.nn.relu(tf.nn.bias_add(net, conv6_biases))

net = tf.reshape(net, shape=[-1, flat_size])
net = tf.nn.relu(tf.add(tf.matmul(net, fc1_weights), fc1_biases))
net = tf.nn.relu(tf.add(tf.matmul(net, fc2_weights), fc2_biases))
net = tf.add(tf.matmul(net, fc3_weights), fc3_biases)
result = net
logits = tf.nn.softmax(net)
classes = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)

global_step = tf.Variable(0, trainable=False)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels))
learning_rate = tf.train.exponential_decay(0.001, global_step, 150, 0.9)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = optimizer.minimize(loss, global_step)
acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ir_data_0, label_0 = read_data(data_path_0, 0, 2)
    ir_data_1, label_1 = read_data(data_path_1, 1, 2)
    test, test_label = read_data(data_path_2, 1, 2)
    print("shape:", ir_data_0.shape)
    train = np.r_[ir_data_0, ir_data_1]
    label = np.r_[label_0, label_1]
    print(label_0)
    print(label_1)
    # print(test_label)
    # print(train)
    print("shape:",train.shape)
    train_dict = {input_data: train, labels: label}
    test_dict = {input_data: test, labels: test_label}
    # result = sess.run(result, feed_dict=train_dict)
    # logits = sess.run(logits, feed_dict=train_dict)
    # classes = sess.run(classes, feed_dict=train_dict)
    loss_ = sess.run(loss, feed_dict=train_dict)
    acc_ = sess.run(acc, feed_dict=train_dict)
    print("before", acc_, "loss: ", loss_)
    train_l, train_a = [], []
    test_l, test_a = [], []
    for i in range(1):
        sess.run(train_step, feed_dict=train_dict)
        loss_, acc_ = sess.run([loss, acc], feed_dict=train_dict)
        if loss_ < 1:
            train_l.append(loss_)
            train_a.append(acc_)
            print("train:", loss_, acc_)
            loss_, acc_ = sess.run([loss, acc], feed_dict=test_dict)
            test_l.append(loss_)
            test_a.append(acc_)
            print("test:", loss_, acc_)
    plt.figure(1)
    plt.plot(test_l,ls="-.")
    plt.plot(train_l,ls="-.")
    plt.figure(2)
    plt.plot(test_a)
    plt.plot(train_a)
    plt.show()

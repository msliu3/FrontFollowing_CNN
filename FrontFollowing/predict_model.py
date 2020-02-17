#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   predict_model.py    
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version
------------      -------    --------
2020/2/17 17:58   msliu      1.0      

@Description
------------
None
"""

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
import Draw_Func.plot_position_label as draw
import FrontFollowing.Train_3 as train3

num_classes = 4
sample_size = 10
strides = 1

input_data = tf.placeholder(tf.float32, shape=[None, 24 * 32 + 4, sample_size], name="input")

labels = tf.placeholder(tf.int32, shape=[None], name="labels")

height, width, num_channels = 24, 32, 1

flat_height = height
flat_width = width
leg_size = 4
flat_size = (flat_height * flat_width + 4) * sample_size

fc1_weights = tf.get_variable('fc1_weights', shape=[flat_size, 512], dtype=tf.float32)
fc1_biases = tf.get_variable('f1_biases', shape=[512], dtype=tf.float32)
fc2_weights = tf.get_variable('fc2_weights', shape=[512, 512], dtype=tf.float32)
fc2_biases = tf.get_variable('f2_biases', shape=[512], dtype=tf.float32)
fc3_weights = tf.get_variable('fc3_weights', shape=[512, num_classes], dtype=tf.float32)
fc3_biases = tf.get_variable('f3_biases', shape=[num_classes], dtype=tf.float32)

variables_dict = {'fc1_weights': fc1_weights,
                  'f1_biases': fc1_biases,
                  'fc2_weights': fc2_weights,
                  'f2_biases': fc2_biases,
                  'fc3_weights': fc3_weights,
                  'f3_biases': fc3_biases}

net = input_data
net = tf.reshape(net, shape=[-1, flat_size])
net = tf.nn.relu(tf.add(tf.matmul(net, fc1_weights), fc1_biases))
net = tf.nn.relu(tf.add(tf.matmul(net, fc2_weights), fc2_biases))
net = tf.add(tf.matmul(net, fc3_weights), fc3_biases)
result = net

logits = tf.nn.softmax(net)
classes = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)

global_step = tf.Variable(0, trainable=False)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels))
learning_rate = tf.train.exponential_decay(0.005, global_step, 150, 0.9)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = optimizer.minimize(loss, global_step)
acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), dtype=tf.float32))

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
dir_path = father_path + os.path.sep + "resource" + os.path.sep + "stop_move" + os.path.sep
data_path_0 = dir_path + "data_combined_3.txt"
print("--------------------------------------------------------------")

np_data = train3.read_data(data_path_0)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"../resource/model_save/0.9965694682675815.ckpt")
    test_ir, test_label, test_leg = train3.extract_training_data(np_data)
    test_ir_list, test_label_list = train3.normalization_build_sample_leg_ir(test_ir, test_leg, test_label,
                                                                      sample_num=sample_size,
                                                                      strides=strides)
    predict_list = []
    sum_acc = 0
    for j in range(len(test_ir_list)):
        test_dict = {input_data: np.array(test_ir_list[j]).reshape([1, 24 * 32 + 4, sample_size]),
                     labels: np.array(test_label_list[j]).reshape([1])}
        loss_, acc_ = sess.run([loss, acc], feed_dict=test_dict)
        classes_ = sess.run(classes, feed_dict=test_dict)
        predict_list.append(classes_)
        sum_acc += acc_
    avg_acc = sum_acc / len(test_ir_list)
    print(avg_acc)
    draw.draw_label_predict(predict_list, np_data, strides=strides, sample=sample_size, fig_num=sum_acc)
    draw.draw_label_predict(test_label_list, np_data, strides=strides, sample=sample_size, fig_num=100)
    plt.show()

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Train_3.py    
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version
------------      -------    --------
2020/2/12 17:55   msliu      1.0      

@Description
------------
None
"""
import random

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
import Draw_Func.plot_position_label as draw

num_classes = 4
sample_size = 10
strides = 1


def extract_data(item_str):
    one_line = list(map(float, item_str.split("\t")))
    return np.array(one_line).reshape(1, -1)


def ir_normalization(ir_list):
    """
    这个函数接收一条ir的list，共32X24个数据
    :param ir_list: The list of ir data, 32x24
    :return:
    """
    max_temp = max(ir_list)
    min_temp = min(ir_list)
    for i in range(len(ir_list)):
        ir_list[i] = (ir_list[i] - min_temp) / (max_temp - min_temp)
    return ir_list


def read_data(path):
    """
    这个函数用来读取，合并后的数据。
    :param path:
    :return:

    """
    with open(path) as file:
        lines = file.readlines()
        line_list = list(map(extract_data, lines))
        return np.array(line_list).reshape(len(line_list), -1)


def extract_training_data(np_data):
    """
    输入是numpy data
    按列提取数据

    :param np_data:
    :return:
    """
    ir_numpy = np_data[:, 12:]
    label = np_data[:, 0]
    legs_position = np_data[:, 7: 11]
    # print(label.shape, ir.shape)
    return ir_numpy, label, legs_position


def normalization_build_sample(ir_np, label, sample_num, strides=2):
    # normalization
    np_size_32 = np.ones(32 * 24).reshape([1, 32 * 24])
    max_colum = ir_np.max(axis=1).reshape([ir_np.shape[0], 1])
    min_colum = ir_np.min(axis=1).reshape([ir_np.shape[0], 1])
    min_np = min_colum.dot(np_size_32)
    max_minus_min = max_colum - min_colum
    normal_ir = (ir_np - min_np) / max_minus_min
    # normal_ir shape[list_num,32*24]

    # sample [list_num - (sample_num - stride)]/stride
    list_num = (normal_ir.shape[0] - (sample_num - strides)) // strides
    ir_list, label_list = [], []
    label = list(label)
    for i in range(list_num):
        ir_list.append(normal_ir[i * strides:i * strides + 4].reshape([24, 32, sample_num]))
        label_list.append(label[i * strides + 3])
    # print("label:", len(label_list))
    return ir_list, label_list


def normalization_build_sample_leg_ir(ir_np, legs, label, sample_num, strides=2):
    """
    这个包含了 IR image 和 legs position两种训练数据

    normalization:
        之前是用单张图片的最大最小值做标准化

    :param ir_np:
    :param legs:
    :param label:
    :param sample_num:
    :param strides:
    :return:
    """
    # IR image normalization
    np_size_32 = np.ones(32 * 24).reshape([1, 32 * 24])
    max_colum = ir_np.max(axis=1).reshape([ir_np.shape[0], 1])
    max_value = max_colum.max()
    min_colum = ir_np.min(axis=1).reshape([ir_np.shape[0], 1])
    min_value = min_colum.min()
    min_np = np_size_32 * min_value

    max_minus_min = max_value - min_value
    normal_ir = (ir_np - min_np) / max_minus_min  # normal_ir shape[list_num,32*24]

    # legs_position normalization
    max_leg = legs.max(axis=0)
    min_leg = legs.min(axis=0)
    max_minus_min = max_leg - min_leg
    normal_leg = (legs - min_leg) / max_minus_min

    # combine ir image and lidar data
    normal = np.c_[normal_ir, normal_leg]

    # sample [list_num - (sample_num - stride)]/stride
    list_num = (normal.shape[0] - (sample_num - strides)) // strides
    train_list, label_list = [], []
    label = list(label)
    for i in range(list_num):
        train_list.append(normal[i * strides:i * strides + sample_num].reshape([24 * 32 + 4, sample_num]))
        label_list.append(label[i * strides + sample_num - 1])
    # print("label:", len(label_list))
    return train_list, label_list


def build_dict_batch(ir_list, label_list, batch_num=10, sample_num=4):
    batch_list_ir, batch_list_label = [], []
    for i in range(batch_num):
        num = random.randint(0, len(ir_list) - 1)
        batch_list_ir.append(ir_list[num])
        batch_list_label.append(label_list[num])
    dict_ir = np.array(batch_list_ir).reshape([batch_num, 24 * 32 + 4, sample_num])
    dict_label = np.array(batch_list_label).reshape(batch_num)
    return dict_ir, dict_label


def test_model(session, test_data, max_acc, baseline, save_model=False, saver=None,
               model_path="../resource/model_save/", input_data=None, labels=None):
    test_ir, test_label, test_leg = extract_training_data(test_data)
    test_ir_list, test_label_list = normalization_build_sample_leg_ir(test_ir, test_leg, test_label,
                                                                      sample_num=sample_size,
                                                                      strides=strides)
    predict_list = []
    sum_acc = 0
    for j in range(len(test_ir_list)):
        test_dict = {input_data: np.array(test_ir_list[j]).reshape([1, 24 * 32 + 4, sample_size]),
                     labels: np.array(test_label_list[j]).reshape([1])}
        loss_, acc_ = session.run([loss, acc], feed_dict=test_dict)
        classes_ = session.run(classes, feed_dict=test_dict)
        predict_list.append(classes_)
        sum_acc += acc_
    avg_acc = sum_acc / len(test_ir_list)
    if max_acc < avg_acc:
        max_acc = avg_acc
        # draw.draw_label(predict, np_data3, strides=1, sample=sample_size, fig_num=sum_acc)
        # draw.draw_label(test_label_list, np_data3, strides=1, sample=sample_size, fig_num=100)
        # plt.show()
    if avg_acc > baseline:
        draw.draw_label_predict(predict_list, test_data, strides=strides, sample=sample_size, fig_num=sum_acc)
        draw.draw_label_predict(test_label_list, test_data, strides=strides, sample=sample_size, fig_num=100)
        plt.show()
        if save_model:
            saver.save(session, model_path + str(max_acc) + ".ckpt")
    return avg_acc, max_acc


if __name__ == '__main__':

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
    data_path_0 = dir_path + "data_combined_0.txt"
    data_path_1 = dir_path + "data_combined_1.txt"
    data_path_2 = dir_path + "data_combined_2.txt"
    data_path_3 = dir_path + "data_combined_3.txt"
    print(data_path_0)
    print(data_path_1)
    print(data_path_2)

    np_data0 = read_data(data_path_0)
    np_data1 = read_data(data_path_1)
    np_data2 = read_data(data_path_2)
    np_data3 = read_data(data_path_3)

    np_data_train = np.r_[np_data0, np_data1]
    np_data_train = np.r_[np_data_train, np_data2]

    ir, l, legs = extract_training_data(np_data_train)
    ir_list, label_list = normalization_build_sample_leg_ir(ir, legs, l, sample_num=sample_size, strides=1)

    # draw.draw_label(label_list, np_data3, strides=2, sample=4, fig_num=111)
    # draw.draw_label(list(l), np_data3, strides=1, sample=1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_l, train_a = [], []
        acc_list = []
        max_acc = 0
        avg_acc = 0
        while avg_acc < 0.9:
            for i in range(1000):
                acc_sum = 0
                dict_ir, dict_lable = build_dict_batch(ir_list, label_list, batch_num=100, sample_num=sample_size)
                sess.run(train_step, feed_dict={input_data: dict_ir, labels: dict_lable})
                loss_, acc_ = sess.run([loss, acc], feed_dict={input_data: dict_ir, labels: dict_lable})
                print("loss", loss_)
            avg_acc, max_acc = test_model(sess, np_data3, max_acc, baseline=0.54, save_model=True, saver=saver)
            print("acc:", avg_acc, "max:", max_acc, "loss:", loss_)

        avg_acc, max_acc = test_model(sess, np_data3, max_acc, baseline=0.9)
        print(avg_acc)

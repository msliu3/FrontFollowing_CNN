#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   plot_position_label.py    
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version
------------      -------    --------
2020/2/12 15:18   msliu      1.0      

@Description
------------
根据label标记出，不同地方的状态，用于查看第一部分label的训练结果
"""

from matplotlib import pyplot as plt
import re
import os
import math
import numpy as np


def extract_data(item_str):
    one_line = list(map(float, item_str.split("\t")))
    # one_line.pop()
    # print(one_line, "hehe")
    return np.array(one_line).reshape(1, -1)


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


def draw_label_predict(label_list, numpy_data, fig_num=100, strides=2, sample=4):
    """
    这个函数中的label_list是正好数量的label
    用于做预测的可视化
    :param label_list:
    :param numpy_data:
    :param fig_num:
    :param strides:
    :param sample:
    :return:
    """
    list_num = (numpy_data.shape[0] - (sample - strides)) // strides
    plt.figure(fig_num)
    stay = []
    forward = []
    right = []
    left = []
    for i in range(list_num - 1):
        X = -numpy_data[i * strides + sample, 3]
        Y = -numpy_data[i * strides + sample, 2]
        # X = [-i for i in X]
        # Y = list(map(lambda x: -x, Y))
        if label_list[i] == 0:
            stay.append([X, Y])
        elif label_list[i] == 1:
            forward.append([X, Y])
        elif label_list[i] == 2:
            right.append([X, Y])
        elif label_list[i] == 3:
            left.append([X, Y])
    stay = np.array(stay).reshape(-1, 2)
    forward = np.array(forward).reshape(-1, 2)
    right = np.array(right).reshape(-1, 2)
    left = np.array(left).reshape(-1, 2)
    plt.scatter(stay[:, 0], stay[:, 1], color="c", label="$stay$")
    plt.scatter(forward[:, 0], forward[:, 1], color="r", label="$forward$")
    plt.scatter(right[:, 0], right[:, 1], color="b", label="$right$")
    plt.scatter(left[:, 0], left[:, 1], color="g", label="$left$")
    plt.legend()

def draw_label_all(label_list, numpy_data, fig_num=100, strides=2, sample=4):
        """
        这个函数中的label_list是 全部数量 的label
        用于做正确数据的可视化

        :param label_list:
        :param numpy_data:
        :param fig_num:
        :param strides:
        :param sample:
        :return:
        """
        list_num = (numpy_data.shape[0] - (sample - strides)) // strides
        plt.figure(fig_num)
        stay = []
        forward = []
        right = []
        left = []
        for i in range(list_num - 1):
            X = -numpy_data[i * strides + sample, 3]
            Y = -numpy_data[i * strides + sample, 2]
            # X = [-i for i in X]
            # Y = list(map(lambda x: -x, Y))
            if label_list[i * strides + sample] == 0:
                stay.append([X, Y])
            elif label_list[i * strides + sample] == 1:
                forward.append([X, Y])
            elif label_list[i * strides + sample] == 2:
                right.append([X, Y])
            elif label_list[i * strides + sample] == 3:
                left.append([X, Y])
        stay = np.array(stay).reshape(-1, 2)
        forward = np.array(forward).reshape(-1, 2)
        right = np.array(right).reshape(-1, 2)
        left = np.array(left).reshape(-1, 2)
        plt.scatter(stay[:, 0], stay[:, 1], color="c", label="$stay$")
        plt.scatter(forward[:, 0], forward[:, 1], color="r", label="$forward$")
        plt.scatter(right[:, 0], right[:, 1], color="b", label="$right$")
        plt.scatter(left[:, 0], left[:, 1], color="g", label="$left$")
        plt.legend()


if __name__ == '__main__':
    data_path = "../resource/stop_move/data_combined_3.txt"
    data_np = read_data(data_path)
    labels = list(data_np[:, 0])
    draw_label_all(labels, data_np, strides=2, sample=4,fig_num=111)
    draw_label_all(labels, data_np, strides=1, sample=1)
    plt.show()

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   RNN.py    
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version
------------      -------    --------
2020/1/31 9:11   msliu      1.0      

@Description
------------
None
"""
import time

"""
This is a sample RNN code.
Please make some changes instead of submitting the original code.
If you have some questions about the code or data,
please contact Y.Q. Deng(yqdeng@cs.hku.hk) or M.M. Kuang(kuangmeng@hku.hk)
"""

import numpy as np
import tensorflow as tf

# Prepare Data(Training and Testing)
filename = "data.txt"
a_list = []
b_list = []
c_list = []


def str_2_list(data_list):
    ret_list = []
    for i in range(len(data_list)):
        tmp_list = data_list[i].strip().split(" ")
        tmp_ret_list = [int(tmp_list[7][0]), int(tmp_list[6]), int(tmp_list[5]), int(tmp_list[4]), int(tmp_list[3]),
                        int(tmp_list[2]), int(tmp_list[1]), int(tmp_list[0][1])]  # changed
        ret_list.append(tmp_ret_list)
    return ret_list


with open(filename, "r") as file:
    filein = file.read().splitlines()
    for item in filein:
        tmp_list = item.strip().split(",")
        a_list.append(tmp_list[0])
        b_list.append(tmp_list[1])
        c_list.append(tmp_list[2])
a_list = str_2_list(a_list)
b_list = str_2_list(b_list)
c_list = str_2_list(c_list)

# Define the dataflow graph
time_steps = 8  # time steps which is the same as the length of the bit-string
input_dim = 2  # number of units in the input layer
hidden_dim = 16  # number of units in the hidden layer
output_dim = 1  # number of units in the output layer
binary_dim = 8
largest_number = pow(2, binary_dim)

tf.reset_default_graph()
# input X and target ouput Y
X = tf.placeholder(tf.float32, [None, time_steps, input_dim], name='x')
Y = tf.placeholder(tf.float32, [None, time_steps], name='y')

# define the RNN cell: can be simple cell, LSTM or GRU
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation=tf.nn.sigmoid)

# values is a tensor of shape [batch_size, time_steps, hidden_dim]
# last_state is a tensor of shape [batch_size, hidden_dim]
values, last_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
values = tf.reshape(values, [time_steps, hidden_dim])

# put the values from the RNN through fully-connected layer
W = tf.Variable(tf.random_uniform([hidden_dim, output_dim], minval=-1.0, maxval=1.0), name='W')
b = tf.Variable(tf.zeros([1, output_dim]), name='b')

# h = tf.nn.sigmoid(tf.matmul(values, W) + b, name='h')
# h = tf.nn.softmax(tf.matmul(values, W) + b)
h = tf.matmul(values, W) + b

# minimize loss, using ADAM as weight update rule
h_ = tf.reshape(h, [1,time_steps])
Y_ = tf.reshape(Y, [1,time_steps])

# temp = -Y_ * tf.log(h_) - (1 - Y_) * tf.log(1 - h_)
# loss = tf.reduce_mean(temp, name='loss')

temp_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_,logits=h_)
loss = tf.reduce_mean(temp_loss)
# train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
# loss = tf.reduce_mean(tf.nn.sigmod(labels=Y,logits=Y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss,gate_gradients=tf.train.Optimizer.GATE_NONE)

# Launch the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())
start = time.time()

# a = np.array(a_list[0], dtype=np.uint8)  # changed
# b = np.array(b_list[0], dtype=np.uint8)  # changed
# c = np.array(c_list[0], dtype=np.uint8)  # changed
# ab = np.c_[a, b]
# x = np.array(ab).reshape([1, binary_dim, 2])
# y = np.array(c).reshape([1, binary_dim])
# print(x,y)
# accuracy = sess.run(loss, feed_dict={X: x, Y: y})
# print(sess.run(h,{X: x, Y: y}))
# print(sess.run(Y,{X: x, Y: y}))
# print(accuracy)
# sess.run(train_step, {X: x, Y: y})

with tf.device("/GPU:0"):
    # Number of data for training and testing
    # Please remember the total number is 5000
    num4train = 10
    num4test = 10
    epoch = 0
    accuracy = 1
    # train
    # while accuracy>0.1:
    #     epoch+=1
    for i in range(100):
        for j in range(num4train):
            a = np.array(a_list[j], dtype=np.uint8)  # changed
            b = np.array(b_list[j], dtype=np.uint8)  # changed
            c = np.array(c_list[j], dtype=np.uint8)  # changed
            ab = np.c_[a, b]
            x = np.array(ab).reshape([1, binary_dim, 2])
            y = np.array(c).reshape([1, binary_dim])
            # print(x,y)

            temp_h,temp_y = sess.run([h_, Y_], feed_dict={X: x, Y: y})
            print("h: ",temp_h)
            print("y: ",temp_y)
            print("temp: ",sess.run(temp_loss, feed_dict={X: x, Y: y}))
            print("loss: ",sess.run(loss, feed_dict={X: x, Y: y}))

            # if j % 100 == 0:
            #     print("Epoch: %d Example: %d is running..." % (epoch, j))
            #     accuracy = sess.run(loss, feed_dict={X: x, Y: y})
            #     print(accuracy)

            #     # if accuracy < 0.001:
            #     #     break;
            sess.run(train_step, {X: x, Y: y})

    remain_result = []
    time_result = time.time()-start
    print("10 loops: {:0.2f}ms".format(1000*time_result))

# Test

# for i in range(num4train + 1, num4train + num4test):
#     a = np.array(a_list[i], dtype=np.uint8)  # changed
#     b = np.array(b_list[i], dtype=np.uint8)  # changed
#     c = np.array(c_list[i], dtype=np.uint8)  # changed
#     ab = np.c_[a, b]
#     x = np.array(ab).reshape([1, binary_dim, 2])
#     y = np.array(c).reshape([1, binary_dim])
#
#     # get predicted value
#     [_probs, _loss] = sess.run([h, loss], {X: x, Y: y})
#     probs = np.array(_probs).reshape([8])
#     prediction = np.array([1 if p >= 0.5 else 0 for p in probs]).reshape([8])
#     # Save the result
#     remain_result.append([prediction, y[0]])
#
#     # calculate error
#     error = np.sum(np.absolute(y - probs))
#
#     # print the prediction, the right y and the error.
#     print("---------------")
#     print(prediction)
#     print(y[0])
#     print(error)
#     print("---------------")
#     print()

sess.close()

# Get the total accuracy (Please don't change this part)

accuracy = 0
# for i in range(len(remain_result)):
#     len_ = len(remain_result[i][0])
#     tmp_num = 0
#     for j in range(len_):
#         if remain_result[i][0][j] == remain_result[i][1][j]:
#             tmp_num += 1
#     accuracy += tmp_num / len_

accuracy /= len(remain_result)
print(accuracy)


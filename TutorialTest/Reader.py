#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Reader.py    
@Contact :   liumingshanneo@163.com

@Modify Time      @Author    @Version
------------      -------    --------
2020/2/10 15:50   msliu      1.0      

@Description
------------
None
"""

# import lib
import tensorflow as tf
import numpy as np
import os
tf.enable_eager_execution()

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
data_path_0 = father_path + os.path.sep + "resource" + os.path.sep + "1-20_19.34直行" + os.path.sep + "leg_odo.txt"
print(data_path_0)
dataset = tf.data.TextLineDataset(data_path_0)
print("hehe")
data = dataset.take(2)
for item in data:
    print(item.numpy().decode())

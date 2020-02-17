import sys, os
import time
import numpy as np
import math

# 温度信息读取
file = open("./ir_data.txt")
list_ir_data = file.readlines()
lists = []
for lines in list_ir_data:
    lines = lines.strip("\n")
    lines = lines.strip('[')
    lines = lines.strip(']')
    lines = lines.split(", ")
    lists.append(lines)
file.close()
array_ir_data = np.array(lists)
rows_ir_data = array_ir_data.shape[0]
columns_ir_data = array_ir_data.shape[1]
# print(columns_ir_data)
ir_data = np.zeros((rows_ir_data, columns_ir_data))
for i in range(rows_ir_data):
    for j in range(columns_ir_data):
        ir_data[i][j] = float(array_ir_data[i][j])
# print(ir_data[0][0])
# print(type(ir_data[0][0]))
# print(ir_data[0][columns_ir_data-1])

# 坐标信息读取
file = open("./leg_odo.txt")
list_leg_odo = file.readlines()
lists = []
for lines in list_leg_odo:
    lines = lines.strip("\n")
    lines = lines.strip('[')
    lines = lines.strip(']')
    lines = lines.split("\t")
    lists.append(lines)
file.close()
array_leg_odo = np.array(lists)
# print(array_leg_odo[0][0])
# print(type(array_leg_odo[0][0]))
rows_leg_odo = array_leg_odo.shape[0]
columns_leg_odo = array_leg_odo.shape[1]
leg_odo = np.zeros((rows_leg_odo, columns_leg_odo))
for i in range(rows_leg_odo):
    for j in range(columns_leg_odo):
        leg_odo[i][j] = float(array_leg_odo[i][j])
for i in range(rows_leg_odo):
    leg_odo[i][3] = math.degrees(leg_odo[i][3])
# print(leg_odo[0][0])
# print(type(leg_odo[0][0]))


def difference_list(list_data):
    dif_list = []
    # 第一个用0写入
    dif_list.append(0)
    for i in range(1, len(list_data) - 1):
        if list_data[i] * list_data[i - 1] < 0 and abs(list_data[i]) + abs(list_data[i - 1]) > 180:
            # print("into", list_data[i-1], list_data[i],end=" ")
            dif = (list_data[i - 1] / abs(list_data[i - 1])) * 180 - list_data[i - 1]
            dif += list_data[i] - (list_data[i] / abs(list_data[i])) * 180
            # print(dif)
        else:
            dif = list_data[i] - list_data[i - 1]
        dif_list.append(dif)
        # 最后一个用倒数第二个插值
    dif_list.append(dif_list[-1])
    return dif_list


# 以温度信息第一位作基准，按序检查时间最相近的位置信息
# 所有温度信息第一位都要比位置信息晚
# 只需要第一位确认了之后，后续温度信息和位置信息的时间戳基本满足最相近

flag = 0  # 记录第flag位位置信息开始进行匹配
error = abs(ir_data[0][0] - leg_odo[0][0])
for i in range(rows_leg_odo):
    error_new = abs(ir_data[0][0] - leg_odo[flag][0])
    if error_new > error:
        flag -= 1
        break
    else:
        error = error_new
        flag += 1
print(flag)

# 数据合并 形成新的txt 时间戳以温度为准
# 确定行列数
rows_new = min(rows_ir_data, rows_leg_odo-flag)
columns_new = columns_ir_data + columns_leg_odo  # 两者数据去除时间戳加入Walker角度差的计算到第二列，加入label到第一列
data_combined = np.zeros((rows_new, columns_new))

# 数据结构： [0]label [1]Walker角度差 [2...11]leg_odo [12...]ir_data
if rows_new == rows_ir_data:
    # 时间戳
    data_combined[:, 2:columns_leg_odo+1] = leg_odo[flag:(rows_new + flag), 1:columns_leg_odo]
    data_combined[:, columns_leg_odo+1:columns_new+1] = ir_data[:, 1:columns_ir_data]
    difference_list_theta = data_combined[:, 4].tolist()
    difference_list_theta = difference_list(difference_list_theta)
    difference_array = np.array(difference_list_theta)
    data_combined[:, 1] = difference_array
else:
    data_combined[:, 2:columns_leg_odo+1] = leg_odo[flag:(rows_new + flag), 1:columns_leg_odo]
    data_combined[:, columns_leg_odo+1:columns_new+1] = ir_data[0:rows_new, 1:columns_ir_data]
    difference_list_theta = data_combined[:, 4].tolist()
    difference_list_theta = difference_list(difference_list_theta)
    difference_array = np.array(difference_list_theta)
    data_combined[:, 1] = difference_array

# 根据角度变化以及车的dX数据判断运动情况
for i in range(rows_new):
    if abs(data_combined[i][5]) < 0.01 and abs(data_combined[i][1]) < 11: #(角度(这个11的阈值随便设的，用来以后原地转的标类)没变化，位置也没变化)原地
        data_combined[i][0] = 0
    elif abs(data_combined[i][1]) < 2.2:  #直行前进
        data_combined[i][0] = 1
    # elif abs(data_combined[i][1]) < 2.2:  #直行后退
    #     data_combined[i][0] = 1.5
    elif data_combined[i][1] < 0:  #右转
        data_combined[i][0] = 2
    elif data_combined[i][1] > 0:  #左转
        data_combined[i][0] = 3

# 用于计算阈值
# flag_count = np.zeros((2, 1))
# for i in range(rows_new):
#     if abs(data_combined[i][0]) > 2.2: #阈值
#         flag_count[1][0] = flag_count[1][0] + 1
#     else:
#         flag_count[0][0] = flag_count[0][0] + 1
# difference_max = max(abs(data_combined[:, 1]))
# print(difference_max)
# print(flag_count)

# 保存txt
#np.savetxt("./data_combined.txt", data_combined, fmt='%.07f' + (columns_leg_odo-1)*' %.06f' + (columns_ir_data-1) * '%.02f', delimiter='\t')
np.savetxt("./data_combined.txt", data_combined, fmt='%f', delimiter='\t')

# 去除头尾
# 头部
for i in range(rows_new):
    if data_combined[i][0] == 0:
        continue
    elif i == 0:
        break
    else:
        data_combined = np.delete(data_combined, range(i), axis=0)
        rows_new = data_combined.shape[0]
        break
# 尾部
for i in range(-1, -rows_new, -1):
    if data_combined[i][0] == 0:
        continue
    elif i == 0:
        break
    else:
        data_combined = np.delete(data_combined, range(rows_new+i+1, rows_new), axis=0)
        rows_new = data_combined.shape[0]
        break

# 保存txt
#np.savetxt("./data_combined.txt", data_combined, fmt='%.07f' + (columns_leg_odo-1)*' %.06f' + (columns_ir_data-1) * '%.02f', delimiter='\t')
np.savetxt("./data_combined_0.txt", data_combined, fmt='%f', delimiter='\t')








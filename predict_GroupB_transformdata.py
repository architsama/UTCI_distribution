# 导入所需要的模块
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
from torch.autograd.variable import Variable

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import time
import model_M


pre_path = r'predict\predict_test.csv'
data_transform = r'predict\predict_transform.csv'
SEED = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model_M.model_13().to(device)
##predict_test 转化为group B的可输入数据，输入数据不打乱顺序

df = pd.read_table(pre_path, sep=',')
df = df.dropna()
ar = np.array(df)
ar = ar[~np.isnan(ar).any(axis=1)]
build_s_1 = ar[:, :900].reshape(-1, 30, 30)
build_v_1 = ar[:, 900:].reshape(-1, 30, 30).transpose(0, 2, 1).reshape(-1, 30, 30)

build_s_1 = np.pad(build_s_1, (20))[20:-20, :, :]  # 三维扩充20，在去掉前20和后20的全0数据
w = np.zeros([1, 82])  # 设定第一行
for m in range(30):
    loc_x = m + 1
    for i in range(30):
        loc_y = i + 1
        case_value = build_v_1[0:1, loc_x - 1:loc_x, loc_y - 1:loc_y]
        case_shape = build_s_1[0:1, loc_x + 20 - 4:loc_x + 20 + 5,
                     loc_y + 20 - 4:loc_y + 20 + 5]
        output = case_value.reshape(1)
        inputdata = case_shape.reshape(81)
        data = np.concatenate((inputdata, output))
        w = np.append(w, [data], axis=0)
##经过预测
data_to_save = w

with open(data_transform, 'w+') as f: #'ab'是追加数据w+:打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
    np.savetxt(f, data_to_save, fmt="%3f",delimiter=",")

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

import data_m
import model_M


print('模块导入成功')
print('设置超参数')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

radom_see= data_m.radom_seed #单词写错了，后面不修改
#
#
#
#这个地方进行修改
#NET_num = 10
#model = model_M.model_10().to(device)
#
#
#
batch_size = 256
path = r'dataset\UTCI_s_r.csv'
file_path_forM1=r'dataset\UTCI_forM1.csv'
file_path_forM2=r'dataset\UTCI_forM2.csv'
file_path_forM3=r'dataset\UTCI_forM3.csv'

def casestart(NET_num):
    if NET_num == 1:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPA_mlp(file_path_forM1)  # 1 2
        model = model_M.model_1().to(device)
    if NET_num == 2:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPA_mlp(file_path_forM1)  # 1 2
        model = model_M.model_2().to(device)
    if NET_num == 3:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPA_cnn(file_path_forM1)  # 3 4 5
        model = model_M.model_3().to(device)
    if NET_num == 4:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPA_cnn(file_path_forM1)  # 3 4 5
        model = model_M.model_4().to(device)
    if NET_num == 5:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPA_cnn(file_path_forM1)  # 3 4 5 6
        model = model_M.model_5().to(device)
    if NET_num == 6:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPA_cnn(file_path_forM1)  # 3 4 5 6
        model = model_M.model_6().to(device)

    if NET_num == 11:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPB_mlp(file_path_forM2)  # 3 4 5 6
        model = model_M.model_11().to(device)
    if NET_num == 12:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPB_mlp(file_path_forM2)  # 3 4 5 6
        model = model_M.model_12().to(device)
    if NET_num == 13:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPB_cnn(file_path_forM2)  # 3 4 5 6
        model = model_M.model_13().to(device)
    if NET_num == 14:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPB_cnn(file_path_forM2)  # 3 4 5 6
        model = model_M.model_14().to(device)
    if NET_num == 15:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPB_cnn(file_path_forM2)  # 3 4 5 6
        model = model_M.model_15().to(device)
    if NET_num == 16:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPB_cnn(file_path_forM2)  # 3 4 5 6
        model = model_M.model_16().to(device)

    else:
        print('NET_NUM超出范围')
    model_name = r'result\model%d' % NET_num
    print('开始计算case%d'%NET_num)
    return train_xt, train_yt, test_xt, test_yt, model, model_name
def Tensorloader(train_xt, train_yt, test_xt, test_yt):
    train_data = Data.TensorDataset(train_xt, train_yt) #加载到训练数据
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True, num_workers=0)

    test_data = Data.TensorDataset(test_xt, test_yt)
    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=0)

    print("data load successfully将训练数据为数据加载器")
    return train_loader,test_loader

def cal_result(w):
    train_xt, train_yt, test_xt, test_yt, model, model_name = casestart(NET_num=w)
    train_loader, test_loader = Tensorloader(train_xt, train_yt, test_xt, test_yt)
    model.load_state_dict(torch.load(model_name))
    loss_func = nn.MSELoss(reduce=True, size_average=True)  # 均方根误差损失函数
    train_loss_all = []
    train_loss_all_r2 = []
    test_loss_all = []
    test_loss_all_r2 = []


    train_loss = 0
    loss_r2_list = 0
    train_num = 0
    test_loss = 0
    loss_r2_pre_list = 0
    test_num = 0


    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = Variable(b_x).to(device), Variable(b_y).to(device)
        # print(b_x.size(), b_y.size())
        output = model(b_x)  # MLP在训练batch上的输出
        loss_MSE = loss_func(output, b_y)  # 均方根误差损失函数
        xxx = output.cpu().detach().numpy()
        yyy = b_y.cpu().detach().numpy()
        loss_r2 = r2_score(xxx.flatten(), yyy.flatten())  # r2_score损失函数
        print( step, loss_MSE, loss_r2)
        loss_r2_list += loss_r2.item() * b_x.size(0)
        train_loss += loss_MSE.item() * b_x.size(0)
        train_num += b_x.size(0)
    train_loss_all.append(train_loss / train_num)
    train_loss_all_r2.append(loss_r2_list / train_num)

    for step, (b_x_pre, b_y_pre) in enumerate(test_loader):
        b_x_pre, b_y_pre = Variable(b_x_pre).to(device), Variable(b_y_pre).to(device)
        # print(b_x.size(), b_y.size())
        output_pre = model(b_x_pre)  # MLP在训练batch上的输出
        loss_pre = loss_func(output_pre, b_y_pre)  # 均方根误差损失函数
        xxx = output_pre.cpu().detach().numpy()
        yyy = b_y_pre.cpu().detach().numpy()
        loss_r2_pre = r2_score(xxx.flatten(), yyy.flatten())  # r2_score损失函数
        print('test', step, loss_pre, loss_r2_pre)

        loss_r2_pre_list += loss_r2_pre.item() * b_x_pre.size(0)
        test_loss += loss_MSE.item() * b_x_pre.size(0)
        test_num += b_x_pre.size(0)
    test_loss_all.append(test_loss / test_num)
    test_loss_all_r2.append(loss_r2_pre_list / test_num)

    final_train_r2 = loss_r2_list / train_num
    final_test_r2 = loss_r2_pre_list / test_num

    final_train_MSE = train_loss / train_num
    final_test_MSE = test_loss / test_num

    print('train:', 'trainloss:', train_loss / train_num, 'trainr2:', final_train_r2)
    print('test', 'testloss:', test_loss / test_num, 'testr2:', final_test_r2)

    writelog = [w, final_train_r2, final_test_r2, final_train_MSE, final_test_MSE]
    with open('result_file.txt', 'a') as f:
        f.writelines(str(writelog))
    print('计算完毕')
for i in [
          6
          ]:
    w = i
    cal_result(w)


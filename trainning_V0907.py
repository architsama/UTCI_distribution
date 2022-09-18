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

batch_size = 512
running_time = 300
# start = time.clock()


radom_see= data_m.radom_seed #单词写错了，后面不修改
#
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


    if NET_num == 21:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPC_mlp(file_path_forM3)  # 3 4 5 6
        model = model_M.model_21().to(device)
    if NET_num == 22:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPC_mlp(file_path_forM3)  # 3 4 5 6
        model = model_M.model_22().to(device)
    if NET_num == 23:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPC_cnn(file_path_forM3)  # 3 4 5 6
        model = model_M.model_23().to(device)
    if NET_num == 24:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPC_cnn(file_path_forM3)  # 3 4 5 6
        model = model_M.model_24().to(device)
    if NET_num == 25:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPC_cnn(file_path_forM3)  # 3 4 5 6
        model = model_M.model_25().to(device)
    if NET_num == 26:
        train_xt, train_yt, test_xt, test_yt = data_m.data_preGROUPC_cnn(file_path_forM3)  # 3 4 5 6
        model = model_M.model_26().to(device)




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
def train():
    print('开始执行训练')
    loss_func = nn.MSELoss(reduce=True, size_average=True)  # 均方根误差损失函数
    train_loss_all = []
    train_loss_all_r2 = []

    test_loss_all = []
    test_loss_all_r2 = []



    for epoch in range(running_time):
        train_loss = 0
        loss_r2_list = 0
        train_num = 0

        test_loss = 0
        loss_r2_pre_list = 0
        test_num = 0

        ## 对训练数据的迭代器进行迭代计算
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x, b_y = Variable(b_x).to(device), Variable(b_y).to(device)
            # print(b_x.size(), b_y.size())
            output = model(b_x)  # MLP在训练batch上的输出
            loss_MSE = loss_func(output, b_y)  # 均方根误差损失函数
            xxx = output.cpu().detach().numpy()
            yyy = b_y.cpu().detach().numpy()
            loss_r2 = r2_score(xxx.flatten(), yyy.flatten())  # r2_score损失函数
            print(epoch, step, loss_MSE,loss_r2)

            optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
            loss_MSE.backward()  # 损失的后向传播，计算梯度
            optimizer.step()  # 使用梯度进行优化

            loss_r2_list += loss_r2.item() * b_x.size(0)
            train_loss += loss_MSE.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        train_loss_all_r2.append(loss_r2_list/ train_num)

        for step, (b_x_pre, b_y_pre) in enumerate(test_loader):
            b_x_pre, b_y_pre = Variable(b_x_pre).to(device), Variable(b_y_pre).to(device)
            # print(b_x.size(), b_y.size())
            output_pre = model(b_x_pre)  # MLP在训练batch上的输出
            loss_pre = loss_func(output_pre, b_y_pre)  # 均方根误差损失函数
            xxx = output_pre.cpu().detach().numpy()
            yyy = b_y_pre.cpu().detach().numpy()
            loss_r2_pre = r2_score(xxx.flatten(), yyy.flatten())  # r2_score损失函数
            print('test',epoch, step, loss_pre,loss_r2_pre)

            loss_r2_pre_list += loss_r2_pre.item() * b_x_pre.size(0)
            test_loss += loss_MSE.item() * b_x_pre.size(0)
            test_num += b_x_pre.size(0)
        test_loss_all.append(test_loss / test_num)
        test_loss_all_r2.append(loss_r2_pre_list/ test_num)

        final_train_r2 = loss_r2_list/ train_num
        final_test_r2 = loss_r2_pre_list/ test_num

        final_train_MSE =train_loss / train_num
        final_test_MSE = test_loss / test_num

        print('train:',epoch,'trainloss:',train_loss / train_num,'trainr2:',final_train_r2)
        print('test','testloss:',epoch,test_loss / test_num,'testr2:',final_test_r2)



    # 模型保存
    torch.save(model.state_dict(), model_name)
    return train_loss_all_r2,test_loss_all_r2,final_train_r2,final_test_r2,train_loss_all,test_loss_all,final_train_MSE,final_test_MSE

def draw_modelpic(train_loss_all_r2,test_loss_all_r2):
    print('开始绘制收敛图像')
    plt.figure(figsize=(12, 8))
    #plt.plot(train_loss_all, "ro-", label="Train loss")
    plt.plot(train_loss_all_r2, "bx-", label="Train R2 Score")
    plt.plot(test_loss_all_r2, "rx-", label="Test R2 Score")

    #plt.axhline(y=final_r2, ls='--', c='blue',label='Final train R2 score = %d'%float("{:.2f}".format(final_r2)))  # 添加水平线
    #plt.axhline(y=final_r2_predict, ls='--', c='red',label='Final test R2 score = %d'%float("{:.2f}".format(final_r2_predict)))  # 添加水平线
    #error_r2 = float("{:.2f}".format(final_r2))
    #ax.annotate('%d'%final_r2_predict, xytext=(running_time-5,final_r2_predict))

    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel("Epoch", size=13)
    plt.ylabel("R2 Score", size=13)
    plt.ylim((0,1))
    plt.savefig(model_name+'_R2')
    #plt.show()
    with open(model_name+'loss_R2.txt','a') as f:
        f.writelines(str(train_loss_all_r2+test_loss_all_r2))
    print('收敛图像完成')

def draw_modelpic_MSE(train_loss_all,test_loss_all):
    print('开始绘制收敛图像')
    plt.figure(figsize=(12, 8))
    #plt.plot(train_loss_all, "ro-", label="Train loss")
    plt.plot(train_loss_all, "bx-", label="Train loss")
    plt.plot(test_loss_all, "rx-", label="Test loss")

    #plt.axhline(y=final_r2, ls='--', c='blue',label='Final train R2 score = %d'%float("{:.2f}".format(final_r2)))  # 添加水平线
    #plt.axhline(y=final_r2_predict, ls='--', c='red',label='Final test R2 score = %d'%float("{:.2f}".format(final_r2_predict)))  # 添加水平线
    #error_r2 = float("{:.2f}".format(final_r2))
    #ax.annotate('%d'%final_r2_predict, xytext=(running_time-5,final_r2_predict))

    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel("Epoch", size=13)
    plt.ylabel("Loss", size=13)
    plt.ylim((0,1))
    plt.savefig(model_name+'_MSE')
    #plt.show()
    print('收敛图像完成')

    with open(model_name+'loss_MSE.txt','a') as f:
        f.writelines(str(train_loss_all+test_loss_all))
    print('收敛图像完成')




for i in [
          26
          # ,2,
          # 3,4,5  #方法一，数据增强
          # ,6, 7,
          # 8, 9, 10
          # , 11, 12, 13, 14, 15, 16, 17   #使用30*30输入，10*10输出
          # ,21 , 22, 23, 24, 25  #使用10*10输入输出
          ]:
    w = i  #w = i + 1
    train_xt, train_yt, test_xt, test_yt, model, model_name = casestart(NET_num=w)


    # model.load_state_dict(torch.load(model_name))    ##如果需要继续算，就需要点这里



    train_loader, test_loader = Tensorloader(train_xt, train_yt, test_xt, test_yt)
    model.train()
    print(model)
    print('model completed')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-04, weight_decay=0)
    print('优化器、误差设置完成')
    train_loss_all_r2,test_loss_all_r2,final_train_r2,final_test_r2,train_loss_all,test_loss_all,final_train_MSE,final_test_MSE= train()
    writelog = [w,'trainr2:', final_train_r2, 'testr2:',final_test_r2, 'trainloss:',final_train_MSE,'testloss:', final_test_MSE]


    with open('runninglog.txt','a') as f:
        f.writelines(str(writelog))
    draw_modelpic(train_loss_all_r2, test_loss_all_r2)
    draw_modelpic_MSE(train_loss_all, test_loss_all)


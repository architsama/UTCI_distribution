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
print('模块导入成功')
import model_M
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


modeln = '\model6'

csv_name ='predict' + modeln + '_predict.csv'
model_name = r'result'+ modeln
pic = r'predict\pic' + modeln


# class CnnNet(nn.Module):
#     def __init__(self):
#         super(CnnNet, self).__init__()
#
#         # 第一层神经网络，包括卷积层、线性激活函数、池化层
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
#             nn.ReLU()
#         )
#         # 全连接层(将神经网络的神经元的多维输出转化为一维)
#         self.fc1 = nn.Sequential(
#             nn.Linear(30*30*16, 30*30*8),  # 进行线性变换
#             nn.ReLU()  # 进行ReLu激活
#         )
#
#         # 输出层(将全连接层的一维输出进行处理)
#         self.fc2 = nn.Sequential(
#             nn.Linear(30*30*8, 30*30*8),
#             nn.ReLU()
#         )
#
#         # 将输出层的数据进行分类(输出预测值)
#         self.fc3 = nn.Linear(30*30*8, 900)
#
#     # 定义前向传播过程，输入为x
#     def forward(self, x):
#         x = self.conv1(x)
#         # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
#         x = x.view(x.size()[0], -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

# net = CnnNet().to(device)
net = model_M.model_6().to(device)   ####

net.load_state_dict(torch.load(model_name))

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        print('---------', self.submodule._modules.items())
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)
            print(module)
            x = module(x)
            print('name', name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


layer_name = ['conv1','conv2','conv3','conv4','conv5','conv6']

def get_feature(layer_name,c_index=0):

    exact_list = [layer_name]
    myexactor = FeatureExtractor(net, exact_list)

    pre_path = r'predict\predict_test.csv'
    SEED = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def per_data(pre_path):
        print('开始导入数据')
        df = pd.read_table(pre_path,sep=',')
        df = df.fillna(0)
        ar = np.array(df)
        pre_shape,pre_value = ar[:, 0:900], ar[:, 900:].reshape(-1,30,30).transpose(0,2,1).reshape(-1,900)
        return pre_shape,pre_value
    pre_shape,pre_value = per_data(pre_path)

    input_data = pre_shape


    input_data = input_data
    # 插入维度
    input_data = torch.from_numpy(input_data.astype(np.float32)).view(-1, 1, 30, 30).to(device)

    x = myexactor(input_data)





    c_index=0
    MATRIC= (30,30)
    # 特征输出可视化

    plt.figure(figsize=(24, 4))
    for i in range(8):  # 可视化了32通道
        ax = plt.subplot(1, 8, i + 1)
        #ax.set_title('Feature {}'.format(i), fontsize='xx-small', verticalalignment='baseline')
        ax.axis('off')
        # ax.set_title('new—conv1-image')
        plt.imshow(x[c_index].data.cpu().numpy()[0, i, :, :], cmap='binary')
    plt.savefig('predict\interpretability\int_'+layer_name+'.png')

    # 特征输出可视化
    plt.figure(figsize=(12, 9))
    plt.title(layer_name)
    plt.imshow(x[c_index].data.cpu().numpy()[0, 0, :, :], cmap='Oranges')
    plt.colorbar()
    plt.savefig('predict\interpretability\int_'+layer_name+'Random.png')

    # 特征输出可视化云图
    plt.figure(figsize=(12, 9))
    plt.title(layer_name)
    plt.imshow(x[c_index].data.cpu().numpy()[0, 1, :, :], cmap='Oranges')
    plt.colorbar()
    plt.savefig('predict\interpretability\int_'+layer_name+'Random_h.png')

    print('完成'+layer_name)

for i in layer_name:
    get_feature(i)
print('完成所有制图')
import torch.nn as nn
import torch.nn.functional as F

#model
# 1  '1_mlp', input = 900, hiddenlayer =30*30*8,900
#
# 2  '2_mlp', input = 900, hiddenlayer =30*30*8,15*15*8,900
#
# 3  '3_cnn', input = 30*30, conv1 =(1,8,k=3,s=1,p=1) pool1,2=(k=2,s=1,p=0);mlp 29*29*8,15*15*8,900
# 0.767/-0.198
# 4  '4_cnn', input = 30*30, conv1 =(1,8,k=9,s=1,p=4) pool1,2=(k=2,s=1,p=0);mlp 29*29*8,15*15*8,900
# 0.876/0.034



# 5  '5_cnn', input = 30*30, [conv1 =(1,1*2,k=3,s=1,p=1) pool1,2=(k=2,s=1,p=0)]*3;mlp 27*27*8,15*15*8,900
# 不修改
# 最后一层*3  0.652/0.271
# 最后一层*5  0.254/0.214  反而下降了
# 如果在多加几层卷积层  加到6  0.942/0.050   0.9655/0.0821
# 卷积6 mlp  27*27*8,15*15*8*10,900           0.9618/0.2292
#MLP加一层 27*27*8,15*15*8*5,15*15*8,900   突变到负数，但是测试集为0.22
#  卷积+ 12  如果把输入端改为10*10，  0.9356/0.7027  , 0.941/0.607
#  卷积按照3层的话，如果把输入端改为10*10，  0.944/0.6074
#
#  卷积1+mlp3 29*29*8, 15*15*8*5, 15*15*8*5，100    0.983/0.650
#  卷积1+mlp1 29*29*8，100    0.9547/0.637
#  卷积1+mlp5#29*29*8, 15*15*8*5, 15*15, 15*15, 15*15，100   0.9776/0.6483
#  卷积3+mlp5#29*29*8, 15*15*8*5, 15*15, 15*15, 15*15，100   0.9812/0.3350   0.9587/0.3257
#
#  卷积0+mlp5 30*30*8, 15*15*8*5, 15*15, 15*15, 15*15，100    0.2390/-130.6243
#  卷积0+mlp5 30*30*8, 15*15*8*5, 15*15*8, 15*15*4, 15*15，100    0.2390/-130.6243    -0.1185/-152.470
######  以上train 2421
#  卷积12+mlp3     0.9711/0.7746   train 2814


# 6  '6_mlp', input = 81, hiddenlayer=8*8*8,1
#
# 7  '7_mlp', input = 81, hiddenlayer=8*8*8,3*3*8,1
#
# 8  '8_cnn', input = 9*9, conv1 =(1,8,k=3,s=1,p=1) pool1,2=(k=2,s=1,p=0);mlp 8*8*8,3*3*8,1
#
# 9  '9_cnn', input = 9*9, conv1 =(1,8,k=9,s=1,p=4) pool1,2=(k=2,s=1,p=0);mlp 8*8*8,3*3*8,1
#
# 10 '10_cnn', input = 9*9, [conv1 =(1,1*2,k=3,s=1,p=1) pool1,2=(k=2,s=1,p=0)]*3;mlp 8*8*8,3*3*8,1
#

class model_1(nn.Module):
    def __init__(self):
        super(model_1, self).__init__()
        ## 定义第一个隐藏层
        self.hidden1 = nn.Linear(in_features=900,
                                 out_features=30*30*30, bias=True)
        ## 回归预测层
        self.predict = nn.Linear(30*30*30, 900)
    ## 定义网络的向前传播路径
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.predict(x)
        return x

class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        ## 定义第一个隐藏层
        self.hidden1 = nn.Linear(in_features=900,
                                 out_features=30*30*16, bias=True)
        ## 定义第二个隐藏层
        self.hidden2 = nn.Linear(30*30*16, 30*30*8)
        self.hidden3 = nn.Linear(30 * 30*8, 30 * 30 * 8)
        self.predict = nn.Linear(30*30*8, 900)
#（4608,2304）
    ## 定义网络的向前传播路径
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)
        ## 输出一个一维向量
        return x

class model_3(nn.Module):
    def __init__(self):
        super(model_3, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #input = 9  output = 9
        # # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=(0, 0), dilation=1, ceil_mode=False)
        #input = 9 output = 8

        # 定义全连接层
        self.fc1 = nn.Linear(30*30*16, 30*30*8)
        self.fc2 = nn.Linear(30*30*8, 30*30*8)
        self.fc3 = nn.Linear(30*30*8, 900)
        #self.dropout = nn.Dropout(p=0.05)  # dropout训练#如果没有dropout，0.974/0.9678，如果有dropout 0.5为0.7508/0.4963
    def forward(self, x):
        x = F.relu((self.conv1(x))).view(-1, 1, 30*30*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        return x

class model_4(nn.Module):
    def __init__(self):
        super(model_4, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        #input = 9  output = 9
        # # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=(0, 0), dilation=1, ceil_mode=False)
        #input = 9 output = 8

        # 定义全连接层
        self.fc1 = nn.Linear(30*30*16, 30*30*8)
        self.fc2 = nn.Linear(30*30*8, 30*30*8)
        self.fc3 = nn.Linear(30*30*8, 900)
        #self.dropout = nn.Dropout(p=0.05)  # dropout训练#如果没有dropout，0.974/0.9678，如果有dropout 0.5为0.7508/0.4963
    def forward(self, x):
        x = F.relu((self.conv1(x))).view(-1, 1, 30*30*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        return x

class model_5(nn.Module):
    def __init__(self):
        super(model_5, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #input = 9  output = 9
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=(0, 0), dilation=1, ceil_mode=False)
        #input = 9 output = 8

        # 定义全连接层
        self.fc1 = nn.Linear(30*30*16, 30*30*8)
        self.fc2 = nn.Linear(30*30*8, 30*30*8)
        self.fc3 = nn.Linear(30 * 30 * 8, 900)
        #self.dropout = nn.Dropout(p=0.05)  # dropout训练#如果没有dropout，0.974/0.9678，如果有dropout 0.5为0.7508/0.4963
    def forward(self, x):
        x = F.relu((self.conv1(x))) #8
        x = F.relu((self.conv2(x))) #7
        x = F.relu((self.conv3(x))).view(-1, 1,30*30*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        return x

class model_6(nn.Module):
    def __init__(self):
        super(model_6, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 定义全连接层
        self.fc1 = nn.Linear(30*30*16, 30*30*8)
        self.fc2 = nn.Linear(30*30*8, 30*30*8)
        self.fc3 = nn.Linear(30*30*8, 900)
    def forward(self, x):
        x = F.relu((self.conv1(x))) #29
        x = F.relu((self.conv2(x))) #28
        x = F.relu((self.conv3(x)))
        x = F.relu((self.conv4(x)))
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv6(x))).view(-1, 1,30*30*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

##GROUP b
class model_11(nn.Module):
    def __init__(self):
        super(model_11, self).__init__()
        ## 定义第一个隐藏层
        self.hidden1 = nn.Linear(in_features=81,
                                 out_features=9*9*8, bias=True)
        ## 回归预测层
        self.predict = nn.Linear(9*9*8, 1)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.predict(x)
        return x

class model_12(nn.Module):
    def __init__(self):
        super(model_12, self).__init__()
        ## 定义第一个隐藏层
        self.hidden1 = nn.Linear(in_features=81,
                                 out_features=9*9*16, bias=True)
        ## 定义第二个隐藏层
        self.hidden2 = nn.Linear(9*9*16, 9*9)
        self.hidden3 = nn.Linear(9*9, 9*9)
        self.predict = nn.Linear(9*9, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)
        ## 输出一个一维向量
        return x

class model_13(nn.Module):
    def __init__(self):
        super(model_13, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # 定义全连接层
        self.fc1 = nn.Linear(9*9*16, 9*9)
        self.fc2 = nn.Linear(9*9, 9*9)
        self.fc3 = nn.Linear(9*9, 1)
        #self.dropout = nn.Dropout(p=0.05)  # dropout训练#如果没有dropout，0.974/0.9678，如果有dropout 0.5为0.7508/0.4963
    def forward(self, x):
        x = F.relu((self.conv1(x))).view(-1, 1, 9*9*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        return x

class model_14(nn.Module):
    def __init__(self):
        super(model_14, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))

        # 定义全连接层
        self.fc1 = nn.Linear(9*9*16, 9*9)
        self.fc2 = nn.Linear(9*9, 9*9)
        self.fc3 = nn.Linear(9*9, 1)
    def forward(self, x):
        x = F.relu((self.conv1(x))).view(-1, 1, 9*9*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        return x

class model_15(nn.Module):
    def __init__(self):
        super(model_15, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # 定义全连接层
        self.fc1 = nn.Linear(9*9*16, 9*9)
        self.fc2 = nn.Linear(9*9, 9*9)
        self.fc3 = nn.Linear(9*9, 1)
    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x))).view(-1, 1, 9 * 9 * 16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        return x

class model_16(nn.Module):
    def __init__(self):
        super(model_16, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # 定义全连接层
        self.fc1 = nn.Linear(9*9*16, 9*9)
        self.fc2 = nn.Linear(9*9, 9*9)
        self.fc3 = nn.Linear(9*9, 1)
    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = F.relu((self.conv4(x)))
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv6(x))).view(-1, 1, 9 * 9 * 16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

##group c
class model_21(nn.Module):
    def __init__(self):
        super(model_21, self).__init__()
        ## 定义第一个隐藏层
        self.hidden1 = nn.Linear(in_features=29*29,
                                 out_features=29*29*16, bias=True)
        ## 回归预测层
        self.predict = nn.Linear(29*29*16, 1)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.predict(x)
        return x

class model_22(nn.Module):
    def __init__(self):
        super(model_22, self).__init__()
        ## 定义第一个隐藏层
        self.hidden1 = nn.Linear(in_features=29*29,
                                 out_features=29*29*16, bias=True)
        ## 定义第二个隐藏层
        self.hidden2 = nn.Linear(29*29*16, 29*29*8)
        self.hidden3 = nn.Linear(29*29*8, 29*29 * 8)
        self.predict = nn.Linear(29*29*8, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)
        ## 输出一个一维向量
        return x

class model_23(nn.Module):
    def __init__(self):
        super(model_23, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #input = 9  output = 9
        # # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=(0, 0), dilation=1, ceil_mode=False)
        #input = 9 output = 8

        # 定义全连接层
        self.fc1 = nn.Linear(29*29*16, 29*29*8)
        self.fc2 = nn.Linear(29*29*8, 29*29*8)
        self.fc3 = nn.Linear(29*29*8, 1)
        #self.dropout = nn.Dropout(p=0.05)  # dropout训练#如果没有dropout，0.974/0.9678，如果有dropout 0.5为0.7508/0.4963
    def forward(self, x):
        x = F.relu((self.conv1(x))).view(-1, 1, 29*29*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        return x

class model_24(nn.Module):
    def __init__(self):
        super(model_24, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        #input = 9  output = 9
        # # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=(0, 0), dilation=1, ceil_mode=False)
        #input = 9 output = 8

        # 定义全连接层
        self.fc1 = nn.Linear(29*29*16, 29*29*8)
        self.fc2 = nn.Linear(29*29*8, 29*29*8)
        self.fc3 = nn.Linear(29*29*8,1)
        #self.dropout = nn.Dropout(p=0.05)  # dropout训练#如果没有dropout，0.974/0.9678，如果有dropout 0.5为0.7508/0.4963
    def forward(self, x):
        x = F.relu((self.conv1(x))).view(-1, 1, 29*29*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        return x

class model_25(nn.Module):
    def __init__(self):
        super(model_25, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #input = 9  output = 9
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=(0, 0), dilation=1, ceil_mode=False)
        #input = 9 output = 8

        # 定义全连接层
        self.fc1 = nn.Linear(29*29*16, 29*29*8)
        self.fc2 = nn.Linear(29*29*8, 29*29*8)
        self.fc3 = nn.Linear(29*29* 8, 1)
        #self.dropout = nn.Dropout(p=0.05)  # dropout训练#如果没有dropout，0.974/0.9678，如果有dropout 0.5为0.7508/0.4963
    def forward(self, x):
        x = F.relu((self.conv1(x))) #8
        x = F.relu((self.conv2(x))) #7
        x = F.relu((self.conv3(x))).view(-1, 1,29*29*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dropout(x)
        return x

class model_26(nn.Module):
    def __init__(self):
        super(model_26, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 定义全连接层
        self.fc1 = nn.Linear(29*29*16, 29*29*8)
        self.fc2 = nn.Linear(29*29*8, 29*29*8)
        self.fc3 = nn.Linear(29*29*8, 1)
    def forward(self, x):
        x = F.relu((self.conv1(x))) #29
        x = F.relu((self.conv2(x))) #28
        x = F.relu((self.conv3(x)))
        x = F.relu((self.conv4(x)))
        x = F.relu((self.conv5(x)))
        x = F.relu((self.conv6(x))).view(-1, 1,29*29*16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
print('模块导入成功')

path = r'dataset\UTCI_s_r.csv'
file_path_forM1=r'dataset\UTCI_forM1.csv'
file_path_forM2=r'dataset\UTCI_forM2.csv'
file_path_forM3=r'dataset\UTCI_forM3.csv'


radom_seed = 120 #单词写错了，后面不修改

###group A####
def data_preGROUPA_mlp(path=file_path_forM1):
    df = pd.read_table(path,sep=',')
    df = df.dropna()
    ar = np.array(df)
    ar = ar[~np.isnan(ar).any(axis=1)]
    build_s = ar[:, :900]
    build_v = ar[:, 900:].reshape(-1,30,30).transpose(0,2,1).reshape(-1,900)
    #build_location = np.where(build_s > 0,0,1)  #找到建筑所在位置
    #build_v = build_v * build_location  #建筑所在位置变为0
    X = build_s
    Y = build_v
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=radom_seed)
    # 将数据集转化为张量
    train_xt = torch.from_numpy(X_train.astype(np.float32)).view(-1, 1, 900)
    train_yt = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1, 900)
    test_xt = torch.from_numpy(X_test.astype(np.float32)).view(-1, 1, 900)
    test_yt = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1, 900)
    return train_xt, train_yt, test_xt, test_yt
# train_xt, train_yt, test_xt, test_yt = data_preGROUPA_mlp()

def check(train_xt,train_yt):
    X = train_xt.data.cpu().numpy()[:1,:,:].reshape(30,30)
    plt.figure(figsize=(12, 9))
    plt.imshow(X,cmap='Oranges')
    plt.savefig('check\check1')
    Y = train_yt.data.cpu().numpy()[:1,:,:].reshape(30,30)
    plt.figure(figsize=(12, 9))
    plt.imshow(Y,cmap='Oranges')
    plt.savefig('check\check2')
# check()

def data_preGROUPA_cnn(path=file_path_forM1):
    df = pd.read_table(path,sep=',')
    df = df.dropna()
    ar = np.array(df)
    ar = ar[~np.isnan(ar).any(axis=1)]
    build_s = ar[:, :900]
    build_v = ar[:, 900:].reshape(-1,30,30).transpose(0,2,1).reshape(-1,900)
    #build_location = np.where(build_s > 0,0,1)  #找到建筑所在位置
    #build_v = build_v * build_location  #建筑所在位置变为0
    X = build_s
    Y = build_v
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=radom_seed)
    # 将数据集转化为张量
    train_xt = torch.from_numpy(X_train.astype(np.float32)).view(-1, 1, 30,30)
    train_yt = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1, 900)
    test_xt = torch.from_numpy(X_test.astype(np.float32)).view(-1, 1, 30,30)
    test_yt = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1, 900)
    return train_xt, train_yt, test_xt, test_yt
# train_xt, train_yt, test_xt, test_yt = data_preGROUPA_cnn()


###group B####
def data_preGROUPB_mlp(path=file_path_forM2):
    df = pd.read_table(path,sep=',')
    df = df.dropna()
    ar = np.array(df)
    ar = ar[~np.isnan(ar).any(axis=1)]
    build_s = ar[:, :81]
    build_v = ar[:, 81:]
    #build_location = np.where(build_s > 0,0,1)  #找到建筑所在位置
    #build_v = build_v * build_location  #建筑所在位置变为0
    X = build_s
    Y = build_v
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=radom_seed)
    # 将数据集转化为张量
    train_xt = torch.from_numpy(X_train.astype(np.float32)).view(-1, 1, 81)
    train_yt = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1, 1)
    test_xt = torch.from_numpy(X_test.astype(np.float32)).view(-1, 1, 81)
    test_yt = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1, 1)
    return train_xt, train_yt, test_xt, test_yt

def data_preGROUPB_cnn(path=file_path_forM2):
    df = pd.read_table(path,sep=',')
    df = df.dropna()
    ar = np.array(df)
    ar = ar[~np.isnan(ar).any(axis=1)]
    build_s = ar[:, :81]
    build_v = ar[:, 81:]
    #build_location = np.where(build_s > 0,0,1)  #找到建筑所在位置
    #build_v = build_v * build_location  #建筑所在位置变为0
    X = build_s
    Y = build_v
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=radom_seed)
    # 将数据集转化为张量
    train_xt = torch.from_numpy(X_train.astype(np.float32)).view(-1, 1, 9,9)
    train_yt = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1, 1)
    test_xt = torch.from_numpy(X_test.astype(np.float32)).view(-1, 1, 9,9)
    test_yt = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1, 1)
    return train_xt, train_yt, test_xt, test_yt
# train_xt, train_yt, test_xt, test_yt = data_preGROUPB_cnn()

###group C####

def data_preGROUPC_mlp(path=file_path_forM3):
    df = pd.read_table(path,sep=',')
    df = df.dropna()
    ar = np.array(df)
    ar = ar[~np.isnan(ar).any(axis=1)]
    build_s = ar[:150000, :29*29]####[270216, 1, 29, 29]
    build_v = ar[:150000, 29*29:]
    #build_location = np.where(build_s > 0,0,1)  #找到建筑所在位置
    #build_v = build_v * build_location  #建筑所在位置变为0
    X = build_s
    Y = build_v
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=radom_seed)
    # 将数据集转化为张量
    train_xt = torch.from_numpy(X_train.astype(np.float32)).view(-1, 1, 29*29)
    train_yt = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1, 1)
    test_xt = torch.from_numpy(X_test.astype(np.float32)).view(-1, 1, 29*29)
    test_yt = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1, 1)
    return train_xt, train_yt, test_xt, test_yt

def data_preGROUPC_cnn(path=file_path_forM3):
    df = pd.read_table(path,sep=',')
    df = df.dropna()
    ar = np.array(df)
    ar = ar[~np.isnan(ar).any(axis=1)]
    build_s = ar[:150000, :29*29]
    build_v = ar[:150000, 29*29:]
    #build_location = np.where(build_s > 0,0,1)  #找到建筑所在位置
    #build_v = build_v * build_location  #建筑所在位置变为0
    X = build_s
    Y = build_v
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=radom_seed)
    # 将数据集转化为张量
    train_xt = torch.from_numpy(X_train.astype(np.float32)).view(-1, 1, 29,29)
    train_yt = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1, 1)
    test_xt = torch.from_numpy(X_test.astype(np.float32)).view(-1, 1, 29,29)
    test_yt = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1, 1)
    return train_xt, train_yt, test_xt, test_yt
train_xt, train_yt, test_xt, test_yt = data_preGROUPC_cnn()
print(train_xt.size())


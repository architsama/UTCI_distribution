
#本文件用于临时测试
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
print('模块导入成功')
###该文件为数据准备的正式文件

def remove_duplicate(ndar):
    y = pd.DataFrame(ndar)
    z = y.drop_duplicates()
    ar = np.array(z)
    return ar
##第一步，将原始文件中UTCI统一为3个小数点

path = r'dataset\UTCI_s_r.csv'
file_path_forM1=r'dataset\UTCI_forM1.csv'
file_path_forM2_t=r'dataset\UTCI_forM2_t.csv'
file_path_forM2=r'dataset\UTCI_forM2.csv'
file_path_forM3_t=r'dataset\UTCI_forM3_t.csv'
file_path_forM3=r'dataset\UTCI_forM3.csv'


def dataforM1(path,file_path_forM1):
    df = pd.read_table(path, sep=',')
    df = df.dropna()
    ar = np.array(df)
    data_to_save = ar[~np.isnan(ar).any(axis=1)]
    file_path = file_path_forM1
    with open(file_path, 'ab') as f:  # 'ab'是追加数据
        np.savetxt(f, data_to_save, fmt="%3f", delimiter=",")
    print("file_path_forM1数据转换完成")
# dataforM1(path,file_path_forM1)
##第二步，利用file_path_forM1 构造file_path_forM2的数据
# #测试绘图，保证图片是正的
def check_or(build_s_1,build_v_1):
    build_s_1_X=build_s_1[0:1,:,:].reshape(30,30)
    build_v_1_Y=build_v_1[0:1,:,:].reshape(30,30)
    plt.figure(figsize=(12, 9))
    plt.title('build_s_1_X')
    plt.imshow(build_s_1_X,cmap='afmhot')
    plt.colorbar()
    plt.savefig('build_s_1_X')

    plt.figure(figsize=(12, 9))
    plt.title('build_v_1_Y')
    plt.imshow(build_v_1_Y,cmap='afmhot')
    plt.colorbar()
    plt.savefig('build_v_1_Y')
    print('查看图片是否是正的')



#开始重组案例
cases= 100  #  设定多少个案例4800条，最多设定960条

#为Y值padding一下
def dataforM2_t(path = file_path_forM1,pathdata = file_path_forM2_t,cases = 100):
    df = pd.read_table(path, sep=',')
    df = df.dropna()
    ar = np.array(df)
    ar = ar[~np.isnan(ar).any(axis=1)]
    build_s_1 = ar[:, :900].reshape(-1, 30, 30)
    build_v_1 = ar[:, 900:].reshape(-1, 30, 30).transpose(0, 2, 1).reshape(-1, 30, 30)

    build_s_1 = np.pad(build_s_1, (20))[20:-20, :, :]  # 三维扩充20，在去掉前20和后20的全0数据

    for p in range(cases):
        w = np.zeros([1, 82])  #设定第一行
        for n in range(5):
            case_num = n + 1 + 5 * p    #5个一组写入
            for m in range(30):
                loc_x = m + 1
                for i in range(30):
                    loc_y = i + 1
                    case_value = build_v_1[case_num:case_num + 1, loc_x - 1:loc_x, loc_y - 1:loc_y]
                    case_shape = build_s_1[case_num:case_num + 1, loc_x + 20 - 4:loc_x + 20 + 5,
                                 loc_y + 20 - 4:loc_y + 20 + 5]
                    output = case_value.reshape(1)
                    inputdata = case_shape.reshape(81)
                    data = np.concatenate((inputdata, output))
                    w = np.append(w, [data], axis=0)
        data_to_save = w
        data_to_save = remove_duplicate(data_to_save)
        print('%d is ok'% case_num)
        with open(pathdata, 'ab') as f: #'ab'是追加数据
            np.savetxt(f, data_to_save, fmt="%3f",delimiter=",")
    print("数据转换完成")
    print('完')

# dataforM2_t(path = file_path_forM1,pathdata = file_path_forM2_t,cases = 100)

def dataforM2():
    df = pd.read_table(file_path_forM2_t, sep=',')
    df_1 = df.drop_duplicates()
    with open(file_path_forM2, 'ab') as f:  # 'ab'是追加数据
        np.savetxt(f, df_1, fmt="%3f", delimiter=",")
# dataforM2()


##第三步，利用利用file_path_forM1 构造file_path_forM3的数据


def convert_to_Data3(path=file_path_forM1,file_path=file_path_forM3_t,case_nums=100):
    df = pd.read_table(path, sep=',')
    df = df.dropna()
    ar = np.array(df)
    ar = ar[~np.isnan(ar).any(axis=1)]
    build_s = ar[:, :900]
    build_v = ar[:, 900:].reshape(-1,30,30).transpose(0,2,1).reshape(-1,900)
    # build_location = np.where(build_s > 0, 0, 1)  # 找到建筑所在位置
    # build_v = build_v * build_location  # 建筑所在位置变为0
    X_re = build_s.reshape(-1, 30, 30)
    X_re = np.pad(X_re,(20))[20:-20,:,:]  #三维扩充20，在去掉前20和后20的全0数据

    Y_re = build_v.reshape(-1, 30, 30)

    for p in range(case_nums):

        w = np.zeros([1, 842])
        for n in range(5):
            case_num = n+1+5*p
            for m in range(30):
                loc_x = m + 1
                for i in range(30):
                    loc_y = i + 1
                    case_value = Y_re[case_num:case_num+1,loc_x-1:loc_x,loc_y-1:loc_y]
                    case_shape = X_re[case_num:case_num+1,loc_x+20-14:loc_x+20+15,loc_y+20-14:loc_y+20+15]
                    output=case_value.reshape(1)
                    inputdata=case_shape.reshape(841)
                    data = np.concatenate((inputdata,output))
                    w=np.append(w,[data],axis=0)
        data_to_save = w
                    #此处需要删除重复的数据+
        data_to_save = remove_duplicate(data_to_save)
        print('%d is ok'% case_num)
        with open(file_path, 'ab') as f: #'ab'是追加数据
            np.savetxt(f, data_to_save, fmt="%3f",delimiter=",")
    print("数据转换完成")

#将数据转换为30*30的单条
# convert_to_Data3(path=file_path_forM1,file_path=file_path_forM3_t,case_nums=100)  #有1500个案例
#
df = pd.read_table(file_path_forM3_t, sep=',')
df_1 = df.drop_duplicates()
with open(file_path_forM3, 'ab') as f:  # 'ab'是追加数据
    np.savetxt(f, df_1, fmt="%3f", delimiter=",")


print('完')




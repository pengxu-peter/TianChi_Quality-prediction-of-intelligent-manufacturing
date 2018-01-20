# -*- coding: utf-8 -*-
"""
Created on：2018.1.13
linear regression V3
使用testA的结果数据
@author: pengxu
"""
import os, sys, time, logging
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import linear_regression as lr

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

INDEX_ID = 'ID'

def cost(sub_y, y):
    m = len(y)
    inner = sub_y - y  # R(m*1)
    square_sum = inner.T @ inner
    cost = square_sum / m
    return cost    

def build_model(x_train,y_train):
    reg_model = LinearRegression()
    reg_model.fit(x_train,y_train)
    return reg_model
    
def dataframe_to_excel(data, savepath, name):
    #将数据按照excel保存,需要将INDEX_ID转化保存
    save_path_name = os.path.join(savepath, name+'.xlsx')
    writer = pd.ExcelWriter(save_path_name)
    data.to_excel(writer,'Sheet1')
    writer.save()       

    

def two_data_split_label(X,test_size=0.3):
    #X:含label的数据集：分割成训练集和测试集
    #test_size:测试集占整个数据集的比例
    X_num=X.shape[0]
    train_index=[x for x in range(X_num)]
    test_index=[]
    test_num=int(X_num*test_size)
    for i in range(test_num):
        randomIndex=int(np.random.uniform(0,len(train_index)))
        test_index.append(train_index[randomIndex])
        del train_index[randomIndex]
    #train,test的index是抽取的数据集X的序号
    train=X[train_index]
    test=X[test_index]
    return train,test    
    
def two_data_split_unlabel(X, Y, test_size=0.3):
    #X:含label的数据集：分割成训练集和测试集
    #test_size:测试集占整个数据集的比例
    X_num=X.shape[0]
    train_index=[x for x in range(X_num)]
    test_index=[]
    test_num=int(X_num*test_size)
    for i in range(test_num):
        randomIndex=int(np.random.uniform(0,len(train_index)))
        test_index.append(train_index[randomIndex])
        del train_index[randomIndex]
    #train,test的index是抽取的数据集X的序号
    train_X = X[train_index]
    train_Y = Y[train_index]
    test_X = X[test_index]
    test_Y = Y[test_index]
    return train_X, train_Y[:,0], test_X, test_Y[:,0]
    
def two_data_split_unlabel_fix(X, Y, test_size=0.3):
    #X:含label的数据集：分割成训练集和测试集
    #test_size:测试集占整个数据集的比例
    X_num=X.shape[0]
    fix_len = int((1-test_size)*X_num)
    #train,test的index是抽取的数据集X的序号
    train_X = X[:fix_len,:]
    train_Y = Y[:fix_len]
    test_X = X[fix_len:,:]
    test_Y = Y[fix_len:]
    return train_X, train_Y[:,0], test_X, test_Y[:,0]    
    
def three_data_split_unlabel(X, Y, cv_size=0.2, test_size=0.2):
    #X:含label的数据集：分割成训练集和测试集
    #test_size:测试集占整个数据集的比例
    X_num=X.shape[0]
    train_index=[x for x in range(X_num)]
    test_index = []
    cv_index = []
    test_num = int(X_num*test_size)
    cv_num = int(X_num*cv_size)
    for i in range(test_num):
        randomIndex=int(np.random.uniform(0,len(train_index)))
        test_index.append(train_index[randomIndex])
        del(train_index[randomIndex])
    for i in range(cv_num):
        randomIndex=int(np.random.uniform(0,len(train_index)))
        cv_index.append(train_index[randomIndex])
        del(train_index[randomIndex])
    #train,test的index是抽取的数据集X的序号
    train_X = X[train_index] 
    train_Y = Y[train_index]
    cv_X = X[cv_index]
    cv_Y = Y[cv_index]
    test_X = X[test_index]
    test_Y = Y[test_index]
    return train_X, train_Y[:,0], cv_X, cv_Y[:,0], test_X, test_Y[:,0]

    
class DataLinReg:
    def __init__(self):
        #日志保存
        log_savepath = r'.'
        log_filename = "record_data_preprocess.log"
        logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(log_savepath, log_filename),
                        filemode='a')
        #################################################################################################
        #定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        #formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s') #levelname共占据8个字段
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        #################################################################################################
            
        self.data_directory = r'temp_data_v6_v6\(0.02_20_0.2)\mean_diff_0.4_pca_99'
        self.train_x_name = '训练'
        self.train_y_name = 'Y'
        self.test_a_name = '测试A'
        self.test_a_y_name = '[new] 测试A-答案_20180114'
        self.test_b_name = '测试B'
        
        logging.info('read files from %s, the data_preprocess started !!!' % (self.data_directory))
        
    def data_read(self):
        #将所有数据读出，四组数据，前三组的列数完全相同
        self.raw_data = {} #保存原始xlsx全部信息
        for data_name in [self.train_x_name, self.train_y_name, self.test_a_name, self.test_b_name]:
            data_path = os.path.join(self.data_directory, data_name + '.xlsx')
            df = pd.read_excel(data_path)
            df.index = df[INDEX_ID].tolist() #将ID取消了
            del(df[INDEX_ID])
            df.index.name = INDEX_ID #将index名称获取
            self.raw_data[data_name] = df
        self.test_a_index = self.raw_data[self.test_a_name].index
        self.test_b_index = self.raw_data[self.test_b_name].index
        logging.info('data_read successfully !!!')
        #将dataframe转为numpy
        self.new_data = {} #保存全部预处理之后的数据
        for data_name in [self.train_x_name, self.train_y_name, self.test_a_name, self.test_b_name]:
            self.new_data[data_name] = np.array(self.raw_data[data_name])#保存所有数据进来
        #将train的数据进行分割，用于训练
        test_a_y_path = os.path.join('data', self.test_a_y_name + '.csv')
        self.test_a_Y = np.array(pd.read_csv(test_a_y_path,header=None))[:,1]
        self.data_X = np.vstack((self.new_data['训练'], self.new_data['测试A']))
        self.data_Y = np.vstack((self.new_data['Y'], np.array([np.float64(self.test_a_Y)]).T))
        
        (self.train_X, self.train_Y, self.cv_X, self.cv_Y) = two_data_split_unlabel(self.data_X, self.data_Y, test_size=0.166) #0.166刚好100都是testA
        logging.info('labeled data seperated successfully !!!')
        
                
    def results_get(self):
        #根据上述结果，计算A和B的数据结果
        model = build_model(self.train_X,self.train_Y)
        sub_y = model.predict(self.cv_X)
        self.cost_y = cost(sub_y, self.cv_Y)
        logging.info('the cost is %f !!!'%(self.cost_y))
        
        subA = model.predict(self.new_data['测试B'])
        # read submit data
        
        sub_df = pd.read_csv(os.path.join('data','测试B-答案模板.csv'),header=None)
        sub_df['Y'] = subA
        sub_df.to_csv(os.path.join(self.data_directory,'github.csv'),header=None,index=False)
        
        logging.info('data saved successfully !!!')
        
def main():
    time_start = time.time()
    
    dlr = DataLinReg()
    dlr.data_read() #读“训练.xlsx”数据时间太长
    dlr.results_get() #根据给出的最优系数，得到所需的预测A和B

    time_stop = time.time()        
    logging.info('The data_preprocess is finished, all process time is %d s !!!' %(time_stop-time_start))
                  
if __name__ == '__main__':
    main()




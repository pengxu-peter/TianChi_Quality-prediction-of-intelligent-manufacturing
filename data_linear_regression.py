# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:43:58 2017
linear regression

@author: pengxu
"""



import os, sys, time, logging
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import linear_regression as lr

INDEX_ID = 'ID'


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
            
        self.data_directory = r'temp_data\median_pca_99'
        self.train_x_name = '训练'
        self.train_y_name = 'Y'
        self.test_a_name = '测试A'
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
        (self.train_X, self.train_Y, self.cv_X, self.cv_Y, self.test_X, self.test_Y) = three_data_split_unlabel(self.new_data['训练'], self.new_data['Y'], cv_size=0.19, test_size=0.01)
        logging.info('labeled data seperated successfully !!!')
        
    def linear_analysis(self):
        #利用线性回归函数进行相关计算
        
        l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
        training_cost, cv_cost, test_cost = [], [], []
        for l in l_candidate:
            res = lr.linear_regression_np(self.train_X, self.train_Y, l)
            tc = lr.cost(res.x, self.train_X, self.train_Y)
            cv = lr.cost(res.x, self.cv_X, self.cv_Y)
            te = lr.cost(res.x, self.test_X, self.test_Y)
            training_cost.append(tc)
            cv_cost.append(cv)
            test_cost.append(te)
        sel_index = np.argmin(cv_cost)
        set_l = l_candidate[np.argmin(cv_cost)]
        #绘图看效果
        plt.figure(figsize = (12.7, 7.8))    
        plt.plot(l_candidate, training_cost, label='training')
        plt.plot(l_candidate, cv_cost, label='cross validation')
        plt.plot(l_candidate, test_cost, label='test')
        plt.xscale('log')
        plt.legend(loc=2)
        plt.yscale('log')
        plt.xlabel('lambda')
        plt.ylabel('cost')
        plt.suptitle('lambda: %f, train: %f, cv: %f, test: %f'%(set_l, training_cost[sel_index], cv_cost[sel_index], test_cost[sel_index]), fontsize = 12, fontweight = 'bold')
        title_cost_l = 'lambda'
        plt.savefig(os.path.join(self.data_directory, title_cost_l))
        plt.close('all')
        
        logging.info('lambda figure saved successfully !!!')
        
        #m个样本进行训练查看，判断bias和variance
        training_cost, cv_cost, test_cost = [], [], []
        m = self.train_X.shape[0]
        for i in range(1, m+1): #查看随着训练样本m的增大，cost的变化情况
            res = lr.linear_regression_np(self.train_X[:i, :], self.train_Y[:i], l=set_l)
            tc = lr.regularized_cost(res.x, self.train_X[:i, :], self.train_Y[:i], l=set_l)
            cv = lr.regularized_cost(res.x, self.cv_X, self.cv_Y, l=set_l)
            te = lr.regularized_cost(res.x, self.test_X, self.test_Y, l=set_l)
            training_cost.append(tc)
            cv_cost.append(cv)
            test_cost.append(te)
        #绘图看效果
        plt.figure(figsize = (12.7, 7.8))    
        plt.plot(np.arange(1, m+1), training_cost, label='training cost')
        plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
        plt.plot(np.arange(1, m+1), test_cost, label='test cost')
        plt.legend(loc=2)
        plt.yscale('log')
        plt.xlabel('sample_num')
        plt.ylabel('cost')
        plt.suptitle('cost vs sample_num', fontsize = 12, fontweight = 'bold')
        title_cost_l = 'sample_num'
        plt.savefig(os.path.join(self.data_directory, title_cost_l))
        plt.close('all')
        
        logging.info('sample_num figure saved successfully !!!')
        
        self.theta = lr.linear_regression_np(self.train_X, self.train_Y, set_l).x
        
    def results_get(self):
        #根据上述结果，计算A和B的数据结果
        test_a_Y = self.new_data['测试A'] @ self.theta
        test_b_Y = self.new_data['测试B'] @ self.theta
        test_a_dataframe = pd.DataFrame(test_a_Y, columns = ['Y'], index = self.test_a_index)
        test_b_dataframe = pd.DataFrame(test_b_Y, columns = ['Y'], index = self.test_b_index)
        dataframe_to_excel(test_a_dataframe, self.data_directory, '测试A_Y')
        dataframe_to_excel(test_b_dataframe, self.data_directory, '测试B_Y')
        logging.info('data saved successfully !!!')
        
def main():
    time_start = time.time()
    
    dlr = DataLinReg()
    dlr.data_read() #读“训练.xlsx”数据时间太长
    dlr.linear_analysis() #获得最优系数，以及train, cv, test的误差（图上直接绘图）
    dlr.results_get() #根据给出的最优系数，得到所需的预测A和B

    time_stop = time.time()        
    logging.info('The data_preprocess is finished, all process time is %d s !!!' %(time_stop-time_start))
                  
if __name__ == '__main__':
    main()




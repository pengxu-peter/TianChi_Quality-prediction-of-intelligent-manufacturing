# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:22:54 2017

数据预处理，导出为excel
均值、中值、指数均值？指数中值？
data_frame太慢，后续需要的话，改为numpy运算

@author: pengxu
"""

import os, sys, time, logging
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

INDEX_ID = 'ID'

def non_digital_find(data):
    #查找DataFrame数据中非数字的值
    wrong_data = []
    wrong_item = []
    data_index = data.index
    for item in data.columns:
        v = data[item][data_index[55]] #查看第56行数据情况
        if not isinstance(v, (np.int64, int, float)): #np.int64和int不同
            wrong_data.append(v)
            wrong_item.append(item)
            #print(item, v)
    return wrong_item, wrong_data

def non_digital_data(non_digital_item, data):
    #将data中non_digital_item提取出来
    return data[non_digital_item]
    
    
def nan_data_find(data):
    #查找DataFrame数据中所有nan的值
    nan_item = []
    nan_data = []
    data_index = data.index
    for item in data.columns:
        v = data[item]
        v_nan_check = pd.isnull(v)
        for index_i in data_index:
            if v_nan_check[index_i]:
                nan_data.append(v[index_i])
                nan_item.append(index_i)
    return nan_item, nan_data
            

def dataframe_to_excel(data, savepath, name):
    #将数据按照excel保存,需要将INDEX_ID转化保存
    save_path_name = os.path.join(savepath, name+'.xlsx')
    writer = pd.ExcelWriter(save_path_name)
    data.to_excel(writer,'Sheet1')
    writer.save()       

class DataPreprocess:
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
            
        self.data_directory = r'data'
        self.data_list = ['训练', '测试A', '测试B']
        
        logging.info('read files from %s, the data_preprocess started !!!' % (self.data_directory))
        
    def data_read(self):
        #将所有数据读出
        self.raw_data = {} #保存原始xlsx全部信息
        for data_name in self.data_list:
            data_path = os.path.join(self.data_directory, data_name + '.xlsx')
            df = pd.read_excel(data_path)
            df.index = df[INDEX_ID].tolist() #将ID取消了
            del(df[INDEX_ID])
            df.index.name = INDEX_ID #将index名称获取
            self.raw_data[data_name] = df
        self.new_data = {} #保存全部预处理之后的数据
        for data_name in self.data_list:
            self.new_data[data_name] = self.raw_data[data_name].copy(deep=True) #两个物理位置，不能相同，要不后面赋值全部错误
        logging.info('data_read successfully !!!')
            
    def ascll_transfer(self):      
        #字符转换ASCLL
        self.data_index_0 = self.new_data[self.data_list[0]].index
        self.data_index_1 = self.new_data[self.data_list[1]].index
        self.data_index_2 = self.new_data[self.data_list[2]].index
        self.data_paramter = self.new_data[self.data_list[0]].columns[:-1] #最后一个数据是预测值
        self.data_y = self.new_data[self.data_list[0]].columns[-1]
        #通过观察‘TOOL_ID (#3)’和‘TOOL (#1)’保留第一个字母，然后转ASCLL码即可
        #通过观察，所有字符类的数据都是非nan的
        non_digital_item, non_digital_data = non_digital_find(self.new_data[self.data_list[0]]) #用训练集的数据来找非数字项目列表
        for item in non_digital_item:
            for index_0 in self.data_index_0:
                self.new_data[self.data_list[0]][item][index_0] = ord(self.new_data[self.data_list[0]][item][index_0][0]) #第一个字母就可以了
            for index_1 in self.data_index_1:
                self.new_data[self.data_list[1]][item][index_1] = ord(self.new_data[self.data_list[1]][item][index_1][0])
            for index_2 in self.data_index_2:
                self.new_data[self.data_list[2]][item][index_2] = ord(self.new_data[self.data_list[2]][item][index_2][0])
        logging.info('ascll_transfer successfully !!!')
        
    def nan_replace(self, mode = 'mean'):
        #将所有nan数据进行填充,第一步：合并所有数据，计算均值或中位数，第二步：对所有原来nan数据进行覆盖
        #有些全部是nan的
        self.data_all = self.new_data[self.data_list[0]]
        del(self.data_all[self.data_y]) #将输出列数据去除
        self.data_all =pd.concat([self.data_all,self.new_data[self.data_list[1]],self.new_data[self.data_list[2]]])#将所有数据合并到data_all中,此时data_all和new_data地址没有共享，数据是分开的
        #按照模式计算填充值
        self.mode = mode
        if mode == 'mean':
            data_replace = self.data_all.mean()
            
        #将所有数据进行替换
        for item in data_replace.index: #避免最后一列Y没有对应的data_replace
            #print(item)
            v = self.data_all[item]
            v_nan_check = pd.isnull(v)
            if sum(v_nan_check): #至少有一个nan，才需要进行替换操作
                v[v_nan_check] = data_replace[item]
        logging.info('nan_replace successfully !!!')
            
    def normalization(self):
        #将上述数据进行归一化，bug：相差为0的，都变成了nan
        self.data_all_mean = self.data_all.mean()
        self.data_all_std = self.data_all.std()
        for item in self.data_paramter:
            self.data_all[item] = (self.data_all[item] - self.data_all_mean[item])/self.data_all_std[item]
        logging.info('normalization successfully !!!')
        
    def data_all_to_savedata(self):
        #将处理后的data_all写入save_data
        self.save_data = {} #保存原始xlsx全部信息
        for data_name in self.data_list:
            data = self.new_data[data_name]
            data_index = data.index
            self.save_data[data_name] = self.data_all.loc[data_index] #对于'训练'数据，需要对Y进行补充
        self.save_data_Y = self.raw_data[self.data_list[0]]['Y'] #单独保存最后一列数据
        logging.info('data_all_to_newdata successfully !!!')
                
    def data_save(self):
        #有些过程太漫长，保存中间步骤，避免后续太麻烦
        base_savepath = 'temp_data'
        savepath = os.path.join(base_savepath, self.mode)
        if not os.path.exists(savepath):
            os.makedirs(savepath) #如果目录不存在，makedirs函数会自动创建父目录
        for k,v in self.save_data.items():
            dataframe_to_excel(v, savepath, k)
        dataframe_to_excel(self.save_data_Y, savepath, 'Y')
        logging.info('data_to_save successfully !!!')
            
        
def main():
    time_start = time.time()
    
    dp = DataPreprocess()
    dp.data_read() #读“训练.xlsx”数据时间太长
    dp.ascll_transfer() #时间也太长。。。
    dp.nan_replace(mode='mean') #时间也太长了。。。
    dp.normalization()
    dp.data_all_to_savedata()
    dp.data_save() #时间太长。。。
    
    time_stop = time.time()        
    logging.info('The data_preprocess is finished, all process time is %d s !!!' %(time_stop-time_start))
                  
if __name__ == '__main__':
    main()





















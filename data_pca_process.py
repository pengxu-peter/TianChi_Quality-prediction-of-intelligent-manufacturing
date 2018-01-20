# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:26:01 2017
pca处理，将data_preprocess处理后的数据再处理，降低维度，导出为excel文件
@author: pengxu
"""


import os, sys, time, logging
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pca
import shutil

INDEX_ID = 'ID'


def dataframe_to_excel(data, savepath, name):
    #将数据按照excel保存,需要将INDEX_ID转化保存
    save_path_name = os.path.join(savepath, name+'.xlsx')
    writer = pd.ExcelWriter(save_path_name)
    data.to_excel(writer,'Sheet1')
    writer.save()       

class DataPCAProcess:
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
            
        self.data_directory = r'temp_data\mean'
        self.data_list = ['训练', '测试A', '测试B']
        
        logging.info('read files from %s, the data_preprocess started !!!' % (self.data_directory))
        
    def data_read(self):
        #将所有数据读出，四组数据，前三组的列数完全相同
        self.raw_data = {} #保存原始xlsx全部信息
        for data_name in self.data_list:
            data_path = os.path.join(self.data_directory, data_name + '.xlsx')
            df = pd.read_excel(data_path)
            df.index = df[INDEX_ID].tolist() #将ID取消了
            del(df[INDEX_ID])
            df.index.name = INDEX_ID #将index名称获取
            self.raw_data[data_name] = df
        self.data_index_0 = self.raw_data[self.data_list[0]].index
        self.data_index_1 = self.raw_data[self.data_list[1]].index
        self.data_index_2 = self.raw_data[self.data_list[2]].index
        self.data_parameter = self.raw_data[self.data_list[0]].columns #已经除去了最后一行
        logging.info('data_read successfully !!!')
        #将dataframe转为numpy
        self.new_data = {} #保存全部预处理之后的数据
        for data_name in self.data_list:
            self.new_data[data_name] = np.array(self.raw_data[data_name])#保存所有数据进来
        #将所有数据合并，便于统一处理
        self.row0, self.col = self.new_data[self.data_list[0]].shape
        self.row1, _ = self.new_data[self.data_list[1]].shape
        self.row2, _ = self.new_data[self.data_list[2]].shape
        self.data_all = np.empty(shape=(self.row0+self.row1+self.row2, self.col), dtype=np.float64)
        self.data_all[0:self.row0,:] = self.new_data[self.data_list[0]]
        self.data_all[self.row0:(self.row0+self.row1),:] = self.new_data[self.data_list[1]]
        self.data_all[(self.row0+self.row1):(self.row0+self.row1+self.row2),:] = self.new_data[self.data_list[2]]
        logging.info('data_combined successfully !!!')
        
    def nan_replace(self):
        #将所有为nan的数据置位0,应该直接删除
        data_simple = []
        self.data_s_parameter = []
        data_sum = np.sum(self.data_all, axis=0)
        data_std = np.std(self.data_all, axis=0)
        for i in range(len(data_sum)): #删除方式，逆序遍历
            if not (np.isnan(data_sum[i]) or data_std[i]==0):
                data_simple.append(self.data_all[:,i])
                self.data_s_parameter.append(self.data_parameter[i])
        self.data_s = np.array(data_simple).T
        logging.info('nan and std=0 data deleter successfully !!!')
                
    def pac_process(self, variance_threshold=90):
        #进行pca处理
        sigma = pca.covariance_matrix(self.data_s)
        self.U, self.S, self.V = np.linalg.svd(sigma)
        
        
        self.keep_variance = np.zeros(self.S.shape, dtype=np.float64)
        for i in range(len(self.keep_variance)):
            self.keep_variance[i] = np.sum(self.S[0:i])/np.sum(self.S)
        self.variance_threshold = variance_threshold
        for i in range(len(self.keep_variance)-1):
            if (self.keep_variance[i] - variance_threshold/100) * (self.keep_variance[i+1] - variance_threshold/100) <= 0: 
                self.pca_num = i
                #break #完全单调，可以
                
        self.Z = pca.project_data(self.data_s, self.U, self.pca_num)        
        logging.info('pca process successfully, the data number is %d !!!' %(self.pca_num))
                
    def data_seperate_transfer(self):
        #将合并的数据分割开来
        self.save_array_data = {}
        self.save_array_data[self.data_list[0]] = self.Z[0:self.row0,:]
        self.save_array_data[self.data_list[1]] = self.Z[self.row0:(self.row0+self.row1),:]
        self.save_array_data[self.data_list[2]] = self.Z[(self.row0+self.row1):(self.row0+self.row1+self.row2),:]
        logging.info('data seperated successfully !!!')
        
        #将分开的数据保存为dataframe数据
        self.pca_parameter_name = ['pca_'+str(i) for i in range(self.pca_num)]
        self.save_dataframe_data = {}
        self.save_dataframe_data[self.data_list[0]] = pd.DataFrame(self.save_array_data[self.data_list[0]], index = self.data_index_0, columns = self.pca_parameter_name)
        self.save_dataframe_data[self.data_list[1]] = pd.DataFrame(self.save_array_data[self.data_list[1]], index = self.data_index_1, columns = self.pca_parameter_name)
        self.save_dataframe_data[self.data_list[2]] = pd.DataFrame(self.save_array_data[self.data_list[2]], index = self.data_index_2, columns = self.pca_parameter_name)
        #保存U的数据，方便后续分析查看
        self.save_dataframe_u = pd.DataFrame(self.U[:, :self.pca_num], columns = self.pca_parameter_name, index = self.data_s_parameter)
        logging.info('array transfer to dataframe successfully !!!')
                
    def data_save(self):
        #有些过程太漫长，保存中间步骤，避免后续太麻烦，将Y数据直接copy进来
        (par_dir, base_dir) = self.data_directory.split('\\')
        new_base_dir = base_dir + '_pca_' + str(self.variance_threshold)
        savepath = os.path.join(par_dir, new_base_dir)
        if not os.path.exists(savepath):
            os.makedirs(savepath) #如果目录不存在，makedirs函数会自动创建父目录
        for k,v in self.save_dataframe_data.items():
            dataframe_to_excel(v, savepath, k)
            
        dataframe_to_excel(self.save_dataframe_u, savepath, 'U')
        
        #将Y数据复制过来
        shutil.copy(os.path.join(self.data_directory, 'Y.xlsx'), savepath)
        
        logging.info('data_save successfully !!!')
            
        
def main():
    time_start = time.time()
    
    dpp = DataPCAProcess()
    dpp.data_read() #读“训练.xlsx”数据时间太长
    dpp.nan_replace()
    dpp.pac_process(variance_threshold=89) #90~100之间选择较合适
    dpp.data_seperate_transfer()
    dpp.data_save() #时间太长。。。
    
    time_stop = time.time()        
    logging.info('The data_preprocess is finished, all process time is %d s !!!' %(time_stop-time_start))
                  
if __name__ == '__main__':
    main()





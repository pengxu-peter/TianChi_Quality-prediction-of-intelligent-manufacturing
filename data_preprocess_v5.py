# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:22:54 2017

v4相比的改进：
1. 剔除和Y相关度很小的数据；
2. 剔除补齐数据很多的数据；

@author: pengxu
"""

import os, sys, time, logging
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

INDEX_ID = 'ID'            

def parameter_to_index(dp, name):
    data_parameter = list(dp.raw_data[dp.data_list[1]].columns)
    use_index = data_parameter.index(name)
    return use_index

def index_get(dp, check_list):
    #有些异常的，只有一点点不同
    check_list_index = []
    check_list_value = []
    for item in check_list:
        check_list_index.append(parameter_to_index(dp, item))
    for index in check_list_index:
        check_list_value.append(np.float64(dp.raw_array_data[:,index]))
    return_data = np.array(check_list_value).T
    return return_data
    
'''    
check_list = ['210X80', '210X110', '210X108', '261X688', '261X689', '261X690', '261X694', '312X314']
check_list2 = ['210X21', '210X22', '220X452', '220X453', '210X108', '360X1400']
check_list3 = ['210X214','220X153', '220X155', '220X156', '220X291', '220X292', '261X456', '261X504', '261X508', '312X93', '312X94']
return_data = index_get(dp, check_list)
s_std = np.nanstd(return_data, axis=0)
s_mean = np.nanmean(return_data, axis=0)
s_min = np.nanmin(return_data,axis=0)
s_max = np.nanmax(return_data,axis=0)
s_kk = s_std / s_mean
'''
    
def dataframe_to_excel(data, savepath, name):
    #将数据按照excel保存,需要将INDEX_ID转化保存
    save_path_name = os.path.join(savepath, name+'.xlsx')
    writer = pd.ExcelWriter(save_path_name)
    data.to_excel(writer,'Sheet1')
    writer.save()       
    
def date_to_strptime(data):
    #将所有数据转化为17位的格式%Y%m%d%H%M%S%ms，全部采用补全的方式即可
    use_len = 17
    data_len = int(np.log10(data) + 1)
    use_data = int(data) * (10**(use_len - data_len))
    ms = use_data%1000
    use_transfer_data = str(int(use_data/1000))
    timeArray = time.strptime(use_transfer_data, "%Y%m%d%H%M%S")
    #dt_new = time.strftime("%Y-%m-%d %H:%M:%S",timeArray)
    timestamp = time.mktime(timeArray)
    new_data = timestamp*1000 + ms
    return new_data

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
            logging.info('%s is read successfully !!!'%(data_name))
        self.new_data = {} #保存全部预处理之后的数据
        for data_name in self.data_list:
            self.new_data[data_name] = np.array(self.raw_data[data_name])
        #将所有X数据合并，方便后续处理
        data0 = self.new_data[self.data_list[0]][:,0:-1] #Y值不处理
        data1 = self.new_data[self.data_list[1]]
        data2 = self.new_data[self.data_list[2]]
        self.raw_array_data = np.vstack((data0, data1, data2)) #合并三个矩阵数据
        logging.info('all data being read successfully !!!')
            
    def ascll_transfer(self):      
        #字符转换ASCLL
        (row, col) = self.raw_array_data.shape
        alpha_index = []
        alpha = []
        for i in range(col):
            v = self.raw_array_data[0,i] #刚好第一行数据都是非nan的
            if not isinstance(v, (np.int64, int, float)): #np.int64和int不同
                alpha_index.append(i)
                alpha.append(v)
        for index in alpha_index:
            for j in range(row):
                self.raw_array_data[j,index] = ord(self.raw_array_data[j,index][0])
        self.new_array_data = np.float64(self.raw_array_data) #将str转化完之后，就可以直接转换
        logging.info('ascll_transfer successfully !!!')
        
    def date_data_transfer(self):
        #将所有date数据（8位：20170804，14位：20170705163823，16位：2017072119260120，排除：14位的“17000000000000”）转换为时间戳
        (row, col) = self.new_array_data.shape
        
        '''#方便后续对比分析查看
        date_value, date_col_index = [], []
        for i in range(col):
                data_value = self.new_array_data[0,i]
                if (not np.isnan(data_value)) and (data_value > 0) and (str(int(data_value))[0:3]=='201'): #1.避免出现nan, 2.所有数据大于0，3.2017年的时间(2016的时间都是有问题的)
                    data_len = int(np.log10(data_value) + 1)
                    if data_len == 14 or data_len == 16 or data_len == 8:
                        date_col_index.append(i)
                        date_value.append(data_value)
        '''
        
        for i in range(col):
            for j in range(row):
                data_value = self.new_array_data[j,i]
                if (not np.isnan(data_value)) and (data_value > 0) and (str(int(data_value))[0:3]=='201'): #1.避免出现nan, 2.所有数据大于0，3.2017年的时间(2016的时间都是有问题的)
                    data_len = int(np.log10(data_value) + 1)
                    if data_len == 14 or data_len == 16 or data_len == 8:
                        if data_value == np.float64(20166616661666.0): #唯一的异常数据
                            self.new_array_data[j,i] = np.nan
                        else:
                            self.new_array_data[j,i] = date_to_strptime(data_value)
                        
        logging.info('date_data_transfer successfully !!!')
        
    def nan_replace(self, mode = 'mean'):
        #将所有nan数据进行填充,第一步：合并所有数据，计算均值或中位数，第二步：对所有原来nan数据进行覆盖
        #将一列全部是nan转为0
        (row, col) = self.new_array_data.shape
        nan_num = np.zeros(col, dtype=np.int32)
        self.nan_num_use = np.zeros(col, dtype=np.int32)
        all_nan_index = []
        for i in range(col):
            nan_num[i] = np.isnan(self.new_array_data[:,i]).sum()
            self.nan_num_use[i] = nan_num[i] #后续判断使用
            if nan_num[i] == row:
                all_nan_index.append(i)
                self.new_array_data[:,i] = 0 #用0填充
                nan_num[i] = 0 #置换了，所以这里设置为0
        #填充非全部为nan的计算结果
        nan_index = list(np.where(nan_num > 0))[0]
        self.mode = mode
        if mode == 'mean':
            data_replace = np.nanmean(self.new_array_data, axis=0)
        elif mode == 'median':
            data_replace = np.nanmedian(self.new_array_data, axis=0)
        for col_index in nan_index:
            for i in range(row):
                if np.isnan(self.new_array_data[i,col_index]):
                    self.new_array_data[i,col_index] = data_replace[col_index]
                    
        #提取可能要删除的index            
        self.nan_ignore_index = list(list(np.where(self.nan_num_use > 100))[0])      #阈值可以选择100（200个特征）、20（1684个特征）
                    
        #检查一遍
        if np.isnan(self.new_array_data).sum() != 0:
            logging.error('not all nan data replaced !!!')

        logging.info('nan_replace successfully !!! nan_ignore number is %d' %(len(self.nan_ignore_index)))
            
        
    def correlation_get(self):
        #将训练集中，与Y相关度低的特征去除
        train_len = self.raw_data[self.data_list[0]].shape[0]
        train_data_X = self.new_array_data[0:train_len,:]
        train_data_Y = np.array(self.raw_data[self.data_list[0]]['Y'])
        #测试数据
        #train_data_X = np.array([[1,1,2,2],[2,1,3,2],[3,1,4,2]])
        #train_data_Y = np.array([1,2,3])
        mean_X = np.mean(train_data_X, axis=0)
        mean_Y = np.mean(train_data_Y)
        std_X = np.std(train_data_X, axis = 0)
        std_Y = np.std(train_data_Y)
        (row, col) = train_data_X.shape
        use_data_X = train_data_X.copy()
        use_data_Y = train_data_Y - mean_Y
        for i in range(col):
            use_data_X[:,i] = train_data_X[:,i] - mean_X[i]
        correlation_data = np.zeros(col, dtype=np.float64)
        for i in range(col):
            correlation_data[i] = np.abs(use_data_X[:,i].dot(use_data_Y)/((row-1)*std_X[i]*std_Y)) #有些std_X是0，这些都是nan了
        self.correlation_index = []
        correlation_threshold = 0.02   #当前阈值设置为0.05
        for i in range(col):
            if np.isnan(correlation_data[i]):
                self.correlation_index.append(i)
            else:
                if correlation_data[i] < correlation_threshold:
                    self.correlation_index.append(i)
            
        logging.info('correlation_get successfully !!!, correlation threshold number is %d' %(len(self.correlation_index)))
        
    def normalization(self):
        #将上述数据进行归一化
        self.std_zero=[]
        (row, col) = self.new_array_data.shape
        self.data_mean = np.mean(self.new_array_data, axis=0)
        self.data_std = np.std(self.new_array_data, axis=0)
        for i in range(col):
            if self.data_std[i] > 1e-12: #算法过程有很多地方出现了累计误差（np.mean等会有小数点后多位之后出现误差），这里严格为0太少了，需要设置一个阈值，原始数据最小值为1e-10
                self.new_array_data[:,i] = (self.new_array_data[:,i] - self.data_mean[i])/self.data_std[i]
                self.new_array_data[:,i][self.new_array_data[:,i] < 1e-12] = 0 #上面计算有时候出现累计误差，这里直接消除掉，免得出现极小值
            else:
                self.new_array_data[:,i] = 0 #将所有相同的数据置为0
                self.std_zero.append(i) #将std为0的数据发现标记出来,后续不保存
                
        #self.data_std = np.std(self.new_array_data, axis=0) #重新计算，有些检在上一次测不到，类似于5568列的“360X1400"附近的很多列，所以后续需要重新检测下
        #for i in range(col):
        #    if self.data_std[i] == 0: #有些检测不到，类似于5568列的“360X1400"附近的很多列，所以后续需要重新检测下
        #        self.std_zero.append(i) #将std为0的数据发现标记出来,后续不保存
                
        logging.info('normalization successfully !!!, std_zero number is %d' %(len(self.std_zero)))
            
    def train_test_diff(self, diff_threshold=[1, 0.5, 0.2, 0.1, 0.05]):
        #将train、testA、testB差异很大的特征选取出来排序：
        train_len = self.raw_data[self.data_list[0]].shape[0]
        testA_len = self.raw_data[self.data_list[1]].shape[0]
        testB_len = self.raw_data[self.data_list[2]].shape[0]
        train_mean = np.mean(self.new_array_data[0:train_len,:],axis=0)
        testA_mean = np.mean(self.new_array_data[train_len:(train_len+testA_len),:],axis=0)
        testB_mean = np.mean(self.new_array_data[(train_len+testA_len):(train_len+testA_len+testB_len),:],axis=0)
        train_testA_diff = np.abs(train_mean - testA_mean)
        train_testB_diff = np.abs(train_mean - testB_mean)
        #将train_testA_diff和train_testB_diff偏差都大于diff_threshold的剔除出来
        self.diff_threshold = diff_threshold
        self.diff_index = dict()
        for diff_threshold_item in self.diff_threshold:
            self.diff_index[diff_threshold_item] = []
            for i in range(len(train_testA_diff)):
                if (train_testA_diff[i] - diff_threshold_item) > 0 or (train_testB_diff[i] - diff_threshold_item) > 0 :
                    self.diff_index[diff_threshold_item].append(i)
            logging.info('train_test_diff for %f done successfully !!!, delete diff number is %d'%(diff_threshold_item, len(self.diff_index[diff_threshold_item])))
        
    def data_seperate_transfer_save(self):
        #参数提取
        self.data_index_0 = self.raw_data[self.data_list[0]].index
        self.data_index_1 = self.raw_data[self.data_list[1]].index
        self.data_index_2 = self.raw_data[self.data_list[2]].index
        self.data_parameter = self.raw_data[self.data_list[1]].columns #不能选择0，因为分开保存
        
        self.row0, _ = self.raw_data[self.data_list[0]].shape #不能选择0，因为分开保存
        self.row1, _ = self.raw_data[self.data_list[1]].shape
        self.row2, self.col_num = self.raw_data[self.data_list[2]].shape
        
        #将合并的数据分割开来
        self.save_array_data = {}
        self.save_array_data[self.data_list[0]] = self.new_array_data[0:self.row0,:]
        self.save_array_data[self.data_list[1]] = self.new_array_data[self.row0:(self.row0+self.row1),:]
        self.save_array_data[self.data_list[2]] = self.new_array_data[(self.row0+self.row1):(self.row0+self.row1+self.row2),:]
        logging.info('data seperated successfully !!!')
        
        
        for diff_threshold_item in self.diff_threshold:
            #将分开的数据保存为dataframe数据
            self.save_dataframe_data = {}
            self.save_dataframe_data[self.data_list[0]] = pd.DataFrame(self.save_array_data[self.data_list[0]], index = self.data_index_0, columns = self.data_parameter)
            self.save_dataframe_data[self.data_list[1]] = pd.DataFrame(self.save_array_data[self.data_list[1]], index = self.data_index_1, columns = self.data_parameter)
            self.save_dataframe_data[self.data_list[2]] = pd.DataFrame(self.save_array_data[self.data_list[2]], index = self.data_index_2, columns = self.data_parameter)
            #删除std为0的数据，没有保存的必要，还增加后续运算的负担
            del_index_1 = list(set(self.diff_index[diff_threshold_item]).union(set(self.std_zero))) #避免重复，其实也不会有重复
            del_index_2 = list(set(del_index_1).union(set(self.nan_ignore_index))) #避免重复，其实也不会有重复
            del_index = list(set(del_index_2).union(set(self.correlation_index))) #避免重复，其实也不会有重复
            for col in del_index:
                del(self.save_dataframe_data[self.data_list[0]][self.data_parameter[col]])
                del(self.save_dataframe_data[self.data_list[1]][self.data_parameter[col]])
                del(self.save_dataframe_data[self.data_list[2]][self.data_parameter[col]])
            #单独保存最后一列数据
            self.save_dataframe_data_Y = self.raw_data[self.data_list[0]]['Y'] 
            logging.info('array transfer to dataframe successfully !!! all remained number is %d' %(self.col_num - len(del_index)))
                    
            #有些过程太漫长，保存中间步骤，避免后续太麻烦
            base_savepath = 'temp_data'
            fold_name = self.mode + '_diff_' + str(diff_threshold_item)
            savepath = os.path.join(base_savepath, fold_name)
            if not os.path.exists(savepath):
                os.makedirs(savepath) #如果目录不存在，makedirs函数会自动创建父目录
            for k,v in self.save_dataframe_data.items():
                dataframe_to_excel(v, savepath, k)
            dataframe_to_excel(self.save_dataframe_data_Y, savepath, 'Y')
            logging.info('data_to_save for %f successfully !!!' %(diff_threshold_item))
        
def main():
    time_start = time.time()
    
    dp = DataPreprocess()
    dp.data_read() #读“训练.xlsx”数据时间太长
    dp.ascll_transfer() #时间也太长。。。
    dp.date_data_transfer()
    dp.nan_replace(mode='mean') #时间也太长了。。。"median""mean"
    dp.correlation_get()
    dp.normalization()
    dp.train_test_diff(diff_threshold=[1, 0.5, 0.2, 0.1, 0.05]) #
    dp.data_seperate_transfer_save()
    
    time_stop = time.time()        
    logging.info('The data_preprocess is finished, all process time is %d s !!!' %(time_stop-time_start))
                  
if __name__ == '__main__':
    main()





















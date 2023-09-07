from numpy import *
class KMeans():
    """
    Parameters
    ----------
    n_clusters 指定了需要聚类的个数，这个超参数需要自己调整，会影响聚类的效果
    n_init 指定计算次数，算法并不会运行一遍后就返回结果，而是运行多次后返回最好的一次结果，n_init即指明运行的次数
    max_iter 指定单次运行中最大的迭代次数，超过当前迭代次数即停止运行
    """
    def __init__(self,n_clusters=5,n_init=10,max_iter=100):

        self.n_clusters = n_clusters#需要聚类的个数
        self.max_iter = max_iter#最大迭代次数
        self.n_init = n_init#计算次数
        self.centers=None
    
    def fit(self, x):
        """
        用fit方法对数据进行聚类
        :param x: 输入数据
        :best_centers: 簇中心点坐标 数据类型: ndarray
        :best_labels: 聚类标签 数据类型: ndarray
        :return: self
        """
        K=self.n_clusters
        #获取输入数据的维度Dim和个数N
        Dim=x.shape[1]
        N=x.shape[0]

        x=array(x)

        #设置中心点序列，每行代表一个中心点，每列依次为特征数
        centers=np.zeros((K,Dim))
        old_centers=np.zeros((K,Dim))

        #随机生成K个Dim维的点
        INDEX=[]
        while len(INDEX)<K:
            index=int(random.uniform(0,N))
            if index not in INDEX:
                centers[len(INDEX),:]=x[index,:]
                INDEX.append(index)
          
        classification=np.zeros(N)

        unfinished=True

        old_to_new=np.ones(K)

        iter=0
        #while(算法未收敛)
        while iter<self.max_iter and unfinished:
            iter+=1
            #对N个点：计算每个点属于哪一类。
            #计算每个点的欧氏距离平方
            point_to_center=np.zeros((N,K),dtype=np.float64)
            for i in range(K):
                point_to_center[:,i]=np.sum((x-centers[i])**2,axis=1)
            
            #更新所属类别
            classification=np.argmin(point_to_center,axis=1)

            #更新质心
            for i in range(K):
                old_centers[i]=centers[i]
                centers[i]=np.mean(x[classification==i],axis=0)
            #判断是否迭代结束
            old_to_new=np.sum((centers-old_centers)**2,axis=1)
            if old_to_new.any()!=0:
                unfinished=True
            else:
                break


        #对于K个中心点：
            #1，找出所有属于自己这一类的所有数据点
            #2，把自己的坐标修改为这些数据点的中心点坐标

        self.cluster_centers_ =centers
        self.labels_ =classification
        return self












import os
import sklearn
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from sklearn.externals import joblib

def preprocess_data(df):
    """
    数据处理及特征工程等
    :param df: 读取原始 csv 数据，有 timestamp、cpc、cpm 共 3 列特征
    :return: 处理后的数据, 返回 pca 降维后的特征
    """
    # 请使用joblib函数加载自己训练的 scaler、pca 模型，方便在测试时系统对数据进行相同的变换
    # ====================数据预处理、构造特征等========================
    # 例如
    # df['hours'] = df['timestamp'].dt.hour
    # df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)


    # ========================  模型加载  ===========================
    # 请确认需要用到的列名，e.g.:columns = ['cpc','cpm']
    # 将 timestamp 列转化为时间类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # 尝试引入非线性关系
    df['cpc X cpm'] = df['cpm'] * df['cpc']
    df['cpc / cpm'] = df['cpc'] / df['cpm']
    # 尝试获取时间关系
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    columns = ['cpc','cpm','cpc X cpm','cpc / cpm','hours','daylight']
    data = df[columns]
    scaler = joblib.load('./results/scaler.pkl')
    pca = joblib.load('./results/pca.pkl')
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=columns)
    n_components = 3
    data = pca.fit_transform(data)
    data = pd.DataFrame(data,columns=['Dimension' + str(i+1) for i in range(n_components)])
    return data













def get_distance(data, kmeans, n_features):
    """
    计算样本点与聚类中心的距离
    :param data: preprocess_data 函数返回值，即 pca 降维后的数据
    :param kmeans: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param n_features: 计算距离需要的特征的数量
    :return:每个点距离自己簇中心的距离，Series 类型
    """
    # ====================计算样本点与聚类中心的距离========================
    distance = []
    for i in range(0,len(data)):
        point = np.array(data.iloc[i,:n_features])
        center = kmeans.cluster_centers_[kmeans.labels_[i],:n_features]
        distance.append(np.linalg.norm(point - center))
    return distance















def get_anomaly(data, kmeans, ratio):
    """
    检验出样本中的异常点，并标记为 True 和 False，True 表示是异常点

    :param data: preprocess_data 函数返回值，即 pca 降维后的数据，DataFrame 类型
    :param kmean: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param ratio: 异常数据占全部数据的百分比,在 0 - 1 之间，float 类型
    :return: data 添加 is_anomaly 列，该列数据是根据阈值距离大小判断每个点是否是异常值，元素值为 False 和 True
    """
    # ====================检验出样本中的异常点========================
    from copy import deepcopy

    num_anomaly = int(len(data) * ratio)
    new_data = deepcopy(data)
    new_data['distance'] = get_distance(new_data,kmeans,n_features=len(new_data.columns))
    threshould = new_data['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
    # 根据阈值距离大小判断每个点是否是异常值
    new_data['is_anomaly'] = new_data['distance'].apply(lambda x: x > threshould)
    return new_data
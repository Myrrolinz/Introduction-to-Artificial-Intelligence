import os
import sklearn
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as LR

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
    
    #############################################################
    # try1
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     # 尝试引入非线性关系
#     df['cpc X cpm'] = df['cpm'] * df['cpc']
#     df['cpc / cpm'] = df['cpc'] / df['cpm']
#     # 尝试获取时间关系
#     df['hours'] = df['timestamp'].dt.hour
#     df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

#     #在进行特征变换之前先对各个特征进行标准化
#     columns = ['cpc', 'cpm', 'cpc X cpm', 'cpc / cpm']
#     data = df[columns]
#     scaler = joblib.load('./results/scaler.pkl')
#     data = scaler.transform(data)
#     data = pd.DataFrame(data, columns=columns)

#     #通过 n_components 指定需要降低到的维度
#     pca = joblib.load('./results/pca.pkl')
#     data = pca.transform(data)
#     data = pd.DataFrame(data,columns=['Dimension' + str(i+1) for i in range(data.shape[1])])

    #############################################################
    # try2
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 加入时间维度
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    df['weekday'] = df['timestamp'].dt.weekday
    df['weekend'] = ((df['weekday'] == 6) | (df['weekday'] == 7)).astype(int)

    df['logcpc'] = np.log(df['cpc'])
    df['logcpm'] = np.log(df['cpm'])

    df['mul'] = df['logcpc'] * df['logcpm']
    df['div'] = df['logcpc'] / df['logcpm']

    # 正交化
    columns_name=['logcpc','logcpm','daylight','weekend']
    df_ortho=df.copy(deep=True)
    df_ortho=df_ortho[columns_name]
    lr_models = joblib.load('./results/lr_models.pkl')
    for col in range(1,len(columns_name),1):
        X = df_ortho.iloc[:,0:col]
        y = df_ortho.iloc[:,col]
        lr = lr_models[col]
        df_ortho[columns_name[col]]=y-lr.predict(X) #计算残差截面

    # 标准化
    scaler = joblib.load('./results/scaler.pkl')
    data = scaler.transform(df_ortho)
    data = pd.DataFrame(data,columns=columns_name)

    # ========================  模型加载  ===========================
    # 请确认需要用到的列名，e.g.:columns = ['cpc','cpm']
#     columns = None
#     data = df[columns]
#     scaler = None
#     pca = None
    # 例如
    # scaler = joblib.load('./results/scaler.pkl')
    # pca = joblib.load('./results/pca.pkl')
    # data = scaler.transform(data)

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
    distance = pd.Series(distance)
    return distance
    

def get_anomaly(data, kmeans, ratio):
    """
    检验出样本中的异常点，并标记为 True 和 False，True 表示是异常点

    :param data: preprocess_data 函数返回值，即 pca 降维后的数据，DataFrame 类型
    :param kmeans: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param ratio: 异常数据占全部数据的百分比,在 0 - 1 之间，float 类型
    :return: data 添加 is_anomaly 列，该列数据是根据阈值距离大小判断每个点是否是异常值，元素值为 False 和 True
    """
    # ====================检验出样本中的异常点========================
    num_anomaly = int(len(data) * ratio)
    new_data = deepcopy(data)
    new_data['distance'] = get_distance(new_data,kmeans,n_features=len(new_data.columns))
    threshould = new_data['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
    new_data['is_anomaly'] = new_data['distance'].apply(lambda x: x > threshould)
    return new_data

def predict(preprocess_data):
    """
    该函数将被用于测试，请不要修改函数的输入输出，并按照自己的模型返回相关的数据。
    在函数内部加载 kmeans 模型并使用 get_anomaly 得到每个样本点异常值的判断
    :param preprocess_data: preprocess_data函数的返回值，一般是 DataFrame 类型
    :return:is_anomaly:get_anomaly函数的返回值，各个属性应该为（Dimesion1,Dimension2,......数量取决于具体的pca），distance,is_anomaly，请确保这些列存在
            preprocess_data:  即直接返回输入的数据
            kmeans: 通过joblib加载的对象
            ratio:  异常点的比例，ratio <= 0.03   返回非异常点得分将受到惩罚！
    """
    # 异常值所占比率
    ratio = 0.03
    # 加载模型 
    kmeans = joblib.load('./results/model.pkl')
    # 获取异常点数据信息
    is_anomaly = get_anomaly(preprocess_data, kmeans, 0.03)
    
    return is_anomaly, preprocess_data, kmeans, ratio
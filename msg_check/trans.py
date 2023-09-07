# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/23 18:37
# @Author  : subjadeites
# @File    : trans.py
# 导入相关的包
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pandas as pd
import numpy as np
# 数据集的路径
data_path = "./datasets/5f9ae242cae5285cd734b91e-momodel/sms_pub.csv"
# 读取数据
sms = pd.read_csv(data_path, encoding='utf-8')
# 显示前 5 条数据
sms.head()
# 显示数据集的一些信息
sms.groupby('label').describe()

def read_stopwords(stopwords_path):
    """
    读取停用词库
    :param stopwords_path: 停用词库的路径
    :return: 停用词列表
    """
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords = stopwords.splitlines()
    return stopwords
# 停用词库路径
stopwords_path = r'scu_stopwords.txt'
# 读取停用词
stopwords = read_stopwords(stopwords_path)
# 展示一些停用词
print(stopwords[-20:])


# 假如我们有这样三条短信
simple_train = ['call you tonight', 'Call me a cab', 'Please call me... PLEASE!']

# 导入 CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

# 从训练数据中学习词汇表
vect.fit(simple_train)

# 查看学习到的词汇表
vect.get_feature_names()

# 将训练数据向量化，得到一个矩阵
simple_train_dtm = vect.transform(simple_train)
# 由于该矩阵的维度可能十分大，而其中大部分都为 0，所以会采用稀疏矩阵来存储
print(simple_train_dtm)

# 将稀疏矩阵转为一般矩阵查看里面的内容
simple_train_dtm.toarray()

# 结合词汇表和转为得到的矩阵来直观查看内容
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())

# 导入 TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
# 在训练数据上拟合并将其转为为 tfidf 的稀疏矩阵形式
simple_train_dtm = tfidf.fit_transform(simple_train)
# 将稀疏矩阵转为一般矩阵
simple_train_dtm.toarray()
# 结合词汇表和转为得到的矩阵来直观查看内容
pd.DataFrame(simple_train_dtm.toarray(), columns=tfidf.get_feature_names())

# 构建训练集和测试集
from sklearn.model_selection import train_test_split
X = np.array(sms.msg_new)
y = np.array(sms.label)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
print("总共的数据大小", X.shape)
print("训练集数据大小", X_train.shape)
print("测试集数据大小", X_test.shape)

# 以 CountVectorizer 为例将数据集向量化
from sklearn.feature_extraction.text import CountVectorizer
# 设置匹配的正则表达式和停用词
vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
print(nb.fit(X_train_dtm, y_train) ) # 计算训练时间

# 对测试集的数据集进行预测
y_pred = nb.predict(X_test_dtm)
print(y_pred)

# 在测试集上评估训练的模型
from sklearn import metrics
print("在测试集上的混淆矩阵：")
print(metrics.confusion_matrix(y_test, y_pred))
print("在测试集上的分类结果报告：")
print(metrics.classification_report(y_test, y_pred))
print("在测试集上的 f1-score ：")
print(metrics.f1_score(y_test, y_pred))


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('cv', CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
    ('classifier', MultinomialNB()),
])

# 可以直接向 Pipeline 中输入文本数据进行训练和预测
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 得到的结果同上面分开的情况是一样的
print("在测试集上的混淆矩阵：")
print(metrics.confusion_matrix(y_test, y_pred))
print("在测试集上的分类结果报告：")
print(metrics.classification_report(y_test, y_pred))
print("在测试集上的 f1-score ：")
print(metrics.f1_score(y_test, y_pred))
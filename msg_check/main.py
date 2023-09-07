# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/23 18:44
# @Author  : subjadeites
# @File    : main.py
import warnings

warnings.filterwarnings('ignore')
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pandas as pd
import numpy as np

# from trans import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ---------- 停用词库路径，若有变化请修改 -------------
stopwords_path = r'scu_stopwords.txt'


# ---------------------------------------------------

def read_stopwords(stopwords_path):
    """
    读取停用词库
    :param stopwords_path: 停用词库的路径
    :return: 停用词列表，如 ['嘿', '很', '乎', '会', '或']
    """
    # ----------- 请完成读取停用词的代码 ------------
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords = stopwords.splitlines()
    # ----------------------------------------------

    return stopwords


# 读取停用词
stopwords = read_stopwords(stopwords_path)

# ----------------- 导入相关的库 -----------------
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler

# pipline_list用于传给Pipline作为参数
pipeline_list = [
    # --------------------------- 需要完成的代码 ------------------------------

    # ========================== 以下代码仅供参考 =============================
    ('cv', CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords,ngram_range=(1,2))),
    ('classifier', MultinomialNB())
    # ========================================================================

    # ------------------------------------------------------------------------
]

# 搭建 pipeline
pipeline = Pipeline(pipeline_list)

# 数据集的路径
data_path = "./datasets/5f9ae242cae5285cd734b91e-momodel/sms_pub.csv"
# 读取数据
sms = pd.read_csv(data_path, encoding='utf-8')
# 显示前 5 条数据
sms.head()
# 显示数据集的一些信息
sms.groupby('label').describe()

# 构建训练集和测试集
from sklearn.model_selection import train_test_split

X = np.array(sms.msg_new)
y = np.array(sms.label)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
print("总共的数据大小", X.shape)
print("训练集数据大小", X_train.shape)
print("测试集数据大小", X_test.shape)

# 训练 pipeline
pipeline.fit(X_train, y_train)

# 对测试集的数据集进行预测
y_pred = pipeline.predict(X_test)

# 在测试集上进行评估
from sklearn import metrics

print("在测试集上的混淆矩阵：")
print(metrics.confusion_matrix(y_test, y_pred))
print("在测试集上的分类结果报告：")
print(metrics.classification_report(y_test, y_pred))
print("在测试集上的 f1-score ：")
print(metrics.f1_score(y_test, y_pred))

# 在所有的样本上训练一次，充分利用已有的数据，提高模型的泛化能力
pipeline.fit(X, y)
# 保存训练的模型，请将模型保存在 results 目录下
import joblib

pipeline_path = 'results/pipeline.model'
joblib.dump(pipeline, pipeline_path)

# 加载训练好的模型
import joblib

# ------- pipeline 保存的路径，若有变化请修改 --------
pipeline_path = 'results/pipeline.model'
# --------------------------------------------------
pipeline = joblib.load(pipeline_path)


def predict(message):
    """
    预测短信短信的类别和每个类别的概率
    param: message: 经过jieba分词的短信，如"医生 拿 着 我 的 报告单 说 ： 幸亏 你 来 的 早 啊"
    return: label: 整数类型，短信的类别，0 代表正常，1 代表恶意
            proba: 列表类型，短信属于每个类别的概率，如[0.3, 0.7]，认为短信属于 0 的概率为 0.3，属于 1 的概率为 0.7
    """
    label = pipeline.predict([message])[0]
    proba = list(pipeline.predict_proba([message])[0])

    return label, proba


# 测试用例


result_list = ['2015 年 招标 师 考试 辅导 招生 方',
               '南京 游泳 培训 泳动 奇迹 游泳 培训 一对二 教学 成人 初学者 包教包会',
               '宠物 乘坐 飞机 需要 提前 预定 有 氧舱',
               'SDOUG 目前 所有 报名 通道 全部 截止',
               '本 公司 新到 各种规格 辐射 松 ， 澳松 ， 花旗 松 ， 无节材 ， 价格 优惠 ！ 欢迎 各位 新 老客户 来 人 来电 选购 … … 上海 傲寒 国际贸易 有限公司 业务经理 ： 陈强',
               '是否 考 金融 之类 研究生 值得 思考',
               '亚马逊 在线 零售商 公布 第二季度 业绩 实现 了 盈利',
               '本 宝宝 不 还是 进去 了 hhhhh',
               '今年 有 17 名 台湾 大学生 走进 嘉兴',
               '然而 自己 并 不是 从事 医生 这个 职业 只是 恰好 有 一身 行头 而已 看 完医龙 后 对 这份 职业 真的 是 充满 了 敬意 …',]

for i in result_list:
    label, proba = predict(i)
    print(label, proba)
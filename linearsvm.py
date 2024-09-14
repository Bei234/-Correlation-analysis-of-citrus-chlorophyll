# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# 加载数据集，你需要把数据放到目录中
# data = pd.read_csv("./data.csv")
data = pd.read_csv("./mydata.csv")

# 数据探索
# 因为数据集中列比较多，我们需要把dataframe中的列全部显示出来
pd.set_option('display.max_columns', None)
print(data.columns)
# print(data.head(5))
# print(data.describe())

# 将特征字段分成3组
# features_mean= list(data.columns[2:12])
# features_se= list(data.columns[12:22])
# features_worst=list(data.columns[22:32])
features_mean= list(data.columns[1:])

# 数据清洗
# ID列没有用，删除该列
data.drop("id",axis=1,inplace=True)
# data['diagnosis']=data['diagnosis'].map({'M':1,'P':0})
data['SPAD']=data['SPAD']



sns.countplot(data['SPAD'],label="Count")
plt.show()
# 用热力图呈现features_mean字段之间的相关性
corr = data[features_mean].corr()
# plt.figure(figsize=(14,14))
plt.figure(figsize=(6,6))
# annot=True显示每个方格的数据
sns.heatmap(corr, annot=True)
plt.show()


# 特征选择
# features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']
# features_remain = ['2','4', '5']
# features_remain = ['2']
features_remain = data.columns[1:7]

# print('features_remain:'%features_remain)
# print('-'*10)
# 抽取30%的数据作为测试集，其余作为训练集
train, test = train_test_split(data, test_size = 0.2)# in this our main data is splitted into train and test
# 抽取特征选择的数值作为训练和测试数据
train_X = train[features_remain]
train_y=train['SPAD']
test_X= test[features_remain]
test_y =test['SPAD']

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
# ss = StandardScaler()
# train_X = ss.fit_transform(train_X)
# test_X = ss.transform(test_X)

# 创建SVM分类器
# model = svm.LinearSVC()
# 用训练集做训练
# model.fit(train_X,train_y)
# 用测试集做预测
# prediction=model.predict(test_X)
# print('准确率: ', metrics.accuracy_score(prediction,test_y))


# 1.建立线性回归
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# model1 = LinearRegression()
model1 = svm.LinearSVC()
model1.fit(train_X, train_y)
print(model1.coef_)  ##检查模型的每个系数
prediction1=model1.predict(test_X)

print('线性回归准确率: ', metrics.accuracy_score(prediction1,test_y))
# print(metrics.mean_absolute_error(test_y, prediction1)) ##计算平均绝对误差
# print(metrics.mean_squared_error(test_y, prediction1)) ##计算均方误差
print(np.sqrt(metrics.mean_squared_error(test_y, prediction1)))   #计算均方根误差


from sklearn.linear_model import LogisticRegression
# 2该模型对应的回归函数为

modelLR=LogisticRegression()
b=modelLR.coef_
a=modelLR.intercept_
print('该模型对应的回归函数为:1/(1+exp-(%f+%f*x))'%(a,b))

# #画出相应的逻辑回归曲线

# exam_X=features_remain.values.reshape(-1,1)
# exam_y=data.values.reshape(-1,1)
plt.scatter(train_X,train_y,color='b',label='train data')
plt.scatter(test_X,test_y,color='r',label='test data')

#plt.plot(train_X,1/(1+np.exp(-(a+b*train_X))),color='r')
#plt.plot(test_X,1/(1+np.exp(-(a+b*test_X))),color='y')

# plt.plot(test_X,1/(1+np.exp(-(a+b*test_X))),color='r')
# plt.plot(exam_X,1/(1+np.exp(-(a+b*exam_X))),color='y')
plt.legend(loc=2)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
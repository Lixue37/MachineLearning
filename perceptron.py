# AUTHOR ：Ann
# TIME ：2022/4/25 20:31

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''数据准备'''
# 鸢尾花数据集
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
X = np.array(data.iloc[:100, [0, 1]])
y = iris.target[:100]
y = [-1 if i==0 else 1 for i in y]
# 数据集可视化
#plt.scatter(X[0:50, 0], X[0:50, 1], c="red", marker="x")
#plt.scatter(X[50:100, 0], X[50:100, 1], c="green")

'''参数初始化'''
w = np.array([0, 0])
b = 0
learning_rate = 1

'''定义损失函数'''


def loss_func(x, y, w, b):
    loss = y * (np.dot(w, x) + b)
    return loss


'''梯度下降函数'''


def gradient_func(x, y, w, b):
    w = w + learning_rate * y * x
    b = b + learning_rate * y
    return w, b


'''模型训练'''


def train(X, y, w, b):
    mistake = []
    for i, x in enumerate(X):
        loss = loss_func(x, y[i], w, b)
        if loss <= 0:
            w, b = gradient_func(x, y[i], w, b)
            mistake.append(1)
    return w, b, mistake


sum_mistake = 1
while (sum_mistake > 0):
    w, b, mistake = train(X, y, w, b)
    sum_mistake = np.sum(mistake)
print("finish")

'''可视化结果'''
print("w:", w)
print("b:", b)
x = np.linspace(4, 7, 10)
y = -(w[0] * x + b) / w[1]
plt.plot(x, y)
plt.scatter(X[:50, 0], X[:50, 1])
plt.scatter(X[50:100, 0], X[50:100, 1])
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.show()





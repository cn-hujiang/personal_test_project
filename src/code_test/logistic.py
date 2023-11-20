#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2023/2/23
# @Author : jiang.hu
# @File : logistic.py
import numpy as np
from matplotlib import pyplot as plt


def logistic(x):
    """
    逻辑斯提方程
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def plot_logistic():
    x = np.linspace(-5, 5, 100)
    y = logistic(x)

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Logistic Function')
    plt.show()


def decision_boundary(X, y, model):
    """
    决策边界
    :param X:
    :param y:
    :param model:
    :return:
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.show()


if __name__ == '__main__':
    pass

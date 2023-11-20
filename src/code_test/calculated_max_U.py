#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2023/2/6
# @Author : jiang.hu
# @File : calculated_max_U.py
import random


class CalculatedMaxU(object):
    """
    U=arr[1]×arr[2]×（1÷arr[3]）×arr[4]×。。。×arr[n-1]×（1÷arr[n]）
    可以理解为
    y= x1*(x2 * x4 * x6 * ... * x(n-1)) / (x1 * x3 * x5 * ... * xn)  n为奇数
    y= x1*(x2 * x4 * x6 * ... * xn) / (x1 * x3 * x5 * ... * x(n-1))  n为偶数
    """
    def __init__(self):
        pass

    @classmethod
    def is_even(cls, n):
        """奇偶数判断"""
        if n % 2 == 0:
            return True
        return False

    def calc_arr_product(self, arr):
        """
        计算数组乘积
        """
        length = len(arr)
        numerator = 1
        denominator = 1
        for i in range(length):
            if i == 0:
                numerator *= arr[0]
                continue
            if not self.is_even(i):
                numerator *= arr[i]
            else:
                denominator *= arr[i]

        return numerator / denominator


if __name__ == '__main__':
    array = [random.randint(1, 5) for i in range(10)]
    # calculated_max_u = CalculatedMaxU()
    # print(calculated_max_u.calc_arr_product(array))
    arr = array.copy()




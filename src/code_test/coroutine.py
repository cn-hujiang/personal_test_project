#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2023/2/8
# @Author : jiang.hu
# @File : coroutine.py
"""
协程
"""
import inspect


def simple_coroutine(a):
    print("a=", a)
    b = yield a
    print("b= ", b)
    c = yield a + b
    print("c=", c)
    yield c


if __name__ == '__main__':
    sc = simple_coroutine(6)
    print(inspect.getgeneratorstate(sc))
    print(sc)
    print(next(sc))
    print(inspect.getgeneratorstate(sc))
    print(sc.send(10))
    print(inspect.getgeneratorstate(sc))
    print(sc.send(99))


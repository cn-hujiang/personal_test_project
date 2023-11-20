#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2023/2/6
# @Author : jiang.hu
# @File : __init__.py.py

"""
堆栈
"""


class Stack(object):
    def __init__(self, size):
        self.size = size
        self.stack = []
        self.top = -1

    def push(self, element):  # 入栈之前检查栈是否已满
        if self.is_full():
            raise "栈空间已经存满"
        else:
            self.stack.append(element)
            self.top = self.top + 1

    def pop(self):  # 出栈之前检查栈是否为空
        if self.is_empty():
            raise "empty"
        else:
            self.top = self.top - 1
            return self.stack.pop()

    def is_full(self):
        return self.top + 1 == self.size

    def is_empty(self):
        return self.top == -1

    def len(self):
        return self.top


if __name__ == '__main__':
    # test
    s = Stack(20)
    for i in range(1, 20, 2):
        s.push(i)
    # 当前堆栈中的元素个数
    print(s.len())
    # 当前堆栈的允许最大存储元素
    print(s.size)
    # 当前堆栈中的元素
    print(s.stack)
    # 取出堆栈中的最顶元素值
    print(s.pop())
    # 判断当前堆栈是否已满
    print(s.is_full())
    # 判断当前堆栈是否为空
    print(s.is_empty())

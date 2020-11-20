#!/usr/bin/env python
"""
    Created by howie.hu at 2019-05-17.
    Description：感知机代码实现
    Changelog: all notable changes to this file will be documented
"""

import numpy as np


class Perceptron:
    """
    代码实现 Frank Rosenblatt 提出的感知器的与非门，加深对感知器的理解
    blog: https://www.howie6879.cn/post/2019/07_use_neural_network_recognize_handwriting/
    """

    def __init__(self, act_func, input_nums=2):
        """
        实例化一些基本参数
        :param act_func: 激活函数
        """
        # 激活函数
        self.act_func = act_func
        # 权重 已经确定只会有两个二进制输入
        self.w = np.zeros(input_nums)
        # 偏置项
        self.b = 0.0

    def fit(self, input_vectors, labels, learn_nums=10, rate=0.1):
        """
        训练出合适的 w 和 b
        :param input_vectors: 样本训练数据集
        :param labels: 标记值
        :param learn_nums: 学习多少次
        :param rate: 学习率
        """
        for i in range(learn_nums):
            for index, input_vector in enumerate(input_vectors):
                label = labels[index]
                output = self.predict(input_vector)
                delta = label - output
                self.w += input_vector * rate * delta
                self.b += rate * delta
        print("此时感知器权重为{0}，偏置项为{1}".format(self.w, self.b))
        return self

    def predict(self, input_vector):
        if isinstance(input_vector, list):
            input_vector = np.array(input_vector)
        return self.act_func(sum(self.w * input_vector) + self.b)


def f(z):
    """
    激活函数
    :param z: (w1*x1+w2*x2+...+wj*xj) + b
    :return: 1 or 0
    """
    return 1 if z > 0 else 0


def get_nand_gate_training_data():
    """
    NAND 训练数据集
    """
    input_vectors = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    labels = np.array([0, 1, 1, 1])
    return input_vectors, labels


def get_and_gate_training_data():
    """
    AND 训练数据集
    """
    input_vectors = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    labels = np.array([1, 0, 0, 0])
    return input_vectors, labels


def get_or_gate_training_data():
    """
    OR 训练数据集
    """
    input_vectors = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    labels = np.array([1, 1, 1, 0])
    return input_vectors, labels


if __name__ == "__main__":
    """
    输出如下：
        此时感知器权重为[ 0.1  0.2]，偏置项为-0.2 与门
        1 and 1 = 1
        1 and 0 = 0
        0 and 1 = 0
        0 and 0 = 0
        此时感知器权重为[ 0.2  0.2]，偏置项为-0.1 或门
        1 or 1 = 1
        1 or 0 = 1
        0 or 1 = 1
        0 or 0 = 0
        此时感知器权重为[-0.1 -0.2]，偏置项为0.20000000000000004 与非门
        1 nand 1 = 0
        1 nand 0 = 1
        0 nand 1 = 1
        0 nand 0 = 1
    """
    # 获取样本数据
    and_input_vectors, and_labels = get_and_gate_training_data()
    or_input_vectors, or_labels = get_or_gate_training_data()
    nand_input_vectors, nand_labels = get_nand_gate_training_data()
    # 实例化感知器模型
    p = Perceptron(f)
    # 开始学习 AND
    p_and = p.fit(and_input_vectors, and_labels)
    # 开始预测 AND
    print("1 and 1 = %d" % p_and.predict([1, 1]))
    print("1 and 0 = %d" % p_and.predict([1, 0]))
    print("0 and 1 = %d" % p_and.predict([0, 1]))
    print("0 and 0 = %d" % p_and.predict([0, 0]))

    # 开始学习 OR
    p_or = p.fit(or_input_vectors, or_labels)
    # 开始预测 OR
    print("1 or 1 = %d" % p_or.predict([1, 1]))
    print("1 or 0 = %d" % p_or.predict([1, 0]))
    print("0 or 1 = %d" % p_or.predict([0, 1]))
    print("0 or 0 = %d" % p_or.predict([0, 0]))

    # 开始学习 NAND
    p_or = p.fit(nand_input_vectors, nand_labels)
    # 开始预测 NAND
    print("1 nand 1 = %d" % p_or.predict([1, 1]))
    print("1 nand 0 = %d" % p_or.predict([1, 0]))
    print("0 nand 1 = %d" % p_or.predict([0, 1]))
    print("0 nand 0 = %d" % p_or.predict([0, 0]))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
knn module
"""

__author__ = 'chaoshen789'

from numpy import *
import operator


def classify0(in_x, data_set, labels, k):
    # 矩阵行数
    data_set_size = data_set.shape[0]
    # 将要进行分类的数据转为和数据集一样大小的矩阵，用来进行矩阵相减
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    # 对相减后得到的矩阵中的每一个元素求平方
    sq_diff_mat = diff_mat ** 2
    # 将列相加，压缩成一列
    sq_distances = sq_diff_mat.sum(axis=1)
    # 将压缩后为一列的矩阵中的所有元素开方得到传入特征与数据集中的所有特征的距离（欧式距离）
    distances = sq_distances ** 0.5
    # 将距离数组排序，并按照顺序返回原距离数组的索引
    sorted_dist_indicies = distances.argsort()
    # 标签计数器
    class_count = {}
    # 循环k次，取前k个最近的距离进行比较距离出现次数
    for i in list(range(k)):
        # 通过距离的索引拿到对应的标签
        vote_ilabel = labels[sorted_dist_indicies[i]]
        # 对标签进行计数
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
    # 将次数进行排序（次数降序）
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 取排序后的列表的第一个元素的第一个元素（即标签）
    return sorted_class_count[0][0]

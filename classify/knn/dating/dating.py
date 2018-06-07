#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dating module
"""

__author__ = 'chaoshen789'

from numpy import *
import classify.knn.knn as knn


def _file2matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0: 3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def _auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set /= tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = _file2matrix('E:\PycharmProjects\machinelearning\classify\knn\dating\datingTestSet2.txt')
    norm_mat, ranges, min_vals = _auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in list(range(num_test_vecs)):
        classifier_result = knn.classify0(norm_mat[i, :], norm_mat[num_test_vecs: m, :], dating_labels[num_test_vecs: m], 6)
        print('预测值: %d, 实际值: %d' % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print('错误率: %f' % (error_count/float(num_test_vecs)))


def classify_person():
    result_list = ['一点不喜欢', '还行', '很喜欢']
    percent_tats = float(input('玩视频游戏所耗时间百分比: '))
    ff_miles = float(input('每年获得的飞行常客里程数: '))
    ice_cream = float(input('每周消费的冰淇淋公升数: '))
    dating_data_mat, dating_labels = _file2matrix('E:\PycharmProjects\machinelearning\classify\knn\dating\datingTestSet2.txt')
    norm_mat, ranges, min_vals = _auto_norm(dating_data_mat)
    in_arr = array([ff_miles, ice_cream, percent_tats])
    classifier_result = knn.classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print('这个人可能属于的分类: ', result_list[classifier_result - 1])

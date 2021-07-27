#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: data_analysis.py
# Time: 2021/05
# Author: Zhu Yutao
# Description: 数据分析


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def null_data_extract(filepath: str):
    output_path = filepath + '_outlier'
    index_df = pd.read_csv(filepath_or_buffer=filepath)
    outlier_data = index_df[index_df.isna().T.any()]
    print(outlier_data)
    outlier_data.to_csv(path_or_buf=output_path)
    return outlier_data


def plot_t1_t0_sync(index_filepath: str, bug_filepath: str):
    """
    绘制同步时间（T1-T0）

    :param index_filepath: 指标数据文件路径
    :param bug_filepath: bug时间戳数据文件路径
    :return:
    """
    index_df = pd.read_csv(filepath_or_buffer=index_filepath)
    bug_df = pd.read_csv(filepath_or_buffer=bug_filepath)
    index_list = index_df['T1_T0_Sync'].tolist()
    time_list = index_df['T0_Sync'].tolist()
    index_max = index_df['T1_T0_Sync'].max()
    plt.scatter(x=time_list, y=index_list)
    for i, row in bug_df.iterrows():
        plt.vlines(x=row['Start_Timestamp'], ymin=0, ymax=index_max, colors='red')
        plt.vlines(x=row['End_Timestamp'], ymin=0, ymax=index_max, colors='green')
    plt.show()


def plot_t2_t1_sync(index_filepath: str, bug_filepath: str):
    """
    绘制同步时间（T1-T0）

    :param index_filepath: 指标数据文件路径
    :param bug_filepath: bug时间戳数据文件路径
    :return:
    """
    index_df = pd.read_csv(filepath_or_buffer=index_filepath)
    bug_df = pd.read_csv(filepath_or_buffer=bug_filepath)
    index_list = index_df['T2_T1_Sync'].tolist()
    time_list = index_df['T0_Sync'].tolist()
    index_max = index_df['T2_T1_Sync'].max()
    plt.scatter(x=time_list, y=index_list)
    for i, row in bug_df.iterrows():
        plt.vlines(x=row['Start_Timestamp'], ymin=0, ymax=index_max, colors='red')
        plt.vlines(x=row['End_Timestamp'], ymin=0, ymax=index_max, colors='green')
    plt.show()


def retransmission_sum(filepath: str):
    index_df = pd.read_csv(filepath_or_buffer=filepath)
    s = index_df['Retransmission_Times'].sum()
    return s

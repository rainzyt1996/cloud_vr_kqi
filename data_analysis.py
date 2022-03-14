#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: data_analysis.py
# Time: 2021/05
# Author: Zhu Yutao
# Description: 数据分析
import os.path

import logging
# import matplotlib.pyplot as plt
# from mifs import MutualInformationFeatureSelector as MIFS
# from minepy import MINE
# from pymrmr import mRMR
from pymrmre import mrmr
import numpy as np
import pandas as pd
import program_logging


# def null_data_extract(filepath: str):
#     output_path = filepath + '_outlier'
#     index_df = pd.read_csv(filepath_or_buffer=filepath)
#     outlier_data = index_df[index_df.isna().T.any()]
#     print(outlier_data)
#     outlier_data.to_csv(path_or_buf=output_path)
#     return outlier_data
#
#
# def plot_t1_t0_sync(index_filepath: str, bug_filepath: str):
#     """
#     绘制同步时间（T1-T0）
#
#     :param index_filepath: 指标数据文件路径
#     :param bug_filepath: bug时间戳数据文件路径
#     :return:
#     """
#     index_df = pd.read_csv(filepath_or_buffer=index_filepath)
#     bug_df = pd.read_csv(filepath_or_buffer=bug_filepath)
#     index_list = index_df['T1_T0_Sync'].tolist()
#     time_list = index_df['T0_Sync'].tolist()
#     index_max = index_df['T1_T0_Sync'].max()
#     plt.scatter(x=time_list, y=index_list)
#     for i, row in bug_df.iterrows():
#         plt.vlines(x=row['Start_Timestamp'], ymin=0, ymax=index_max, colors='red')
#         plt.vlines(x=row['End_Timestamp'], ymin=0, ymax=index_max, colors='green')
#     plt.show()
#
#
# def plot_t2_t1_sync(index_filepath: str, bug_filepath: str):
#     """
#     绘制同步时间（T1-T0）
#
#     :param index_filepath: 指标数据文件路径
#     :param bug_filepath: bug时间戳数据文件路径
#     :return:
#     """
#     index_df = pd.read_csv(filepath_or_buffer=index_filepath)
#     bug_df = pd.read_csv(filepath_or_buffer=bug_filepath)
#     index_list = index_df['T2_T1_Sync'].tolist()
#     time_list = index_df['T0_Sync'].tolist()
#     index_max = index_df['T2_T1_Sync'].max()
#     plt.scatter(x=time_list, y=index_list)
#     for i, row in bug_df.iterrows():
#         plt.vlines(x=row['Start_Timestamp'], ymin=0, ymax=index_max, colors='red')
#         plt.vlines(x=row['End_Timestamp'], ymin=0, ymax=index_max, colors='green')
#     plt.show()
#
#
# def retransmission_sum(filepath: str):
#     index_df = pd.read_csv(filepath_or_buffer=filepath)
#     s = index_df['Retransmission_Times'].sum()
#     return s


# def correlation():
#     mine = MINE(alpha=0.6, c=15, est="mic_approx")
#     dir_log = 'data/data_video/v5/v5_1/v5_1_1/ap_log_v5_1_1/'
#     path = dir_log + 'feature_label'
#     path_log = dir_log + 'mic.txt'
#     file = open(path_log, 'a')
#     data = pd.read_csv(filepath_or_buffer=path)
#     y = data['Label'].values.tolist()
#     data_pkt = ['T0', 'T1', 'T2']
#     index_pkt = ['T1_T0', 'T2_T1', 'T2_T0', 'Length', 'Retry']
#     index_stm = ['PacketRate', 'DataRate']
#     columns = []
#     # columns.extend(index_pkt)
#     region = [100, 200, 500, 1000, 2000]    # 100, 200, 500, 1000, 2000
#     for rgn in region:
#         # for item in data_pkt:
#         #     columns.append(item + '_' + str(rgn))
#         for item in index_pkt:
#             columns.append(item + '_Avg_' + str(rgn))
#             columns.append(item + '_Std_' + str(rgn))
#     columns.extend(index_stm)
#     for column in columns:
#         logging.info(column)
#         x = data[column].values.tolist()
#         mine.compute_score(x, y)
#         log = column + ': ' + str(mine.mic())
#         logging.info(log)
#         file.write(log + '\n')
#     file.close()


def mrmr_analysis(data, path_output: str):
    if type(data) is str:
        logging.info('mRMR analysis.(' + data + ')')
        data = pd.read_csv(filepath_or_buffer=data)
    elif type(data) is pd.DataFrame:
        logging.info('mRMR analysis.')
    else:
        print('Data type invalid!')
        return
    data.drop(columns=['Source', 'Destination', 'ID&Fragment', 'Protocol', 'T0', 'T1', 'T2', 'TS_No'],
              inplace=True)
    iLabel = data.columns.tolist().index('Label')
    label = data.iloc[:, iLabel:(iLabel + 1)]
    data.drop(columns=['Label'], inplace=True)
    # data = data.iloc[:, 0:3]
    # print(data)
    res = mrmr.mrmr_ensemble(features=data,
                             targets=label,
                             solution_length=data.shape[1])
    res = res.values.tolist()[0][0]
    file = open(path_output, 'w')
    file.write(',\n'.join([x for x in res]))
    file.close()
    print(res)


if __name__ == '__main__':
    mrmr_analysis()

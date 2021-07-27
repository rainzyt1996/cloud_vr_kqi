#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: data_preprocess.py
# Time: 2021/05
# Author: Zhu Yutao
# Description:
import logging
import numpy as np
import os
import pandas as pd
import program_logging


def null_data_delete(filepath: str, print_on=False, to_csv_on=True):
    index_df = pd.read_csv(filepath_or_buffer=filepath)
    outlier_list = index_df[index_df.isna().T.any()].index.tolist()
    index_df = index_df.drop(outlier_list)
    if print_on:
        print(index_df)
    if to_csv_on:
        output_path = filepath + '_nonnull'
        index_df.to_csv(path_or_buf=output_path, index=False)
    return index_df


def index_null_fill(filepath: str, print_on=False, to_csv_on=True):
    """
    空值填充处理

    :param filepath: 指标数据文件路径
    :param print_on: 输出打印开关
    :param to_csv_on: 输出至文件开关
    :return:
    """
    logging.info('Index null fill start.')
    column = ['Source',
              'Destination',
              'ID&Fragment',
              'Protocol',
              'T0_Sync',
              'T0_Local',
              'T1_Sync',
              'T1_Local',
              'T2_Sync',
              'T2_Local',
              'T1_T0_Sync',
              'T1_T0_Local',
              'T2_T1_Sync',
              'T2_T1_Local',
              'Retransmission_Times']
    index_df = pd.read_csv(filepath_or_buffer=filepath)
    len_index = len(index_df)
    index_arr = index_df.values

    # 处理T0空值
    for i in range(0, len_index):
        if i % int(len_index / 100) == 0:
            logging.info('Index null fill ... (%d%%)', i * 100 / len_index)
        if pd.isna(index_arr[i, 4]):
            n = 0
            for j in range(i, len_index):
                if pd.isna(index_arr[j, 4]):
                    n = n + 1
                else:
                    break
            for k in range(1, n + 1):
                index_arr[i - 1 + k, 4] = np.around((index_arr[i - 2 + k, 4] + index_arr[i - 1 + k, 6]) / 2, 6)
                index_arr[i - 1 + k, 5] = np.around((index_arr[i - 2 + k, 5] + index_arr[i - 1 + k, 7]) / 2, 6)
                index_arr[i - 1 + k, 10] = np.around(index_arr[i - 1 + k, 6] - index_arr[i - 1 + k, 4], 6)
                index_arr[i - 1 + k, 11] = np.around(index_arr[i - 1 + k, 7] - index_arr[i - 1 + k, 5], 6)
        if pd.isna(index_arr[i, 14]):
            index_arr[i, 14] = 0

    index_fill_df = pd.DataFrame(data=index_arr, columns=column)

    if print_on:
        print(index_fill_df)
    if to_csv_on:
        output_path = filepath + '_fill'
        index_fill_df.to_csv(path_or_buf=output_path, index=False)
    logging.info('Index null fill complete.')
    return index_fill_df


def set_label(index_filepath: str, bug_filepath: str, print_on=False, to_csv_on=True):
    index_df = pd.read_csv(filepath_or_buffer=index_filepath)
    bug_df = pd.read_csv(filepath_or_buffer=bug_filepath)

    index_df['Label'] = 0

    for i, row in bug_df.iterrows():
        if row['Type'] == 'caton':
            start_time = row['Start_Timestamp']
            end_time = row['End_Timestamp']
            index_df.loc[(index_df['T0_Sync'] >= start_time) & (index_df['T0_Sync'] <= end_time), 'Label'] = 1

    if print_on:
        print(index_df)
    if to_csv_on:
        output_path = os.path.join(os.path.dirname(index_filepath), 'index_label')
        index_df.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")

    return index_df

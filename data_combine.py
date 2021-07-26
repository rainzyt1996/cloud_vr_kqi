#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: data_combine.py
# Time: 2021/05
# Author: Zhu Yutao
# Description: 数据融合
import logging
import os
import pandas as pd

import program_logging
import data_import as di


def combine_timestamp(filepath0: str, filepath1: str, filepath2: str, print_on=False, to_csv_on=True):
    """
    路由器时间戳log数据合并

    :param filepath0: T0数据文件路径
    :param filepath1: T1数据文件路径
    :param filepath2: T2数据文件路径
    :param print_on: 打印至控制台开关
    :param to_csv_on: 输出至csv文件开关
    :return: 合并log数据
    """
    logging.info('Data merge ... (1/3)')
    log0 = di.import_timestamp(filepath=filepath0)
    log1 = di.import_timestamp(filepath=filepath1)
    log2 = pd.concat(objs=[log0, log1])
    logging.info('Data merge ... (2/3)')
    log1 = di.import_timestamp(filepath=filepath2)
    log0 = pd.concat(objs=[log2, log1])
    logging.info('Data merge ... (3/3)')
    logging.info('Data sort ...')
    log0.sort_values(by=['Time_Sync', 'Time_Local'], inplace=True)
    logging.info('Data output ...')
    if print_on:
        print(log0)
    if to_csv_on:
        output_path = os.path.join(os.path.dirname(filepath0), 'log_tall')
        log0.to_csv(path_or_buf=output_path, index=False)
    logging.info('Data merge complete.')
    return log0


def combine_bug_timestamp(log_filepath: str, mark_filepath: str, print_on=False, to_csv_on=True):
    log_df = pd.read_csv(filepath_or_buffer=log_filepath,
                         header=None,
                         names=['Filepath', 'Start_Timestamp', 'End_Timestamp'])
    mark_df = pd.read_csv(filepath_or_buffer=mark_filepath)
    mark_df['Start_Timestamp'] = mark_df['Start_Time'] + mark_df['Start_Frame'] / 30 + log_df.iloc[0, 1] / 1000
    mark_df['End_Timestamp'] = mark_df['End_Time'] + mark_df['End_Frame'] / 30 + log_df.iloc[0, 1] / 1000

    if print_on:
        print(mark_df)
    if to_csv_on:
        output_path = os.path.join(os.path.dirname(mark_filepath), 'log_bug_timestamp')
        mark_df.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")

    return mark_df

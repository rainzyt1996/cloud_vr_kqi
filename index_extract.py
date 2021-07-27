#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: extract_index.py
# Time: 2021/05
# Author: Zhu Yutao
# Description: 指标提取
import logging
import numpy as np
import pandas as pd
import program_logging


def index_extract(log_df: pd.DataFrame, output_filepath=None, print_on=False, to_csv_on=False):
    """
    提取指标

    :param log_df: 路由器时间戳log数据
    :param output_filepath: 输出csv文件路径
    :param print_on: 打印至控制台开关
    :param to_csv_on: 输出至csv文件开关
    :return: 指标列表
    """
    logging.info('Index extract start.')
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
    index_df = pd.DataFrame(columns=column)
    obj_curr = pd.Series(data=[None] * len(column), index=column)
    len_log = len(log_df)
    num_read = [0]

    # 添加读取标志
    log_df['Flag_Read'] = log_df['Tn'] * 0

    def log_statistics(idx_start: int):
        """
        数据匹配，统计指标

        :param idx_start: 开始索引
        :return:
        """
        # 判断索引是否越界
        if idx_start < len_log:
            # 开始遍历匹配
            for idx_cal in range(idx_start, len_log):
                # 判断数据标识项是否匹配
                if (obj_curr['Source':'Protocol'] != log_df.loc[idx_cal, 'Source':'Protocol']).sum() == 0:
                    # 判断当前log行Tn项的值
                    tn = log_df.loc[idx_cal, 'Tn']
                    if tn == 0:
                        # 判断缓存变量中的T0, T1, T2
                        if (obj_curr['T0_Sync':'T2_Local'].values == [None] * 6).sum() == 6:
                            obj_curr['T0_Sync'] = log_df.loc[idx_cal, 'Time_Sync']
                            obj_curr['T0_Local'] = log_df.loc[idx_cal, 'Time_Local']
                            log_df.loc[idx_cal, 'Flag_Read'] = 1
                            num_read[0] = num_read[0] + 1
                            logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        else:
                            break
                    elif tn == 1:
                        # 判断缓存变量中的T1, T2
                        if (obj_curr['T1_Sync':'T2_Local'].values == [None] * 4).sum() == 4:
                            obj_curr['T1_Sync'] = log_df.loc[idx_cal, 'Time_Sync']
                            obj_curr['T1_Local'] = log_df.loc[idx_cal, 'Time_Local']
                            log_df.loc[idx_cal, 'Flag_Read'] = 1
                            num_read[0] = num_read[0] + 1
                            logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        elif (obj_curr['T1_Sync':'T1_Local'].values == [None] * 2).sum() < 2 and (obj_curr['T2_Sync':'T2_Local'].values == [None] * 2).sum() == 2:
                            if obj_curr['Retransmission_Times'] is None:
                                obj_curr['Retransmission_Times'] = 1
                            else:
                                obj_curr['Retransmission_Times'] = obj_curr['Retransmission_Times'] + 1
                            log_df.loc[idx_cal, 'Flag_Read'] = 1
                            num_read[0] = num_read[0] + 1
                            logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        else:
                            break
                    elif tn == 2:
                        # 判断缓存变量中的T2
                        if (obj_curr['T2_Sync':'T2_Local'].values == [None] * 2).sum() == 2:
                            obj_curr['T2_Sync'] = log_df.loc[idx_cal, 'Time_Sync']
                            obj_curr['T2_Local'] = log_df.loc[idx_cal, 'Time_Local']
                            log_df.loc[idx_cal, 'Flag_Read'] = 1
                            num_read[0] = num_read[0] + 1
                            logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        break
                    else:
                        logging.error('Invalid data: [%d]Tn=%d', idx_cal, tn)
                        break

    # 遍历数据，计算指标
    for idx_ex in range(0, len_log):
        # 判断当前log行是否已读取
        if log_df.loc[idx_ex, 'Flag_Read'] == 0:
            # 将当前log数据标识项写入缓存变量
            obj_curr['Source':'Protocol'] = log_df.loc[idx_ex, 'Source':'Protocol']
            # 计算当前log数据标识项对应指标
            log_statistics(idx_start=idx_ex)
            obj_curr = index_calculate(obj_curr)
            # 将缓存变量写入结果，清空缓存变量
            index_df = index_df.append(obj_curr, ignore_index=True)
            obj_curr.loc['Source':'Retransmission_Times'] = None

    if print_on:
        print(index_df)
    if to_csv_on:
        if output_filepath is None:
            logging.error('Invalid parameter: extract_index->output_filepath=None')
        else:
            index_df.to_csv(path_or_buf=output_filepath, index=False)
    logging.info('Index extract complete.')
    return index_df


def index_extract2(log_df: pd.DataFrame, output_filepath=None, print_on=False, to_csv_on=False):
    """
    提取指标

    :param log_df: 路由器时间戳log数据
    :param output_filepath: 输出csv文件路径
    :param print_on: 打印至控制台开关
    :param to_csv_on: 输出至csv文件开关
    :return: 指标列表
    """
    logging.info('Index extract start.')
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
    index_df = pd.DataFrame(columns=column)
    obj_curr = pd.Series(data=[None] * len(column), index=column)
    len_log = len(log_df)
    # num_read = [0]

    # 添加读取标志
    log_df['Flag_Read'] = log_df['Tn'] * 0

    log_arr = log_df.values

    def log_statistics(idx_start: int):
        """
        数据匹配，统计指标

        :param idx_start: 开始索引
        :return:
        """
        # 判断索引是否越界
        if idx_start < len_log:
            # 开始遍历匹配
            for idx_cal in range(idx_start, len_log):
                # 判断数据标识项是否匹配
                if obj_curr['Source'] == log_arr[idx_cal, 1] and obj_curr['Destination'] == log_arr[idx_cal, 2] and obj_curr['ID&Fragment'] == log_arr[idx_cal, 3] and obj_curr['Protocol'] == log_arr[idx_cal, 4]:
                    # 判断当前log行Tn项的值
                    tn = log_arr[idx_cal, 0]
                    if tn == 0:
                        # 判断缓存变量中的T0, T1, T2
                        if (obj_curr['T0_Sync':'T2_Local'].values == [None] * 6).sum() == 6:
                            obj_curr['T0_Sync'] = log_arr[idx_cal, 5]
                            obj_curr['T0_Local'] = log_arr[idx_cal, 6]
                            log_arr[idx_cal, 7] = 1
                            # num_read[0] = num_read[0] + 1
                            # logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        else:
                            break
                    elif tn == 1:
                        # 判断缓存变量中的T1, T2
                        if (obj_curr['T1_Sync':'T2_Local'].values == [None] * 4).sum() == 4:
                            obj_curr['T1_Sync'] = log_arr[idx_cal, 5]
                            obj_curr['T1_Local'] = log_arr[idx_cal, 6]
                            log_arr[idx_cal, 7] = 1
                            # num_read[0] = num_read[0] + 1
                            # logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        elif (obj_curr['T1_Sync':'T1_Local'].values == [None] * 2).sum() < 2 and (obj_curr['T2_Sync':'T2_Local'].values == [None] * 2).sum() == 2:
                            if obj_curr['Retransmission_Times'] is None:
                                obj_curr['Retransmission_Times'] = 1
                            else:
                                obj_curr['Retransmission_Times'] = obj_curr['Retransmission_Times'] + 1
                            log_arr[idx_cal, 7] = 1
                            # num_read[0] = num_read[0] + 1
                            # logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        else:
                            break
                    elif tn == 2:
                        # 判断缓存变量中的T2
                        if (obj_curr['T2_Sync':'T2_Local'].values == [None] * 2).sum() == 2:
                            obj_curr['T2_Sync'] = log_arr[idx_cal, 5]
                            obj_curr['T2_Local'] = log_arr[idx_cal, 6]
                            log_arr[idx_cal, 7] = 1
                            # num_read[0] = num_read[0] + 1
                            # logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        break
                    else:
                        logging.error('Invalid data: [%d]Tn=%d', idx_cal, tn)
                        break

    # 遍历数据，计算指标
    for idx_ex in range(0, len_log):
        if idx_ex % 10000 == 0:
            logging.info('Index extracting ... (%d/%d)', idx_ex, len_log)
        # 判断当前log行是否已读取
        if log_arr[idx_ex, 7] == 0:
            # 将当前log数据标识项写入缓存变量
            obj_curr['Source'] = log_arr[idx_ex, 1]
            obj_curr['Destination'] = log_arr[idx_ex, 2]
            obj_curr['ID&Fragment'] = log_arr[idx_ex, 3]
            obj_curr['Protocol'] = log_arr[idx_ex, 4]
            # 计算当前log数据标识项对应指标
            log_statistics(idx_start=idx_ex)
            obj_curr = index_calculate(obj_curr)
            # 将缓存变量写入结果，清空缓存变量
            index_df = index_df.append(obj_curr, ignore_index=True)
            obj_curr.loc['Source':'Retransmission_Times'] = None

    if print_on:
        print(index_df)
    if to_csv_on:
        if output_filepath is None:
            logging.error('Invalid parameter: extract_index->output_filepath=None')
        else:
            index_df.to_csv(path_or_buf=output_filepath, index=False)
    logging.info('Index extract complete.')
    return index_df


def index_extract3(log_df: pd.DataFrame, output_filepath=None, print_on=False, to_csv_on=False):
    """
    提取指标

    :param log_df: 路由器时间戳log数据
    :param output_filepath: 输出csv文件路径
    :param print_on: 打印至控制台开关
    :param to_csv_on: 输出至csv文件开关
    :return: 指标列表
    """
    logging.info('Index extract start.')
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
    # index_res = pd.DataFrame(columns=column)
    # obj_curr = pd.Series(data=[None] * len(column), index=column)
    obj_curr = [None] * len(column)
    index_res = [obj_curr]
    len_log = len(log_df)
    # num_read = [0]

    # 添加读取标志
    log_df['Flag_Read'] = log_df['Tn'] * 0

    log_arr = log_df.values

    def log_statistics(idx_start: int):
        """
        数据匹配，统计指标

        :param idx_start: 开始索引
        :return:
        """
        # 判断索引是否越界
        if idx_start < len_log:
            # 开始遍历匹配
            for idx_cal in range(idx_start, len_log):
                # 判断数据标识项是否匹配
                if obj_curr[0] == log_arr[idx_cal, 1] and obj_curr[1] == log_arr[idx_cal, 2] and obj_curr[2] == log_arr[idx_cal, 3] and obj_curr[3] == log_arr[idx_cal, 4]:
                    # 判断当前log行Tn项的值
                    tn = log_arr[idx_cal, 0]
                    if tn == 0:
                        # 判断缓存变量中的T0, T1, T2
                        if obj_curr[4:10] == [None] * 6:
                            obj_curr[4] = log_arr[idx_cal, 5]
                            obj_curr[5] = log_arr[idx_cal, 6]
                            log_arr[idx_cal, 7] = 1
                            # num_read[0] = num_read[0] + 1
                            # logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        else:
                            break
                    elif tn == 1:
                        # 判断缓存变量中的T1, T2
                        if obj_curr[6:10] == [None] * 4:
                            obj_curr[6] = log_arr[idx_cal, 5]
                            obj_curr[7] = log_arr[idx_cal, 6]
                            log_arr[idx_cal, 7] = 1
                            # num_read[0] = num_read[0] + 1
                            # logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        elif obj_curr[6:8] != [None] * 2 and obj_curr[8:10] == [None] * 2:
                            if obj_curr[14] is None:
                                obj_curr[14] = 1
                            else:
                                obj_curr[14] = obj_curr[14] + 1
                            log_arr[idx_cal, 7] = 1
                            # num_read[0] = num_read[0] + 1
                            # logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        else:
                            break
                    elif tn == 2:
                        # 判断缓存变量中的T2
                        if obj_curr[8:10] == [None] * 2:
                            obj_curr[8] = log_arr[idx_cal, 5]
                            obj_curr[9] = log_arr[idx_cal, 6]
                            log_arr[idx_cal, 7] = 1
                            # num_read[0] = num_read[0] + 1
                            # logging.info('Index extracting ... (%d/%d)', num_read[0], len_log)
                        break
                    else:
                        logging.error('Invalid data: [%d]Tn=%d', idx_cal, tn)
                        break

    # 遍历数据，计算指标
    for idx_ex in range(0, len_log):
        if idx_ex % 10000 == 0:
            logging.info('Index extracting ... (%d/%d)', idx_ex, len_log)
        # 判断当前log行是否已读取
        if log_arr[idx_ex, 7] == 0:
            # 将当前log数据标识项写入缓存变量
            obj_curr[0] = log_arr[idx_ex, 1]
            obj_curr[1] = log_arr[idx_ex, 2]
            obj_curr[2] = log_arr[idx_ex, 3]
            obj_curr[3] = log_arr[idx_ex, 4]
            obj_curr[14] = 0
            # 计算当前log数据标识项对应指标
            log_statistics(idx_start=idx_ex)
            obj_curr = index_calculate3(obj_curr)
            # 将缓存变量写入结果，清空缓存变量
            index_res.append(obj_curr)
            obj_curr = [None] * len(column)
    index_df = pd.DataFrame(data=index_res, columns=column)
    index_df.drop([0], inplace=True)

    if print_on:
        print(index_df)
    if to_csv_on:
        if output_filepath is None:
            logging.error('Invalid parameter: extract_index->output_filepath=None')
        else:
            index_df.to_csv(path_or_buf=output_filepath, index=False)
    logging.info('Index extract complete.')
    return index_df


def index_calculate(log_s: pd.Series):
    """
    计算指标

    :param log_s: 单条log数据
    :return: 带指标的log数据
    """
    if log_s['T0_Sync'] is not None and log_s['T1_Sync'] is not None:
        log_s['T1_T0_Sync'] = np.around(log_s['T1_Sync'] - log_s['T0_Sync'], 6)
    if log_s['T0_Local'] is not None and log_s['T1_Local'] is not None:
        log_s['T1_T0_Local'] = np.around(log_s['T1_Local'] - log_s['T0_Local'], 6)
    if log_s['T2_Sync'] is not None and log_s['T1_Sync'] is not None:
        log_s['T2_T1_Sync'] = np.around(log_s['T2_Sync'] - log_s['T1_Sync'], 6)
    if log_s['T2_Local'] is not None and log_s['T1_Local'] is not None:
        log_s['T2_T1_Local'] = np.around(log_s['T2_Local'] - log_s['T1_Local'], 6)
    return log_s


def index_calculate3(log_l: list):
    """
    计算指标

    :param log_l: 单条log数据
    :return: 带指标的log数据
    """
    if log_l[4] is not None and log_l[6] is not None:
        log_l[10] = np.around(log_l[6] - log_l[4], 6)
    if log_l[5] is not None and log_l[7] is not None:
        log_l[11] = np.around(log_l[7] - log_l[5], 6)
    if log_l[8] is not None and log_l[6] is not None:
        log_l[12] = np.around(log_l[8] - log_l[6], 6)
    if log_l[9] is not None and log_l[7] is not None:
        log_l[13] = np.around(log_l[9] - log_l[7], 6)
    return log_l

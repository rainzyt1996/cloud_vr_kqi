#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: data_utils.py
# Time: 2021/06
# Author: Zhu Yutao
# Description:

import logging
import numpy as np
import os
import pandas as pd
import program_logging
import progressbar as bar
from pyshark import FileCapture


class DataUtils:

    def __init__(self):
        self.columnTimestamp = ['Tn', 'Source', 'Destination', 'ID&Fragment', 'Protocol', 'Time_Sync', 'Time_Local']
        self.columnInfo = ['Source', 'Destination', 'ID', 'Flags', 'Protocol', 'Length']
        self.columnIndex = ['Source', 'Destination', 'ID&Fragment', 'Protocol',
                            'T0_Sync', 'T0_Local', 'T1_Sync', 'T1_Local', 'T2_Sync', 'T2_Local',
                            'T1_T0_Sync', 'T1_T0_Local', 'T2_T1_Sync', 'T2_T1_Local', 'Retransmission_Times']
        self.captureFilter = 'ip.dst == 192.168.200.160'

    def combine_bug_timestamp(self, path_vtsp: str, path_bm: str, print_on=False, to_csv_on=True):
        """
        合并 视频时间戳&异常标注

        :param path_vtsp: 视频时间戳文件路径
        :param path_bm: 异常标注文件路径
        :param print_on: 是否打印
        :param to_csv_on: 是否输出csv文件
        :return:
        """
        vtsp = pd.read_csv(filepath_or_buffer=path_vtsp,
                           header=None,
                           names=['Filepath', 'Start_Timestamp', 'End_Timestamp'])
        btsp = pd.read_csv(filepath_or_buffer=path_bm)
        btsp['Start_Timestamp'] = btsp['Start_Time'] + btsp['Start_Frame'] / 30 + vtsp.iloc[0, 1] / 1000
        btsp['End_Timestamp'] = btsp['End_Time'] + btsp['End_Frame'] / 30 + vtsp.iloc[0, 1] / 1000
        if print_on:
            print(btsp)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(path_bm), 'log_bug_timestamp')
            btsp.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return btsp

    def combine_router_timestamp(self, filepath_list: list, print_on=False, to_csv_on=True):
        """
        合并 路由器时间戳数据T0&T1&T2

        :param filepath_list: T0,T1,T2文件路径列表
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return: 整体路由器时间戳数据
        """
        if len(filepath_list) != 3:
            logging.error('Wrong number(%d) of lists. The number of target lists is 3.', len(filepath_list))
            return
        log0 = self.import_router_timestamp(filepath=filepath_list[0])
        logging.info('Data merge ... (1/3)')
        log1 = self.import_router_timestamp(filepath=filepath_list[1])
        log2 = pd.concat(objs=[log0, log1])
        logging.info('Data merge ... (2/3)')
        log1 = self.import_router_timestamp(filepath=filepath_list[2])
        log0 = pd.concat(objs=[log2, log1])
        logging.info('Data merge ... (3/3)')
        log0.sort_values(by=['Time_Sync', 'Time_Local'], inplace=True)
        logging.info('Data sort ...')
        if print_on:
            print(log0)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(filepath_list[0]), 'log_tall')
            log0.to_csv(path_or_buf=output_path, index=False)
            logging.info('Data output ...')
        logging.info('Data merge complete.')
        return log0

    def delete_null_data(self, filepath: str, print_on=False, to_csv_on=True):
        """
        删除空数据所在行

        :param filepath: 文件路径
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return: 处理后数据
        """
        logging.info('Null data delete start.')
        data = pd.read_csv(filepath_or_buffer=filepath)
        null_list = data[data.isna().T.any()].index.tolist()
        data = data.drop(null_list)
        if print_on:
            print(data)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(filepath), os.path.basename(filepath) + '_nonnull')
            data.to_csv(path_or_buf=output_path, index=False)
        logging.info('Null data delete complete.')
        return data

    def extract_capture_info(self, filepath: str, print_on=False, to_csv_on=True):
        # logging.info('Capture info extract start.')
        capture = self.import_capture(filepath=filepath, keep_packets=False)
        pgb = bar.ProgressBar()
        info_list = []
        i = [0]

        def get_packet_info(packet):
            if 'wlan_aggregate' in dir(packet):
                if isinstance(packet.wlan_aggregate._all_fields['wlan_aggregate.a_mdsu.subframe'], dict):
                    a_msdu_subframe = [packet.wlan_aggregate._all_fields['wlan_aggregate.a_mdsu.subframe']]
                elif isinstance(packet.wlan_aggregate._all_fields['wlan_aggregate.a_mdsu.subframe'], list):
                    a_msdu_subframe = packet.wlan_aggregate._all_fields['wlan_aggregate.a_mdsu.subframe']
                else:
                    logging.error('wlan_aggregate error: %d', i[0])
                    return
                for subframe in a_msdu_subframe:
                    info = [subframe['ip']['ip.src'],
                            subframe['ip']['ip.dst'],
                            subframe['ip']['ip.id'],
                            subframe['ip']['ip.flags'],
                            subframe['ip']['ip.proto']]
                    info_list.append(info)
                    i[0] += 1
                    pgb.update(i[0])

        capture.apply_on_packets(get_packet_info)
        if print_on:
            print(info_list)
        if to_csv_on:
            info_df = pd.DataFrame(data=info_list, columns=self.columnInfo)
            info_df.to_csv(path_or_buf=os.path.join(os.path.dirname(filepath), 'capture_info'), index=False)
        # logging.info('Capture info extract complete.')
        return info_list

    def extract_index(self, rt_tsp: pd.DataFrame, output_path=None, print_on=False, to_csv_on=False):
        """
        提取指标

        :param rt_tsp: 路由器时间戳数据
        :param output_path: 输出csv文件路径
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return: 指标列表
        """
        logging.info('Index extract start.')
        column = self.columnIndex
        obj_curr = [None] * len(column)
        index_res = [obj_curr]
        len_log = len(rt_tsp)
        # 添加读取标志
        rt_tsp['Flag_Read'] = rt_tsp['Tn'] * 0
        log_arr = rt_tsp.values

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
                    if obj_curr[0] == log_arr[idx_cal, 1] and obj_curr[1] == log_arr[idx_cal, 2] and obj_curr[2] == \
                            log_arr[idx_cal, 3] and obj_curr[3] == log_arr[idx_cal, 4]:
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

        def index_calculate(log_l: list):
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

        # 遍历数据，计算指标
        for idx_ex in bar.progressbar(range(0, len_log)):
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
                obj_curr = index_calculate(obj_curr)
                # 将缓存变量写入结果，清空缓存变量
                index_res.append(obj_curr)
                obj_curr = [None] * len(column)
        index_df = pd.DataFrame(data=index_res, columns=column)
        index_df.drop([0], inplace=True)

        if print_on:
            print(index_df)
        if to_csv_on:
            if output_path is None:
                logging.error('Invalid parameter: extract_index->output_filepath=None')
            else:
                index_df.to_csv(path_or_buf=output_path, index=False)
        logging.info('Index extract complete.')
        return index_df

    def fill_null_index(self, filepath: str, print_on=False, to_csv_on=True):
        """
        空值填充处理

        :param filepath: 指标数据文件路径
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return: 处理后数据
        """
        logging.info('Index null fill start.')
        column = self.columnIndex
        index_df = pd.read_csv(filepath_or_buffer=filepath)
        len_index = len(index_df)
        index_arr = index_df.values
        # 处理T0空值
        for i in bar.progressbar(range(0, len_index)):
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
            output_path = os.path.join(os.path.dirname(filepath), os.path.basename(filepath) + '_fill')
            index_fill_df.to_csv(path_or_buf=output_path, index=False)
        logging.info('Index null fill complete.')
        return index_fill_df

    def filter(self, filepath_or_dataframe, key: str, value, print_on=False, to_csv_on=True, savepath='./data_filter'):
        """
        筛选某一属性为特定值的数据

        :param filepath_or_dataframe: 文件路径或数据
        :param key: 属性名称
        :param value: 属性值
        :param print_on: 是否打印
        :param to_csv_on: 是否输出为csv文件
        :param savepath: 保存路径
        :return: 筛选后的数据表
        """
        if type(filepath_or_dataframe) is str:
            data = pd.read_csv(filepath_or_buffer=filepath_or_dataframe)
        elif type(filepath_or_dataframe) is pd.DataFrame:
            data = filepath_or_dataframe
        else:
            logging.error('Parameter(filepath_or_dataframe) is error!')
            return None
        if key in data.columns:
            data = data[data[key] == value]
        else:
            logging.error('The key is not in data!')
            return None
        if print_on:
            print(data)
        if to_csv_on:
            data.to_csv(path_or_buf=savepath, index=False)
        return data

    def import_capture(self, filepath: str, only_summaries=False, keep_packets=True, use_json=True):
        """
        空口数据导入

        :param filepath: 空口数据文件路径
        :param only_summaries: 仅主要信息
        :param keep_packets: 遍历时保存每一步packet
        :param use_json: 通过tshark json解析packet（针对packet json中含有重复的key）
        :return
        """
        capture = FileCapture(input_file=filepath,
                              display_filter=self.captureFilter,
                              only_summaries=only_summaries,
                              keep_packets=keep_packets,
                              use_json=use_json)
        return capture

    def import_router_timestamp(self, filepath: str, print_on=False):
        """
        路由器时间戳原始数据导入

        :param filepath: 数据文件路径
        :param print_on: 是否打印结果
        :return: 路由器时间戳原始数据表
        """
        router_timestamp = pd.read_csv(filepath_or_buffer=filepath,
                                       header=None,
                                       names=self.columnTimestamp)
        if print_on:
            print(router_timestamp)
        return router_timestamp

    def set_label(self, index_filepath: str, bug_filepath: str, print_on=False, to_csv_on=True):
        """
        设置标签

        :param index_filepath: 特征数据文件路径
        :param bug_filepath: 异常时间戳文件路径
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return:
        """
        logging.info('Label set start.')
        index_df = pd.read_csv(filepath_or_buffer=index_filepath)
        bug_df = pd.read_csv(filepath_or_buffer=bug_filepath)
        index_df['Label'] = 0
        for i, row in bug_df.iterrows():
            if row['Type'] == 'caton':
                start_time = row['Start_Time']
                end_time = row['End_Time']
                index_df.loc[(index_df['T0_Sync'] >= start_time) & (index_df['T0_Sync'] <= end_time), 'Label'] = 1
        if print_on:
            print(index_df)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(index_filepath), 'index_label')
            index_df.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        logging.info('Label set complete.')
        return index_df

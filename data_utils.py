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
        self.columnInfo = ['Source', 'Destination', 'ID&Fragment', 'Protocol', 'Length', 'Retry', 'Sniff_Timestamp']
        self.columnIndex = ['Source', 'Destination', 'ID&Fragment', 'Protocol',
                            'T0_Sync', 'T0_Local', 'T1_Sync', 'T1_Local', 'T2_Sync', 'T2_Local',
                            'T1_T0_Sync', 'T1_T0_Local', 'T2_T1_Sync', 'T2_T1_Local', 'T0_100', 'Length', 'Retry']
        self.captureFilter = 'ip.dst == 192.168.137.160 && tcp'

    def change_unit(self, filepath: str, column: list, coefficient: float):
        data = pd.read_csv(filepath_or_buffer=filepath)
        data[column] = data[column] * coefficient
        output_path, ext = os.path.splitext(filepath)
        data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data

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

    def combine_index(self, dir_ap_log: str, dir_pcap: str, print_on=False, to_csv_on=True):
        """
        合并ap端log提取指标和空口数据提取指标

        :param dir_ap_log:
        :param dir_pcap:
        :param print_on:
        :param to_csv_on:
        :return:
        """
        logging.info('Index combine start.')
        path_ap_index = os.path.join(dir_ap_log, 'index_tall')
        path_pcap_index = os.path.join(dir_pcap, 'capture_info')
        ap_index_arr = pd.read_csv(filepath_or_buffer=path_ap_index).values.tolist()
        pcap_index_df = pd.read_csv(filepath_or_buffer=path_pcap_index)
        pcap_index_df['Read'] = pcap_index_df['Retry'] * 0
        pcap_index_df['Index'] = pcap_index_df['Retry'] * 0
        pcap_index_arr = pcap_index_df.values.tolist()

        nSniffTimestamp = 6
        nRead = 7
        nIndex = 8
        len_ap_index = len(ap_index_arr)
        len_pcap_index = len(pcap_index_arr)

        column = self.columnIndex
        obj_curr = [None] * len(column)
        index_res = [obj_curr]

        start = 0
        while not (ap_index_arr[0][0:4] == pcap_index_arr[start][0:4]):
            start += 1

        for i in bar.progressbar(range(0, len_ap_index)):
            obj_curr[0:15] = ap_index_arr[i][0:15]
            flag = True
            for j in range(start, len_pcap_index):
                if pcap_index_arr[j][nRead] == 0:
                    if flag:
                        start = j
                        flag = False
                    if ap_index_arr[i][0:4] == pcap_index_arr[j][0:4]:
                        if not -15 < pcap_index_arr[j][nSniffTimestamp] - ap_index_arr[i][4] < 15:
                            obj_curr = [None] * len(column)
                            break
                        obj_curr[15:17] = pcap_index_arr[j][4:6]
                        pcap_index_arr[j][nRead] = 1
                        pcap_index_arr[j][nIndex] = i
                        dj = 0
                        while dj < 20:
                            dj += 1
                            if pcap_index_arr[j+dj][nRead] == 0 and (ap_index_arr[i][0:4] == pcap_index_arr[j+dj][0:4]):
                                obj_curr[16] += pcap_index_arr[j+dj][5]
                                pcap_index_arr[j+dj][nRead] = 1
                                pcap_index_arr[j+dj][nIndex] = i
                                j = j + dj
                                dj = 0
                        index_res.append(obj_curr)
                        obj_curr = [None] * len(column)
                        break
        index_df = pd.DataFrame(data=index_res, columns=column)
        pcap_index_df = pd.DataFrame(data=pcap_index_arr)

        if print_on:
            print(index_df)
        if to_csv_on:
            index_df.to_csv(path_or_buf=os.path.join(dir_ap_log, 'index'), index=False, float_format="%.6f")
            pcap_index_df.to_csv(path_or_buf=os.path.join(dir_pcap, 'capture_info_read'), index=False)
        logging.info('Index combine complete.')
        return index_df

    def combine_router_timestamp(self, filepath_list: list, print_on=False, to_csv_on=True):
        """
        合并 路由器时间戳数据T0&T1&T2

        :param filepath_list: T0,T1,T2文件路径列表
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return: 整体路由器时间戳数据
        """
        logging.info('Data merging...')
        if len(filepath_list) != 3:
            logging.error('Wrong number(%d) of lists. The number of target lists is 3.', len(filepath_list))
            return
        log0 = self.import_router_timestamp(filepath=filepath_list[0])
        log1 = self.import_router_timestamp(filepath=filepath_list[1])
        log2 = pd.concat(objs=[log0, log1])
        log1 = self.import_router_timestamp(filepath=filepath_list[2])
        log0 = pd.concat(objs=[log2, log1])
        log0.sort_values(by=['Time_Sync', 'Time_Local'], inplace=True)
        if print_on:
            print(log0)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(filepath_list[0]), 'log_tall')
            log0.to_csv(path_or_buf=output_path, index=False)
        return log0

    def delete_null_data(self, data: pd.DataFrame, output_path=None, print_on=False, to_csv_on=False):
        """
        删除空数据所在行

        :param data: 数据
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return: 处理后数据
        """
        # data = pd.read_csv(filepath_or_buffer=filepath)
        null_list = data[data.isna().T.any()].index.tolist()
        data = data.drop(null_list)
        if print_on:
            print(data)
        if to_csv_on:
            # output_path = os.path.join(os.path.dirname(filepath), os.path.basename(filepath) + '_nonnull')
            data.to_csv(path_or_buf=output_path, index=False)
        return data

    def extract_ap_index(self, dir_ap_log: str, print_on=False, to_csv_on=True):
        """
        提取AP指标

        :param dir_ap_log:
        :param print_on:
        :param to_csv_on:
        :return:
        """
        logging.info('Index extracting...')
        path_ap_log = os.path.join(dir_ap_log, 'log_tall')

        rt_tsp = pd.read_csv(filepath_or_buffer=path_ap_log)
        # 提取指标T1-T0, T2-T1
        result = self.extract_delta_t(rt_tsp=rt_tsp)
        # 空值填充处理
        result = self.fill_null_index(index_df=result)
        # 删除空数据所在行
        result = self.delete_null_data(data=result)
        # 筛选TCP数据
        result = self.filter(filepath_or_dataframe=result, key='Protocol', value=6)
        # 提取指标T2_T0_Sync
        logging.info('Extract index: T2-T0')
        result.insert(14, 'T2_T0_Sync', result['T1_T0_Sync'] + result['T2_T1_Sync'])
        # 提取指标T0_100, T0_200, T0_500, T0_1000, T0_2000
        result = self.extract_tx_n(data=result, insert_i=15, column_name='T0_100', index_name='T0_Sync', index_i=4, numstep=100, delete_on=False)
        result = self.extract_tx_n(data=result, insert_i=16, column_name='T0_200', index_name='T0_Sync', index_i=4, numstep=200, delete_on=False)
        result = self.extract_tx_n(data=result, insert_i=17, column_name='T0_500', index_name='T0_Sync', index_i=4, numstep=500, delete_on=False)
        result = self.extract_tx_n(data=result, insert_i=18, column_name='T0_1000', index_name='T0_Sync', index_i=4, numstep=1000, delete_on=False)
        result = self.extract_tx_n(data=result, insert_i=19, column_name='T0_2000', index_name='T0_Sync', index_i=4, numstep=2000)

        if print_on:
            print(result)
        if to_csv_on:
            result.to_csv(path_or_buf=os.path.join(dir_ap_log, 'index_tall'), index=False)
        return result

    def extract_capture_info(self, filepath: str, print_on=False, to_csv_on=True):
        """
        提取抓包信息

        :param filepath:
        :param print_on:
        :param to_csv_on:
        :return:
        """
        # logging.info('Capture info extract start.')
        column = self.columnInfo
        capture = self.import_capture(filepath=filepath, keep_packets=False)
        pgb = bar.ProgressBar()
        info_list = []
        i = [0]

        def ip2int(ip: str):
            return sum([256 ** m * int(n) for m, n in enumerate(ip.split('.')[::1])])

        def combine_id_flags(ipid: str, flags: str):
            return '0x' + ipid[-4:] + flags[-2:] + flags[-4:-2]

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
                    if 'ip' in subframe:
                        info = [ip2int(subframe['ip']['ip.src']),
                                ip2int(subframe['ip']['ip.dst']),
                                combine_id_flags(subframe['ip']['ip.id'], subframe['ip']['ip.flags']),
                                subframe['ip']['ip.proto'],
                                subframe['wlan_aggregate.a_mdsu.length'],
                                packet.wlan.fc_tree.flags_tree.retry,
                                packet.sniff_timestamp]
                        info_list.append(info)
            i[0] += 1
            pgb.update(i[0])

        capture.apply_on_packets(get_packet_info)
        if print_on:
            print(info_list)
        if to_csv_on:
            info_df = pd.DataFrame(data=info_list, columns=column)
            info_df.to_csv(path_or_buf=os.path.join(os.path.dirname(filepath), 'capture_info'), index=False)
        # logging.info('Capture info extract complete.')
        return info_list

    def extract_delta_t(self, rt_tsp: pd.DataFrame, output_path=None, print_on=False, to_csv_on=False):
        """
        提取指标：T1-T0, T2-T1

        :param rt_tsp: 路由器时间戳数据
        :param output_path: 输出csv文件路径
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return: 指标列表
        """
        logging.info('Extract index: T1-T0, T2-T1')
        column = self.columnIndex[0:14]
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
                                # if obj_curr[14] is None:
                                #     obj_curr[14] = 1
                                # else:
                                #     obj_curr[14] = obj_curr[14] + 1
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
                # obj_curr[14] = 0
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
                logging.error('Invalid parameter: extract_delta_t->output_filepath=None')
            else:
                index_df.to_csv(path_or_buf=output_path, index=False)
        return index_df

    def extract_t0_100(self, data: pd.DataFrame, output_path=None, print_on=False, to_csv_on=False):
        """
        提取指标T1_100

        :param data:
        :param output_path:
        :param print_on:
        :param to_csv_on:
        :return:
        """
        column = self.columnIndex[0:15]
        column_name = self.columnIndex[14]
        data[column_name] = data['T0_Sync'] * 0
        len_data = len(data)
        data_arr = data.values

        for i in range(100, len_data):
            data_arr[i, 14] = data_arr[i, 4] - data_arr[i - 100, 4]
        data_arr = np.delete(arr=data_arr, obj=np.s_[0:100], axis=0)
        data = pd.DataFrame(data=data_arr, columns=column)

        if print_on:
            print(data)
        if to_csv_on:
            data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data

    def extract_tx_n(self,
                     data: pd.DataFrame,
                     insert_i: int,
                     column_name: str,
                     index_name: str,
                     index_i: int,
                     numstep: int,
                     output_path=None,
                     delete_on=True,
                     print_on=False,
                     to_csv_on=False):
        """
        提取指标Tx_n

        :param data: 数据
        :param insert_i: 提取特征的插入位置
        :param column_name: 提取特征的名称
        :param index_name: 提取特征的源特征Tx名称
        :param index_i: 提取特征的源特征位置
        :param numstep: 步进值n
        :param output_path: 结果保存路径
        :param delete_on: 是否删除前n项
        :param print_on: 是否在控制台打印
        :param to_csv_on: 是否输出为csv
        :return:
        """
        logging.info('Extract index: ' + column_name)
        data.insert(insert_i, column_name, '')
        column = list(data)
        len_data = len(data)
        data_arr = data.values

        for i in range(numstep, len_data):
            data_arr[i, insert_i] = data_arr[i, index_i] - data_arr[i - numstep, index_i]
        if delete_on:
            data_arr = np.delete(arr=data_arr, obj=np.s_[0:numstep], axis=0)
        data = pd.DataFrame(data=data_arr, columns=column)

        if print_on:
            print(data)
        if to_csv_on:
            data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data

    def fill_null_index(self, index_df: pd.DataFrame, output_path=None, print_on=False, to_csv_on=False):
        """
        空值填充处理

        :param index_df: 指标数据文件路径
        :param output_path: 保存路径
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return: 处理后数据
        """
        column = self.columnIndex[0:14]
        # index_df = pd.read_csv(filepath_or_buffer=filepath)
        len_index = len(index_df)
        index_arr = index_df.values
        # 处理T0空值
        for i in range(0, len_index):
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
            # if pd.isna(index_arr[i, 14]):
            #     index_arr[i, 14] = 0
        index_fill_df = pd.DataFrame(data=index_arr, columns=column)
        if print_on:
            print(index_fill_df)
        if to_csv_on:
            # output_path = os.path.join(os.path.dirname(filepath), os.path.basename(filepath) + '_fill')
            index_fill_df.to_csv(path_or_buf=output_path, index=False)
        return index_fill_df

    def filter(self, filepath_or_dataframe, key: str, value, output_path=None, print_on=False, to_csv_on=False):
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
            data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
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
                index_df.loc[((index_df['T1_Sync'] + index_df['T2_Sync']) / 2 >= start_time) & (
                            (index_df['T1_Sync'] + index_df['T2_Sync']) / 2 <= end_time), 'Label'] = 1
            elif row['Type'] == 'whole':
                start_time = row['Start_Time']
                end_time = row['End_Time']
                index_df = index_df[((index_df['T1_Sync'] + index_df['T2_Sync']) / 2 >= start_time) & (
                            (index_df['T1_Sync'] + index_df['T2_Sync']) / 2 <= end_time)]
        if print_on:
            print(index_df)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(index_filepath), 'index_label')
            index_df.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        logging.info('Label set complete.')
        return index_df

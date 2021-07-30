#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: main.py
# Time: 2021/05
# Author: Zhu Yutao
# Description: 主函数入口

import data_combine as dc
import data_analysis as da
import data_preprocess as dp
from data_utils import DataUtils
import index_extract as ie
import logging
import model
import numpy as np
import os
import pandas as pd
import program_logging
import progressbar as bar
import test
import test_bilstm_crf as tbc
import test_model as tm
import torch


def combine_router_timestamp(ap_log_dir_list: list):
    """
    批量合成路由器时间戳原始数据

    :param ap_log_dir_list: AP log路径列表
    :return:
    """
    data_utils = DataUtils()
    for i, ap_log_dir in enumerate(ap_log_dir_list):
        logging.info('Router timestamp combine: %d', i + 1)
        if os.path.exists(ap_log_dir):
            filepath0 = os.path.join(ap_log_dir, 'log_t0')
            filepath1 = os.path.join(ap_log_dir, 'log_t1')
            filepath2 = os.path.join(ap_log_dir, 'log_t2')
            data_utils.combine_router_timestamp(filepath_list=[filepath0, filepath1, filepath2])
        else:
            logging.error('Error directory: ' + ap_log_dir)
    logging.info('Router timestamp combine complete.')
    return


def extract_capture_info(option: str, start_end: list):
    """
    批量提取抓包信息

    :param option: 选项：video/game
    :param start_end: 起止序号
    :return:
    """
    if option == 'video':
        file_dir = 'data/data_video'
    elif option == 'game':
        file_dir = 'data/data_game'
    else:
        logging.error('Invalid parameter value: option=%s', option)
        return
    if len(start_end) != 2:
        logging.error('Invalid parameter value: len(start_end)=%d', len(start_end))
        return

    def pcap_filter(fp):
        return fp[-5:] == '.pcap'

    data_utils = DataUtils()
    for i in range(start_end[0], start_end[1]):
        logging.info('Capture info extract: %s (%d/%d)', option, i + 1 - start_end[0], start_end[1] - start_end[0])
        pcap_dir = os.path.join(file_dir, 'data_' + option + '_' + str(i))
        file_list = os.listdir(pcap_dir)
        pcap_name = list(filter(pcap_filter, file_list))[0]
        pcap_path = os.path.join(pcap_dir, pcap_name)
        data_utils.extract_capture_info(filepath=pcap_path)

    logging.info('Capture info extract complete.')
    return


def extract_index(ap_log_dir_list: list):
    """
    批量提取特征

    :param ap_log_dir_list: AP log路径列表
    :return:
    """
    data_utils = DataUtils()
    for i, ap_log_dir in enumerate(ap_log_dir_list):
        logging.info('Index extract: %d', i + 1)
        path_tall = os.path.join(ap_log_dir, 'log_tall')
        output_path = os.path.join(ap_log_dir, 'index_all')
        rt_tsp = pd.read_csv(filepath_or_buffer=path_tall)
        data_utils.extract_index(rt_tsp=rt_tsp, output_path=output_path, to_csv_on=True)
        path_index = os.path.join(ap_log_dir, 'index_all')
        data_utils.fill_null_index(filepath=path_index)
        path_fill = os.path.join(ap_log_dir, 'index_all_fill')
        data_utils.delete_null_data(filepath=path_fill)
    logging.info('Index extract complete.')
    return


def get_path():
    """
    获取路径

    :return:
    """
    ap_log_dir_list = []
    bug_time_dir_list = []
    type = 'video'
    num = 1
    root = 'D:/Project/PyCharm/huawei_kqi/data/data_20210723'
    for idx in range(1, 6):
        type_idx = type + '_' + str(idx)
        type_idx_num = type_idx + '_' + str(num)
        ap_log_dir = root + '/data_' + type + '/data_' + type_idx + '/data_' + type_idx_num + '/ap_log_' + type_idx_num
        ap_log_dir_list.append(ap_log_dir)
        bug_time_dir = root + '/data_' + type + '/data_' + type_idx + '/data_' + type_idx_num + '/screenrecord_' + type_idx_num
        bug_time_dir_list.append(bug_time_dir)
    return ap_log_dir_list, bug_time_dir_list


def set_label(ap_log_dir_list: list, bug_time_dir_list: list):
    """
    标注

    :param ap_log_dir_list: AP log路径列表
    :param bug_time_dir_list: 录屏log异常时间标注文件路径列表
    :return:
    """
    data_utils = DataUtils()
    for i, ap_log_dir in enumerate(ap_log_dir_list):
        logging.info('Label set: %d', i + 1)
        index_path = os.path.join(ap_log_dir, 'index_all_fill_nonnull')
        bug_path = os.path.join(bug_time_dir_list[i], 'log_bug_timestamp.txt')
        data_utils.set_label(index_filepath=index_path, bug_filepath=bug_path)
    logging.info('Index extract complete.')
    return


if __name__ == '__main__':
    # 获取路径
    ap_log_dir_list, bug_time_dir_list = get_path()
    # # 合并AP端log
    # combine_router_timestamp(ap_log_dir_list)
    # # 从AP端log提取特征
    # extract_index(ap_log_dir_list)
    # 异常数据标注
    set_label(ap_log_dir_list, bug_time_dir_list)

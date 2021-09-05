#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: main.py
# Time: 2021/05
# Author: Zhu Yutao
# Description: 主函数入口

# import data_combine as dc
# import data_analysis as da
# import data_preprocess as dp
import logging

from data_utils import DataUtils
# import index_extract as ie
# import logging
# import model
import numpy as np
import os
# import pandas as pd
# import program_logging
# import progressbar as bar
# import svm
import test
# import test_bilstm_crf as tbc
# import test_model as tm
# import torch


# def change_bug_time_unit(bug_time_dir_list: list):
#     data_utils = DataUtils()
#     for i, bug_time_dir in enumerate(bug_time_dir_list):
#         logging.info('Bug time unit change: %d', i + 1)
#         bug_time_path = os.path.join(bug_time_dir, 'log_bug_timestamp.txt')
#         data_utils.change_unit(filepath=bug_time_path, column=['Start_Time', 'End_Time'], coefficient=0.001)
#     logging.info('Bug time unit change complete.')
#
#
def get_path():
    """
    获取路径

    :return:
    """
    ap_log_dir_list = []
    bug_time_dir_list = []
    pcap_path_list = []
    type = 'video'
    num = 2
    root = 'data/data_20210831'
    i_list = [1, 2, 3, 5, 6]
    for idx in i_list:
        type_idx = type + '_' + str(idx)
        type_idx_num = type_idx + '_' + str(num)
        ap_log_dir = root + '/data_' + type + '/data_' + type_idx + '/data_' + type_idx_num + '/ap_log_' + type_idx_num
        ap_log_dir_list.append(ap_log_dir)
        bug_time_dir = root + '/data_' + type + '/data_' + type_idx + '/data_' + type_idx_num + '/ScreenRecord_' + type_idx_num
        bug_time_dir_list.append(bug_time_dir)
        pcap_path = root + '/data_' + type + '/data_' + type_idx + '/data_' + type_idx_num + '/pcap_wifi_' + type_idx_num + '.pcap'
        pcap_path_list.append(pcap_path)
    return ap_log_dir_list, bug_time_dir_list, pcap_path_list


def data_processing():
    """

    :return:
    """
    data_utils = DataUtils()

    # 获取路径
    ap_log_dir_list, bug_time_dir_list, pcap_path_list = get_path()

    for i in range(0, len(ap_log_dir_list)):
        logging.info('Data processing...(%d/%d)', i, len(ap_log_dir_list))
        ap_log_dir = ap_log_dir_list[i]
        bug_time_dir = bug_time_dir_list[i]
        pcap_path = pcap_path_list[i]

        # 合并AP端log  -->  log_tall
        ap_path_list = [ap_log_dir + '/log_t0',
                        ap_log_dir + '/log_t1',
                        ap_log_dir + '/log_t2']
        data_utils.combine_router_timestamp(filepath_list=ap_path_list)

        # 提取AP特征信息  -->  index_tall
        data_utils.extract_ap_index(dir_ap_log=ap_log_dir)

        # # 提取抓包特征信息  -->  capture_info
        # data_utils.extract_capture_info(filepath=pcap_path)

        # # 合并特征信息  -->  index, capture_info_read
        # data_utils.combine_index(dir_ap_log=ap_log_dir,
        #                          dir_pcap=os.path.dirname(pcap_path)

        # # 异常数据标注  -->  index_label
        # data_utils.set_label(index_filepath=os.path.join(ap_log_dir, 'index'),
        #                      bug_filepath=os.path.join(bug_time_dir, 'log_bug_timestamp'))

    # # 异常时间标注单位换算  -->  log_bug_timestamp
    # change_bug_time_unit(bug_time_dir_list)


if __name__ == '__main__':
    # 数据处理
    data_processing()

    # # 训练
    # svm.test_svm()

    # # 测试
    # test.test_time_label()

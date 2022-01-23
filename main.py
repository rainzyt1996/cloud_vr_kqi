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

import model
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
import sys
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
    pcap_wifi_path_list = []
    pcap_wan_path_list = []
    root = 'data/data_video'
    dtype = 'v'     # 样本类型
    if len(sys.argv) >= 4:
        objs = [int(sys.argv[1])]
        envs = [int(sys.argv[2])]
        idxs = [int(sys.argv[3])]
    else:
        objs = [1]  # 对象编号: 1, 5, 6, 7, 8, 9, 10, 11, 12, 13
        envs = [1]  # 环境编号: 0, 1, 2, 3
        idxs = [1]  # 样本编号: 1, 2
    for obj in objs:
        for env in envs:
            for idx in idxs:
                name_to = dtype + str(obj)
                name_toe = name_to + '_' + str(env)
                name_toei = name_toe + '_' + str(idx)
                dirname = root + '/' + name_to + '/' + name_toe + '/' + name_toei
                ap_log_dir_list.append(dirname + '/ap_log_' + name_toei)
                bug_time_dir_list.append(dirname + '/ScreenRecorder')
                pcap_wifi_path_list.append(dirname + '/pcap_wifi_' + name_toei + '.pcap')
                pcap_wan_path_list.append(dirname + '/pcap_wan_' + name_toei + '.pcap')
    return ap_log_dir_list, bug_time_dir_list, pcap_wifi_path_list, pcap_wan_path_list


def data_processing():
    """

    :return:
    """
    data_utils = DataUtils()

    # 获取路径
    ap_log_dir_list, bug_time_dir_list, pcap_wifi_path_list, pcap_wan_path_list = get_path()

    if len(sys.argv) > 4:
        process_list = list(map(int, sys.argv[4:]))
    else:
        process_list = [1, 2, 5, 6, 7, 8]

    for i in range(len(ap_log_dir_list)):
        logging.info('Data processing...(%d/%d)', i, len(ap_log_dir_list))

        ap_log_dir = ap_log_dir_list[i]
        bug_time_dir = bug_time_dir_list[i]
        pcap_wifi_path = pcap_wifi_path_list[i]
        pcap_wan_path = pcap_wan_path_list[i]

        for process in process_list:
            if int(process) == 1:
                # 合并AP端log  -->  log_tall
                ap_path_list = [ap_log_dir + '/log_t0',
                                ap_log_dir + '/log_t1',
                                ap_log_dir + '/log_t2']
                data_utils.combine_ap_timestamp(filepath_list=ap_path_list)
            if int(process) == 2:
                # 提取AP端单包数据  -->  rawdata_ap
                data_utils.extract_ap_pkt_data(path_ap_log_all=os.path.join(ap_log_dir, 'log_tall'))
            if int(process) == 3:
                # 提取wifi抓包特征  -->  capture_info
                output_path = os.path.join(os.path.dirname(pcap_wifi_path), 'capture_info')
                data_utils.extract_cap_info(filepath=pcap_wifi_path, output_path=output_path)
            if int(process) == 4:
                # 提取wan抓包特征  -->  capture_wan_info  (暂不支持)
                output_path = os.path.join(os.path.dirname(pcap_wan_path), 'capture_wan_info')
                data_utils.extract_cap_info(filepath=pcap_wan_path, output_path=output_path)
            if int(process) == 5:
                # 合并单包数据  -->  rawdata, capture_info_read
                data_utils.combine_pkt_data(path_ap_pkt_data=os.path.join(ap_log_dir, 'rawdata_ap'),
                                            path_cap_info=os.path.join(os.path.dirname(pcap_wifi_path), 'capture_info'))
            if int(process) == 6:
                # 数据预处理  -->  data_pkt
                data_utils.preprocessing(path_data=os.path.join(ap_log_dir, 'rawdata'))
            if int(process) == 7:
                # 提取特征  -->  index_pkt, index
                data_utils.extract_index(path_data=os.path.join(ap_log_dir, 'data_pkt'))
            if int(process) == 8:
                # 数据标注  -->  index_label
                data_utils.set_label(path_index=os.path.join(ap_log_dir, 'index'),
                                     path_timestamp=os.path.join(bug_time_dir, 'result_detection_timestamp.txt'))


if __name__ == '__main__':
    # 数据处理
    data_processing()

    # 训练
    # svm.test_svm()
    # model.train()
    # model.get_realdata(filepath='data/data_video/v1/v1_1/v1_1_1/ap_log_v1_1_1/index_label', batch_size=100)

    # 测试
    # test.test()

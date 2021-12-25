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
from data_utils import DataUtils
# import index_extract as ie
import logging
# import model
# import numpy as np
import os
# import pandas as pd
# import program_logging
# import progressbar as bar
import svm
# import test
# import test_bilstm_crf as tbc
# import test_model as tm
# import torch
# import tsne


# def change_bug_time_unit(bug_time_dir_list: list):
#     data_utils = DataUtils()
#     for i, bug_time_dir in enumerate(bug_time_dir_list):
#         logging.info('Bug time unit change: %d', i + 1)
#         bug_time_path = os.path.join(bug_time_dir, 'log_bug_timestamp.txt')
#         data_utils.change_unit(filepath=bug_time_path, column=['Start_Time', 'End_Time'], coefficient=0.001)
#     logging.info('Bug time unit change complete.')


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
    objs = [1]  # 对象编号: 1, 5, 6, 7, 8, 9, 10, 11, 12, 13
    envs = [1]  # 环境编号: 1, 2, 3
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

    for i in range(len(ap_log_dir_list)):
        logging.info('Data processing...(%d/%d)', i, len(ap_log_dir_list))
        ap_log_dir = ap_log_dir_list[i]
        bug_time_dir = bug_time_dir_list[i]
        pcap_wifi_path = pcap_wifi_path_list[i]
        pcap_wan_path = pcap_wan_path_list[i]

        # # 合并AP端log  -->  log_tall
        # ap_path_list = [ap_log_dir + '/log_t0',
        #                 ap_log_dir + '/log_t1',
        #                 ap_log_dir + '/log_t2']
        # data_utils.combine_ap_timestamp(filepath_list=ap_path_list)

        # # 提取AP端单包数据  -->  rawdata_ap
        # data_utils.extract_ap_pkt_data(path_ap_log_all=os.path.join(ap_log_dir, 'log_tall'))

        # # 提取wifi抓包特征  -->  capture_info
        # output_path = os.path.join(os.path.dirname(pcap_wifi_path), 'capture_info')
        # data_utils.extract_capture_info(filepath=pcap_wifi_path, output_path=output_path)

        # # 提取wan抓包特征  -->  capture_wan_info  (暂不支持)
        # output_path = os.path.join(os.path.dirname(pcap_wan_path), 'capture_wan_info')
        # data_utils.extract_capture_info(filepath=pcap_wan_path, output_path=output_path)

        # 合并单包数据  -->  rawdata
        data_utils.combine_pkt_data(path_ap_pkt_data=os.path.join(ap_log_dir, 'rawdata_ap'),
                                    path_cap_info=os.path.join(os.path.dirname(pcap_wifi_path), 'capture_info'))

        # # 合并单包特征  -->  index_pkg, capture_info_read
        # data_utils.combine_index(dir_ap_log=ap_log_dir,
        #                          dir_pcap=os.path.dirname(pcap_wifi_path))

        # 提取区间特征  -->  index

        # # 异常数据标注  -->  index_label
        # data_utils.set_label(path_index=os.path.join(ap_log_dir, 'index'),
        #                      path_timestamp=os.path.join(bug_time_dir, 'result_detection_timestamp.txt'))

    # # 异常时间标注单位换算  -->  log_bug_timestamp
    # change_bug_time_unit(bug_time_dir_list)


if __name__ == '__main__':
    data_utils = DataUtils()

    # 获取路径
    ap_log_dir_list, bug_time_dir_list, pcap_wifi_path_list, pcap_wan_path_list = get_path()

    # # 数据处理
    # data_processing()

    # # t-SNE
    # tsne.test_tsne()

    # 训练
    ap_log_dir = ap_log_dir_list[0]
    path_data = os.path.join(ap_log_dir, 'index_label')
    index_list = data_utils.cIndex
    svm.test_svm(path_data=path_data, index_list=index_list)

    # # 测试
    # print('test')


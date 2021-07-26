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


def combine_router_timestamp(option: str, start_end: list):
    """
    批量合成路由器时间戳原始数据

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
    data_utils = DataUtils()
    for i in range(start_end[0], start_end[1]):
        logging.info('Router timestamp combine: %s (%d/%d)', option, i + 1 - start_end[0], start_end[1] - start_end[0])
        filepath0 = os.path.join(os.path.join(file_dir, 'data_' + option + '_' + str(i)), 'log/log_t0')
        filepath1 = os.path.join(os.path.join(file_dir, 'data_' + option + '_' + str(i)), 'log/log_t1')
        filepath2 = os.path.join(os.path.join(file_dir, 'data_' + option + '_' + str(i)), 'log/log_t2')
        data_utils.combine_router_timestamp(filepath_list=[filepath0, filepath1, filepath2])
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


def extract_index(option: str, start_end: list):
    """
    批量提取特征

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
    data_utils = DataUtils()
    for i in range(start_end[0], start_end[1]):
        logging.info('Index extract: %s (%d/%d)', option, i + 1 - start_end[0], start_end[1] - start_end[0])
        path_tall = os.path.join(os.path.join(file_dir, 'data_' + option + '_' + str(i)), 'log/log_tall')
        output_path = os.path.join(os.path.join(file_dir, 'data_' + option + '_' + str(i)), 'log/index_all')
        rt_tsp = pd.read_csv(filepath_or_buffer=path_tall)
        data_utils.extract_index(rt_tsp=rt_tsp, output_path=output_path, to_csv_on=True)
        path_index = os.path.join(os.path.join(file_dir, 'data_' + option + '_' + str(i)), 'log/index_all')
        data_utils.fill_null_index(filepath=path_index)
        path_fill = os.path.join(os.path.join(file_dir, 'data_' + option + '_' + str(i)), 'log/index_all_fill')
        data_utils.delete_null_data(filepath=path_fill)
        index_path = os.path.join(os.path.join(file_dir, 'data_' + option + '_' + str(i)), 'log/index_all_fill_nonnull')
        bug_path = os.path.join(os.path.join(file_dir, 'data_' + option + '_' + str(i)), 'log_bug_timestamp')
        data_utils.set_label(index_filepath=index_path, bug_filepath=bug_path)
    logging.info('Index extract complete.')
    return


if __name__ == '__main__':
    # combine_router_timestamp('video', [1, 4])
    # extract_index('video', [1, 4])
    extract_capture_info('video', [1, 4])

    # train_filepath_1 = 'data/data_video/data_video_1/log/index_label'
    # data, label = model.data_preprocess(filepath=train_filepath_1, time_step=100, batch_size=16, delta=60)
    # print(data.shape, label.shape)

    # model.train()
    # test.test_extract_capture_info()

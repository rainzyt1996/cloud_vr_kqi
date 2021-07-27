#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: data_import.py
# Time: 2021/5
# Author: Zhu Yutao
# Description: 数据导入
import logging

import numpy as np
import pandas as pd
from pyshark import FileCapture

import program_logging


def import_timestamp(filepath: str, print_on=False):
    """
    路由器时间戳log数据导入

    :param filepath: log数据文件路径
    :param print_on: 打印至控制台开关
    :return: 时间戳log数据表
    """
    log = pd.read_csv(filepath_or_buffer=filepath,
                      header=None,
                      names=['Tn', 'Source', 'Destination', 'ID&Fragment', 'Protocol', 'Time_Sync', 'Time_Local'])
    if print_on:
        print(log)

    return log


def import_capture(filepath: str):
    """
    空口数据导入
    """
    capture = FileCapture(input_file=filepath,
                          display_filter='ip.dst == 192.168.200.160',
                          keep_packets=False)
    length = 0
    for packet in capture:
        length += 1
    print(length)

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: program_logging.py
# Time: 2021/05
# Author: Zhu Yutao
# Description: 配置程序日志


import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s (%(levelname)s): %(message)s')
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s (%(levelname)s): [%(filename)s(line:%(lineno)d)] %(message)s')

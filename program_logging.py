#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: program_logging.py
# Time: 2021/05
# Author: Zhu Yutao
# Description: 配置程序日志


import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d]: (%(levelname)s) %(message)s')

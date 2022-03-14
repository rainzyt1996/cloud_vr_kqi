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

    columnApTimestamp = ['Tn', 'Source', 'Destination', 'ID&Fragment', 'Protocol', 'Time_Sync', 'Time_Local']
    columnPacketID = ['Source', 'Destination', 'ID&Fragment', 'Protocol']
    columnApIndexList = ['T0_Sync', 'T0_Local', 'T1_Sync', 'T1_Local', 'T2_Sync', 'T2_Local',
                         'T1_T0_Sync', 'T1_T0_Local', 'T2_T1_Sync', 'T2_T1_Local', 'T2_T0_Sync',
                         'T0_100', 'T0_200', 'T0_500', 'T0_1000', 'T0_2000',
                         'Retry_ID']
    columnApIndex = columnPacketID + columnApIndexList
    columnPcapIndexList = ['Length', 'Retry']
    columnPcapIndex = columnPacketID + columnPcapIndexList
    columnPcapInfo = columnPcapIndex + ['Sniff_Timestamp']
    columnIndex = columnPacketID + columnApIndexList + columnPcapIndexList
    captureFilter = 'ip.dst == 192.168.137.160 && tcp'
    nPacketID = len(columnPacketID)     # 4
    nApIndex = len(columnApIndex)       # 21
    nPcapIndex = len(columnPcapIndex)   # 6
    nIndex = len(columnIndex)           # 23

    def __init__(self):
        """
        初始化
        """
        # 数据项
        self.cPktId = ['Source', 'Destination', 'ID&Fragment', 'Protocol']    # 包标识
        self.cApTimestamp = ['T0', 'T1', 'T2']    # AP端时间戳
        self.cStmData = ['Number', 'SumLength', 'TimeInterval']     # 流统计数据
        # 特征项
        self.cApPktFeature = ['T1_T0', 'T2_T1', 'T2_T0']  # AP端单包特征
        self.cCapPktFeature = ['Length', 'Retry']     # 抓包端单包特征
        self.cPktFeature = self.cApPktFeature + self.cCapPktFeature    # 单包特征
        self.region = [100, 200, 500, 1000, 2000]  # 区间值
        self.cRgnSpnFeature = []  # 区间跨度特征
        self.cRgnSttFeature = []  # 区间统计特征
        for rgn in self.region:
            for item in self.cApTimestamp:
                self.cRgnSpnFeature.append(item + '_' + str(rgn))
            for item in self.cPktFeature:
                self.cRgnSttFeature.append(item + '_Avg_' + str(rgn))
                self.cRgnSttFeature.append(item + '_Std_' + str(rgn))
        self.cRgnFeature = self.cRgnSpnFeature + self.cRgnSttFeature  # 区间特征
        self.cLocalFeature = self.cPktFeature + self.cRgnFeature   # 局部特征
        self.cStmFeature = ['PacketRate', 'DataRate']  # 流特征
        self.cFeature = self.cLocalFeature + self.cStmFeature   # 特征
        # 文件信息项
        self.cFileApLog = ['Tn'] + self.cPktId + ['Time_Sync', 'Time_Local']   # AP端时间戳Log
        self.cFileCapInfo = self.cPktId + self.cCapPktFeature + ['Sniff_Timestamp']    # 抓包端信息
        self.cFileWanStmInfo = self.cPktId + ['Length', 'Sniff_Timestamp']     # WAN口流数据信息
        self.cFileApPktData = self.cPktId + self.cApTimestamp    # AP端合并时间戳Log
        self.cFilePktData = self.cFileApPktData + self.cCapPktFeature  # 单包数据
        self.cFilePktFeature = self.cFilePktData + self.cApPktFeature     # 单包特征
        self.cFileLcFeature = self.cFilePktFeature + self.cRgnFeature   # 局部特征
        self.cFileLcFeatLabel = self.cFileLcFeature + ['Label']    # 带标签的局部特征
        self.cFileFeature = self.cFileLcFeature + self.cStmFeature  # 特征
        self.cFileFeatLabel = self.cFileFeature + ['Label']     # 带标签的特征
        # 常量

    ################################################################################################################
    # 数据整合
    def combine_ap_timestamp(self, filepath_list: list, print_on=False, to_csv_on=True):
        """
        合并路由器时间戳数据T0&T1&T2

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

    def extract_ap_pkt_data(self, path_ap_log_all: str, print_on=False, to_csv_on=True):
        """
        提取AP端单包数据

        :param path_ap_log_all: AP端时间戳Log文件路径
        :param print_on: 打印与否
        :param to_csv_on: 输出文件与否
        :return: AP端单包数据
        """
        logging.info('AP packet data extracting ...')

        # 变量定义 #######################################################
        column = self.cFileApPktData
        obj_curr = [None] * len(column)
        data_res = [obj_curr]
        data_log_df = pd.read_csv(filepath_or_buffer=path_ap_log_all)
        len_log = len(data_log_df)
        data_log_df['Flag_Read'] = data_log_df['Tn'] * 0
        data_log = data_log_df.values
        iTn = data_log_df.columns.tolist().index('Tn')              # 0
        iSrc = data_log_df.columns.tolist().index('Source')         # 1
        iDst = data_log_df.columns.tolist().index('Destination')    # 2
        iIdf = data_log_df.columns.tolist().index('ID&Fragment')    # 3
        iPrt = data_log_df.columns.tolist().index('Protocol')       # 4
        iTsy = data_log_df.columns.tolist().index('Time_Sync')      # 5
        iTlc = data_log_df.columns.tolist().index('Time_Local')     # 6
        iFlg = data_log_df.columns.tolist().index('Flag_Read')      # 7
        iEnd = iFlg + 1
        oSrc = column.index('Source')       # 0
        oDst = column.index('Destination')  # 1
        oIdf = column.index('ID&Fragment')  # 2
        oPrt = column.index('Protocol')     # 3
        oT0 = column.index('T0')            # 4
        oT1 = column.index('T1')            # 5
        oT2 = column.index('T2')            # 6
        oEnd = oT2 + 1

        # 函数定义 ########################################################
        def get_ap_pkt_data(idx_start: int):
            """
            包标识匹配，获取单包数据

            :param idx_start: 开始索引
            :return:
            """
            # 判断索引是否越界
            if idx_start < len_log:
                # 开始遍历匹配
                for idx_cal in range(idx_start, len_log):
                    # 判断包标识项是否匹配
                    if obj_curr[oSrc] == data_log[idx_cal, iSrc] and obj_curr[oDst] == data_log[idx_cal, iDst] and \
                            obj_curr[oIdf] == data_log[idx_cal, iIdf] and obj_curr[oPrt] == data_log[idx_cal, iPrt]:
                        # 判断当前log行Tn项的值
                        tn = data_log[idx_cal, iTn]
                        if tn == 0:
                            # 判断缓存变量中的T0, T1, T2
                            if obj_curr[oT0:oEnd] == [None] * (oEnd - oT0):
                                obj_curr[oT0] = data_log[idx_cal, iTsy]
                                data_log[idx_cal, iFlg] = 1
                            else:
                                break
                        elif tn == 1:
                            # 判断缓存变量中的T1, T2
                            if obj_curr[oT1:oEnd] == [None] * (oEnd - oT1):
                                obj_curr[oT1] = data_log[idx_cal, iTsy]
                                data_log[idx_cal, iFlg] = 1
                            elif obj_curr[oT1] is not None and obj_curr[oT2] is None:
                                data_log[idx_cal, iFlg] = 1
                            else:
                                break
                        elif tn == 2:
                            # 判断缓存变量中的T2
                            if obj_curr[oT2] is None:
                                obj_curr[oT2] = data_log[idx_cal, iTsy]
                                data_log[idx_cal, iFlg] = 1
                            break
                        else:
                            logging.error('Invalid data: [%d]Tn=%d', idx_cal, tn)
                            break

        # 功能实现 ########################################################
        for idx_ex in bar.progressbar(range(0, len_log)):
            # 判断当前log行是否已读取
            if data_log[idx_ex, iFlg] == 0:
                # 将当前包标识项写入缓存变量
                obj_curr[oSrc] = data_log[idx_ex, iSrc]
                obj_curr[oDst] = data_log[idx_ex, iDst]
                obj_curr[oIdf] = data_log[idx_ex, iIdf]
                obj_curr[oPrt] = data_log[idx_ex, iPrt]
                # 获取当前包数据
                get_ap_pkt_data(idx_start=idx_ex)
                # 将缓存变量写入结果，清空缓存变量
                data_res.append(obj_curr)
                obj_curr = [None] * len(column)
        data_res_df = pd.DataFrame(data=data_res, columns=column)

        # 结果处理 ########################################################
        if print_on:
            print(data_res_df)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(path_ap_log_all), 'rawdata_ap')
            data_res_df.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data_res_df

    def extract_cap_info(self, filepath: str, output_path: str, print_on=False, to_csv_on=True):
        """
        提取抓包信息

        :param filepath:
        :param output_path:
        :param print_on:
        :param to_csv_on:
        :return:
        """
        logging.info('Capture info extracting ...')

        # 变量定义 ################################################################################################
        column = self.columnPcapInfo
        capture = self.import_capture(filepath=filepath, keep_packets=False)
        pgb = bar.ProgressBar()
        info_list = []
        i = [0]

        # 函数定义 ################################################################################################
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

        # 功能实现 ################################################################################################
        capture.apply_on_packets(get_packet_info)

        # 结果处理 ################################################################################################
        print('\n')
        if print_on:
            print(info_list)
        if to_csv_on:
            info_df = pd.DataFrame(data=info_list, columns=column)
            info_df.to_csv(path_or_buf=output_path, index=False)
        return info_list

    def extract_cap_stm_info(self, filepath: str, output_path: str, type_capture='wifi', print_on=False, to_csv_on=True):
        """
        提取抓包业务流信息（HTTP上行报文）

        :param filepath:
        :param output_path:
        :param type_capture:
        :param print_on:
        :param to_csv_on:
        :return:
        """
        logging.info('Capture stream info extracting ...')

        # 变量定义 ################################################################
        capture = FileCapture(input_file=filepath,
                              display_filter='tcp && ip.dst == 192.168.137.160',
                              only_summaries=False,
                              keep_packets=False,
                              use_json=True)
        capture_http = FileCapture(input_file=filepath,
                                   display_filter='http && ip.src == 192.168.137.160',
                                   only_summaries=False,
                                   keep_packets=False,
                                   use_json=True)
        column = []
        info_list = []
        pgb = bar.ProgressBar()
        i = [0]

        # 函数定义 ################################################################
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
                                int(subframe['ip']['ip.proto']),
                                int(subframe['wlan_aggregate.a_mdsu.length']),
                                int(packet.wlan.fc_tree.flags_tree.retry),
                                float(packet.sniff_timestamp)]
                        info_list.append(info)
            i[0] += 1
            pgb.update(i[0])

        def get_wan_packet_info(packet):
            if 'ip' in packet:
                info = [ip2int(packet.ip.src),
                        ip2int(packet.ip.dst),
                        combine_id_flags(packet.ip.id, packet.ip.flags),
                        int(packet.ip.proto),
                        int(packet.length),
                        float(packet.sniff_timestamp)]
                info_list.append(info)
            # else:
            #     info_list.append([-1, -1, -1, -1, -1])
            i[0] += 1
            pgb.update(i[0])

        # 功能实现 ################################################################
        # 提取报文
        if type_capture == 'wifi':
            column = self.cFileCapInfo
            iLen = 4
            iSnt = 6
            capture_http.apply_on_packets(get_packet_info)  # 提取HTTP上行报文
            http_list = info_list
            info_list = []
            capture.apply_on_packets(get_packet_info)       # 提取下行报文
        elif type_capture == 'wan':
            column = self.cFileWanStmInfo
            iLen = 4
            iSnt = 5
            capture_http.apply_on_packets(get_wan_packet_info)  # 提取HTTP上行报文
            http_list = info_list
            info_list = []
            capture.apply_on_packets(get_wan_packet_info)       # 提取下行报文
        else:
            logging.error('Invalid capture type')
            return
        # 筛选流报文
        info_list = pd.DataFrame(data=info_list, columns=column)
        src_flt = info_list['Source'].value_counts().idxmax()
        info_list = self.filter(filepath_or_dataframe=info_list, key='Source', value=src_flt).values.tolist()
        http_list = pd.DataFrame(data=http_list, columns=column)
        http_list = self.filter(filepath_or_dataframe=http_list, key='Destination', value=src_flt).values.tolist()
        df = pd.DataFrame(data=info_list, columns=column)
        df.to_csv(path_or_buf=os.path.join(os.path.dirname(filepath), 'capture_info_wan'), index=False)
        df = pd.DataFrame(data=http_list, columns=column)
        df.to_csv(path_or_buf=os.path.join(os.path.dirname(filepath), 'capture_info_wan_http'), index=False)
        # http_list = pd.read_csv(filepath_or_buffer=os.path.join(os.path.dirname(filepath), 'capture_info_wan_http')).values.tolist()
        # info_list = pd.read_csv(filepath_or_buffer=os.path.join(os.path.dirname(filepath), 'capture_info_wan')).values.tolist()
        # 统计流信息
        column_out = column + ['Number', 'SumLength', 'TimeInterval', 'PacketRate', 'DataRate']
        iInfo = 0
        for i in range(len(info_list)):
            if info_list[i][iSnt] >= http_list[0][iSnt]:
                iInfo = i
                break
        num = 0
        sumLen = 0
        iHttp = 0
        tHttp = http_list[iHttp][iSnt]
        for i in bar.progressbar(range(iInfo, len(info_list))):
            if iHttp >= len(http_list) - 1:
                num += 1
                sumLen += info_list[i][iLen]
            elif info_list[i][iSnt] >= http_list[iHttp + 1][iSnt]:
                num = 1
                sumLen = info_list[i][iLen]
                iHttp += 1
                tHttp = http_list[iHttp][iSnt]
            else:
                num += 1
                sumLen += info_list[i][iLen]
            tInterval = info_list[i][iSnt] - tHttp
            info_list[i].extend([num, sumLen, tInterval, num / tInterval, sumLen / tInterval])

        # 结果处理 ################################################################
        if print_on:
            print(info_list)
        if to_csv_on:
            info_df = pd.DataFrame(data=info_list[iInfo:], columns=column_out)
            info_df.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return info_list

    def combine_pkt_data(self, path_ap_pkt_data: str, path_cap_info: str, print_on=False, to_csv_on=True):
        """
        合并单包数据

        :param path_ap_pkt_data: AP端时间戳数据文件路径
        :param path_cap_info: 抓包端数据文件路径
        :param print_on: 打印与否
        :param to_csv_on: 输出文件与否
        :return: 单包数据
        """
        logging.info('Packet data combining ...')

        # 变量定义 ################################################
        column = self.cFilePktData
        data_ap = pd.read_csv(filepath_or_buffer=path_ap_pkt_data)
        data_cap = pd.read_csv(filepath_or_buffer=path_cap_info)
        data_cap['Read'] = data_cap['Retry'] * 0
        data_cap['Index'] = data_cap['Retry'] * 0
        iApSrc = data_ap.columns.tolist().index('Source')        # 0
        iApDst = data_ap.columns.tolist().index('Destination')   # 1
        iApIdf = data_ap.columns.tolist().index('ID&Fragment')   # 2
        iApPrt = data_ap.columns.tolist().index('Protocol')      # 3
        iApT0 = data_ap.columns.tolist().index('T0')             # 4
        iApT1 = data_ap.columns.tolist().index('T1')             # 5
        iApT2 = data_ap.columns.tolist().index('T2')             # 6
        iApPkgIdEnd = iApPrt + 1
        iCapSrc = data_cap.columns.tolist().index('Source')              # 0
        iCapDst = data_cap.columns.tolist().index('Destination')         # 1
        iCapIdf = data_cap.columns.tolist().index('ID&Fragment')         # 2
        iCapPrt = data_cap.columns.tolist().index('Protocol')            # 3
        iCapLen = data_cap.columns.tolist().index('Length')              # 4
        iCapRty = data_cap.columns.tolist().index('Retry')               # 5
        iCapSts = data_cap.columns.tolist().index('Sniff_Timestamp')     # 6
        iCapRd = data_cap.columns.tolist().index('Read')                 # 7
        iCapIdx = data_cap.columns.tolist().index('Index')               # 8
        iCapPkgIdEnd = iCapPrt + 1
        oSrc = column.index('Source')       # 0
        oDst = column.index('Destination')  # 1
        oIdf = column.index('ID&Fragment')  # 2
        oPrt = column.index('Protocol')     # 3
        oT0 = column.index('T0')            # 4
        oT1 = column.index('T1')            # 5
        oT2 = column.index('T2')            # 6
        oLen = column.index('Length')       # 7
        oRty = column.index('Retry')        # 8
        oPkgIdEnd = oPrt + 1
        data_ap = data_ap.values.tolist()
        data_cap = data_cap.values.tolist()
        len_data_ap = len(data_ap)
        len_data_cap = len(data_cap)
        idxApBgn = 0    # AP端搜索起始索引
        idxCapBgn = 0   # 抓包端搜索起始索引

        # 功能实现 ################################################
        while idxApBgn < len_data_ap and data_ap[idxApBgn][iApPrt] != 6:
            data_ap[idxApBgn].extend([None, None])
            idxApBgn += 1
        while idxCapBgn < len_data_cap and not (data_ap[idxApBgn][iApSrc:iApPkgIdEnd] == data_cap[idxCapBgn][iCapSrc:iCapPkgIdEnd]):
            idxCapBgn += 1
        for i in bar.progressbar(range(idxApBgn, len_data_ap)):
            flag = True
            for j in range(idxCapBgn, len_data_cap):
                if data_cap[j][iCapRd] == 0:
                    if flag:
                        idxCapBgn = j
                        flag = False
                    if data_ap[i][iApSrc:iApPkgIdEnd] == data_cap[j][iCapSrc:iCapPkgIdEnd]:
                        if not -15 < data_cap[j][iCapSts] - data_ap[i][iApT0] < 15:
                            data_ap[i].extend([None, None])
                            break
                        data_ap[i].extend([data_cap[j][iCapLen], data_cap[j][iCapRty]])
                        data_cap[j][iCapRd] = 1
                        data_cap[j][iCapIdx] = i
                        dj = 0
                        nRty = 1
                        while dj < 20 and j + dj + 1 < len_data_cap:
                            dj += 1
                            if data_cap[j+dj][iCapRd] == 0 and (data_ap[i][iApSrc:iApPkgIdEnd] == data_cap[j+dj][iCapSrc:iCapPkgIdEnd]):
                                nRty += 1
                                data_cap[j+dj][iCapRd] = 1
                                data_cap[j+dj][iCapIdx] = i
                                j = j + dj
                                dj = 0
                        data_ap[i][oRty] = nRty
                        break
        data_ap = pd.DataFrame(data=data_ap, columns=column)
        data_cap = pd.DataFrame(data=data_cap)

        # 结果处理 ################################################
        if print_on:
            print(data_ap)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(path_ap_pkt_data), 'rawdata')
            data_ap.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
            output_path = os.path.join(os.path.dirname(path_cap_info), 'capture_info_read')
            data_cap.to_csv(path_or_buf=output_path, index=False)
        return data_ap

    def combine_stm_index(self, path_index: str, path_stm_info: str, print_on=False, to_csv_on=True):
        """
        合并流信息

        :param path_index:
        :param path_stm_info:
        :param print_on:
        :param to_csv_on:
        :return:
        """
        logging.info('Stream features combining ...')

        # 变量定义 ###########################################################################################
        column = self.cFileFeature
        lcFeat = pd.read_csv(filepath_or_buffer=path_index)
        stmInfo = pd.read_csv(filepath_or_buffer=path_stm_info)
        stmInfo['Read'] = stmInfo['Source'] * 0
        iLcT0 = lcFeat.columns.tolist().index('T0')
        iStmSts = stmInfo.columns.tolist().index('Sniff_Timestamp')
        iStmPktRate = stmInfo.columns.tolist().index('PacketRate')
        iStmDataRate = stmInfo.columns.tolist().index('DataRate')
        iStmRead = stmInfo.columns.tolist().index('Read')
        lcFeat = lcFeat.values.tolist()
        stmInfo = stmInfo.values.tolist()
        lenLcFeat = len(lcFeat)
        lenStmInfo = len(stmInfo)
        iLcFeatBgn = 0
        iStmInfoBgn = 0

        # 功能实现 ###########################################################################################
        while iStmInfoBgn < lenStmInfo and not (lcFeat[iLcFeatBgn][0:4] == stmInfo[iStmInfoBgn][0:4]):
            iStmInfoBgn += 1
        for i in bar.progressbar(range(iLcFeatBgn, lenLcFeat)):
            flag = True
            for j in range(iStmInfoBgn, lenStmInfo):
                if stmInfo[j][iStmRead] == 0:
                    if flag:
                        iStmInfoBgn = j
                        flag = False
                    if lcFeat[i][0:4] == stmInfo[j][0:4]:
                        if not -15 < stmInfo[j][iStmSts] - lcFeat[i][iLcT0] < 15:
                            lcFeat[i].extend([None, None])
                        else:
                            lcFeat[i].extend([stmInfo[j][iStmPktRate], stmInfo[j][iStmDataRate]])
                            stmInfo[j][iStmRead] = 1
                        break
        lcFeat = pd.DataFrame(data=lcFeat, columns=column)
        stmInfo = pd.DataFrame(data=stmInfo)
        lcFeat = lcFeat.dropna()

        # 结果处理 ###########################################################################################
        if print_on:
            print(lcFeat)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(path_index), 'feature')
            lcFeat.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
            output_path = os.path.join(os.path.dirname(path_stm_info), 'capture_info_wan_stream_read')
            stmInfo.to_csv(path_or_buf=output_path, index=False)
        return lcFeat

    ################################################################################################################
    # 数据预处理
    def preprocessing(self, path_data: str, print_on=False, to_csv_on=True):
        """
        数据预处理

        :param path_data:
        :param print_on:
        :param to_csv_on:
        :return:
        """
        logging.info('Preprocessing ...')

        # 变量定义 ########################################################
        data = pd.read_csv(filepath_or_buffer=path_data)
        columns_del = ['T1', 'T2', 'Length', 'Retry']
        src_flt = data['Source'].value_counts().idxmax()
        prt_flt = 6

        # 功能实现 ########################################################
        # 空值舍弃
        data = self.delete_null_data(data=data, columns=columns_del)
        # 空值填充
        data = self.fill_null_timestamp(data=data)
        # 业务流筛选
        data = self.filter(filepath_or_dataframe=data, key='Protocol', value=prt_flt)     # 筛选TCP协议
        data = self.filter(filepath_or_dataframe=data, key='Source', value=src_flt)     # 筛选源地址

        # 结果处理 ########################################################
        if print_on:
            print(data)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(path_data), 'data_pkt')
            data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data

    def delete_null_data(self, data: pd.DataFrame, columns: list, output_path=None, print_on=False, to_csv_on=False):
        """
        删除空数据所在行

        :param data: 数据
        :param columns: 待处理的列索引
        :param output_path: 保存路径
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return: 处理后数据
        """
        # 功能实现 ##############################################################
        index_null = data[data[columns].isna().T.any()].index.tolist()
        data = data.drop(index_null)

        # 结果处理 ##############################################################
        if print_on:
            print(data)
        if to_csv_on:
            data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data

    def fill_null_timestamp(self, data: pd.DataFrame, output_path=None, print_on=False, to_csv_on=False):
        """
        空值填充处理（T0）

        :param data: 数据
        :param output_path: 保存路径
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return: 处理后数据
        """
        # 变量定义 #########################################################
        len_data = len(data)
        data_arr = data.values
        iT0 = data.columns.tolist().index('T0')
        iT1 = data.columns.tolist().index('T1')

        # 功能实现 #########################################################
        for i in range(0, len_data):
            if pd.isna(data_arr[i, iT0]):
                n = 0
                for j in range(i, len_data):
                    if pd.isna(data_arr[j, iT0]):
                        n = n + 1
                    else:
                        break
                for k in range(1, n + 1):
                    data_arr[i - 1 + k, iT0] = np.around((data_arr[i - 2 + k, iT0] + data_arr[i - 1 + k, iT1]) / 2, 6)
        data = pd.DataFrame(data=data_arr, columns=data.columns)

        # 结果处理 #########################################################
        if print_on:
            print(data)
        if to_csv_on:
            data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data

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

    ################################################################################################################
    # 特征提取
    def extract_index(self, path_data: str, print_on=False, to_csv_on=True):
        """
        特征提取

        :param path_data: 数据路径
        :param print_on: 是否打印
        :param to_csv_on: 是否输出csv文件
        :return:
        """
        logging.info('Index extracting ...')

        # 变量定义 ################################################################
        data = pd.read_csv(filepath_or_buffer=path_data)

        # 功能实现 ################################################################
        # 提取单包特征
        output_path = os.path.join(os.path.dirname(path_data), 'index_pkt')
        data = self.extract_pkt_index(data=data, output_path=output_path, to_csv_on=to_csv_on)
        # 提取区间特征
        output_path = os.path.join(os.path.dirname(path_data), 'index')
        data = self.extract_region_index(data=data, del_on=True, output_path=output_path, to_csv_on=to_csv_on)

        # 结果处理 ################################################################
        if print_on:
            print(data)
        return data

    def extract_pkt_index(self, data: pd.DataFrame, output_path=None, print_on=False, to_csv_on=False):
        """
        提取单包特征：T1-T0, T2-T1, T2-T0

        :param data: 数据
        :param output_path: 输出路径
        :param print_on: 是否打印
        :param to_csv_on: 是否输出csv文件
        :return:
        """
        logging.info('Packet index extracting ...')

        # 变量定义 ##############################################################

        # 功能实现 ##############################################################
        data['T1_T0'] = np.around(data['T1'] - data['T0'], 6)
        data['T2_T1'] = np.around(data['T2'] - data['T1'], 6)
        data['T2_T0'] = np.around(data['T2'] - data['T0'], 6)

        # 结果处理 ##############################################################
        if print_on:
            print(data)
        if to_csv_on:
            data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data

    def extract_region_index(self, data: pd.DataFrame, output_path=None, del_on=False, print_on=False, to_csv_on=False):
        """
        提取区间特征

        :param data: 数据
        :param output_path: 输出路径
        :param del_on: 是否删除首区间
        :param print_on: 是否打印
        :param to_csv_on: 是否输出csv文件
        :return:
        """
        logging.info('Region index extracting ...')

        # 变量定义 ###################################################
        region = self.region
        apTsp = self.cApTimestamp
        pktIndex = self.cPktFeature
        column = self.cFileLcFeature
        len_data = len(data)

        # 功能实现 ###################################################
        # 添加列索引
        num = 0
        for rgn in region:
            num += (len(apTsp) + len(pktIndex)) * (len_data - rgn + 1)
            # 时间跨度特征
            for at in apTsp:
                name_rgn_at = at + '_' + str(rgn)
                data.insert(len(data.columns), name_rgn_at, None)
            # 区间统计特征
            for pidx in pktIndex:
                name_rgn_pidx_avg = pidx + '_Avg_' + str(rgn)
                data.insert(len(data.columns), name_rgn_pidx_avg, None)
                name_rgn_pidx_std = pidx + '_Std_' + str(rgn)
                data.insert(len(data.columns), name_rgn_pidx_std, None)
        data_arr = data.values
        # 计算特征
        pbar = bar.ProgressBar(max_value=num)
        ibar = [0]
        for rgn in region:
            # 时间跨度特征
            for at in apTsp:
                name_rgn_at = at + '_' + str(rgn)
                iPktIndex = data.columns.tolist().index(at)
                iRgnIndex = data.columns.tolist().index(name_rgn_at)
                for i in range(rgn - 1, len_data):
                    pbar.update(ibar[0])
                    ibar[0] += 1
                    data_arr[i, iRgnIndex] = np.around(data_arr[i, iPktIndex] - data_arr[i - rgn + 1, iPktIndex], 6)
            # 区间统计特征
            for pidx in pktIndex:
                name_rgn_pidx_avg = pidx + '_Avg_' + str(rgn)
                name_rgn_pidx_std = pidx + '_Std_' + str(rgn)
                iPktIndex = data.columns.tolist().index(pidx)
                iRgnAvg = data.columns.tolist().index(name_rgn_pidx_avg)
                iRgnStd = data.columns.tolist().index(name_rgn_pidx_std)
                for i in range(rgn - 1, len_data):
                    pbar.update(ibar[0])
                    ibar[0] += 1
                    data_rgn = [d[iPktIndex] for d in data_arr[(i - rgn + 1):(i + 1)]]
                    data_arr[i, iRgnAvg] = np.around(np.mean(data_rgn), 6)
                    data_arr[i, iRgnStd] = np.around(np.std(data_rgn, ddof=1), 6)
        if del_on:
            data_arr = np.delete(arr=data_arr, obj=np.s_[0:(max(region) - 1)], axis=0)
        data = pd.DataFrame(data=data_arr, columns=data.columns)

        # 结果处理 ###################################################
        if print_on:
            print(data)
        if to_csv_on:
            data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data

    ################################################################################################################
    # 数据标注
    def set_label(self, path_index: str, path_timestamp: str, filename='index_label', print_on=False, to_csv_on=True):
        """
        设置标签

        :param path_index: 特征数据文件路径
        :param path_timestamp: 异常时间戳文件路径
        :param filename: 输出文件名
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出csv文件
        :return:
        """
        logging.info('Label setting ...')

        # 变量定义 #######################################################################
        index = pd.read_csv(filepath_or_buffer=path_index)
        timestamp = pd.read_csv(filepath_or_buffer=path_timestamp)
        index['Label'] = 0

        # 功能实现 #######################################################################
        for i, row in timestamp.iterrows():
            if row['Type'] == 'caton':
                start_time = row['Start_Time']
                end_time = row['End_Time']
                idx_true = ((index['T1']+index['T2'])/2 >= start_time) & ((index['T1']+index['T2'])/2 <= end_time)
                index.loc[idx_true, 'Label'] = 1
            elif row['Type'] == 'whole':
                start_time = row['Start_Time']
                end_time = row['End_Time']
                idx_true = ((index['T1']+index['T2'])/2 >= start_time) & ((index['T1']+index['T2'])/2 <= end_time)
                index = index[idx_true]

        # 结果处理 #######################################################################
        if print_on:
            print(index)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(path_index), filename)
            index.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return index

    ################################################################################################################
    # 数据转换
    def change_unit(self, filepath: str, column: list, coefficient: float):
        """
        数据单位转换

        :param filepath: 数据文件路径
        :param column: 待转换的条目
        :param coefficient: 转换系数
        :return: 转换后的数据
        """
        data = pd.read_csv(filepath_or_buffer=filepath)
        data[column] = data[column] * coefficient
        output_path, ext = os.path.splitext(filepath)
        data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data

    def change_detection_frame_to_time(self, path_vtsp: str, path_bm: str, print_on=False, to_csv_on=True):
        """
        将帧标注转换为时间戳标注

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

    ################################################################################################################
    # 其他
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

        iPcapRetry = self.columnPcapInfo.index('Retry')    # 5
        iPcapSniffTimestamp = self.columnPcapInfo.index('Sniff_Timestamp')  # 6
        iPcapRead = iPcapSniffTimestamp + 1     # 7
        iPcapIdx = iPcapRead + 1                # 8
        iIndexRetry = self.columnIndex.index('Retry')   # 22
        len_ap_index = len(ap_index_arr)
        len_pcap_index = len(pcap_index_arr)

        obj_curr = [None] * self.nIndex
        index_res = [obj_curr]

        start = 0
        while not (ap_index_arr[0][0:self.nPacketID] == pcap_index_arr[start][0:self.nPacketID]):
            start += 1

        for i in bar.progressbar(range(len_ap_index)):
            obj_curr[0:self.nApIndex] = ap_index_arr[i][0:self.nApIndex]
            flag = True
            for j in range(start, len_pcap_index):
                if pcap_index_arr[j][iPcapRead] == 0:
                    if flag:
                        start = j
                        flag = False
                    if ap_index_arr[i][0:self.nPacketID] == pcap_index_arr[j][0:self.nPacketID]:
                        if not -15 < pcap_index_arr[j][iPcapSniffTimestamp] - ap_index_arr[i][self.nPacketID] < 15:
                            obj_curr = [None] * self.nIndex
                            break
                        obj_curr[self.nApIndex:self.nIndex] = pcap_index_arr[j][self.nPacketID:self.nPcapIndex]
                        pcap_index_arr[j][iPcapRead] = 1
                        pcap_index_arr[j][iPcapIdx] = i
                        dj = 0
                        while dj < 20 and j + dj + 1 < len_pcap_index:
                            dj += 1
                            if pcap_index_arr[j+dj][iPcapRead] == 0 and (ap_index_arr[i][0:self.nPacketID] == pcap_index_arr[j+dj][0:self.nPacketID]):
                                obj_curr[iIndexRetry] += pcap_index_arr[j+dj][iPcapRetry]
                                pcap_index_arr[j+dj][iPcapRead] = 1
                                pcap_index_arr[j+dj][iPcapIdx] = i
                                j = j + dj
                                dj = 0
                        index_res.append(obj_curr)
                        obj_curr = [None] * self.nIndex
                        break
        index_df = pd.DataFrame(data=index_res, columns=self.columnIndex)
        pcap_index_df = pd.DataFrame(data=pcap_index_arr)

        if print_on:
            print(index_df)
        if to_csv_on:
            index_df.to_csv(path_or_buf=os.path.join(dir_ap_log, 'index'), index=False, float_format="%.6f")
            pcap_index_df.to_csv(path_or_buf=os.path.join(dir_pcap, 'capture_info_read'), index=False)
        logging.info('Index combine complete.')
        return index_df

    def extract_ap_index(self, dir_ap_log: str, print_on=False, to_csv_on=True):
        """
        提取AP端特征

        :param dir_ap_log: AP时间戳log目录
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
        result = self.fill_null_timestamp(data=result)
        # 删除空数据所在行
        result = self.delete_null_data(data=result)
        # 筛选TCP数据
        result = self.filter(filepath_or_dataframe=result, key='Protocol', value=6)
        # 筛选源地址
        src = result['Source'].value_counts().idxmax()
        result = self.filter(filepath_or_dataframe=result, key='Source', value=src)
        # 提取指标T2_T0_Sync
        result.insert(14, 'T2_T0_Sync', result['T1_T0_Sync'] + result['T2_T1_Sync'])
        # 提取指标T0_100, T0_200, T0_500, T0_1000, T0_2000
        result = self.extract_tx_n(data=result, insert_i=15, column_name='T0_100', index_name='T0_Sync', index_i=4, numstep=100, delete_on=False)
        result = self.extract_tx_n(data=result, insert_i=16, column_name='T0_200', index_name='T0_Sync', index_i=4, numstep=200, delete_on=False)
        result = self.extract_tx_n(data=result, insert_i=17, column_name='T0_500', index_name='T0_Sync', index_i=4, numstep=500, delete_on=False)
        result = self.extract_tx_n(data=result, insert_i=18, column_name='T0_1000', index_name='T0_Sync', index_i=4, numstep=1000, delete_on=False)
        result = self.extract_tx_n(data=result, insert_i=19, column_name='T0_2000', index_name='T0_Sync', index_i=4, numstep=2000)
        # 提取指标Retry_ID
        result = self.extract_retry_id(data=result, insert_i=20, column_name='Retry_ID')

        if print_on:
            print(result)
        if to_csv_on:
            result.to_csv(path_or_buf=os.path.join(dir_ap_log, 'index_tall'), index=False)
        return result

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

    def extract_index_by_window(self, filepath: str, window_size=1000, percentage=95, print_on=False, to_csv_on=True):
        """
        滑动窗口法提取特征（统计特征：均值，标准差；区间特征：跨度）

        :param filepath: 单包特征文件路径
        :param window_size: 滑动窗口大小
        :param percentage: 标记判定阈值
        :param print_on: 是否打印结果
        :param to_csv_on: 是否输出文件
        :return:
        """
        logging.info('Extract index by window(' + str(window_size) + ', ' + str(percentage) + '%).')
        data = pd.read_csv(filepath_or_buffer=filepath)
        columns = data.columns.values.tolist()
        len_columns = len(columns)
        data = data.values.tolist()
        len_data = len(data)
        columns_res = []
        # 统计特征（均值，标准差）的指标
        statistic_index_list = ['T1_T0_Sync', 'T2_T1_Sync', 'T2_T0_Sync', 'Length', 'Retry']
        stat_idx_list = []
        for index in statistic_index_list:
            if index in columns:
                stat_idx_list.append(columns.index(index))
                columns_res.append(index + '_' + str(window_size) + '_Avg')
                columns_res.append(index + '_' + str(window_size) + '_Std')
            else:
                logging.error('Incorrect index name: ' + index)
                return None
        # 区间特征（跨度）的指标
        region_index_list = ['T0_Sync']
        reg_idx_list = []
        for index in region_index_list:
            if index in columns:
                reg_idx_list.append(columns.index(index))
                columns_res.append(index + '_' + str(window_size) + '_Len')
            else:
                logging.error('Incorrect index name: ' + index)
                return None
        label_idx = columns.index('Label')
        columns_res.append('Label')

        result = []
        index_temp = []
        for i in bar.progressbar(range(len_data-window_size+1)):   # len_data-window_size+1
            for idx in stat_idx_list:
                reg_data = [d[idx] for d in data[i:(i+window_size)]]
                index_temp.append(np.mean(reg_data))
                index_temp.append(np.std(reg_data, ddof=1))
            for idx in reg_idx_list:
                index_temp.append(data[i+window_size-1][idx] - data[i][idx])
            if percentage == 0:
                index_temp.append(data[i+window_size-1][label_idx])
            else:
                reg_data = [d[label_idx] for d in data[i:(i + window_size)]]
                if np.mean(reg_data) < percentage / 100:
                    index_temp.append(0)
                else:
                    index_temp.append(1)
            result.append(index_temp)
            index_temp = []
        result = pd.DataFrame(data=result, columns=columns_res)

        if print_on:
            print(result)
        if to_csv_on:
            output_path = os.path.join(os.path.dirname(filepath), 'index_label_region_' + str(window_size) + '_' + str(percentage) + '%')
            result.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return result

    def extract_retry_id(self, data: pd.DataFrame, insert_i: int, column_name: str, output_path=None, print_on=False, to_csv_on=False):
        """
        提取指标Retry_ID

        :param data:
        :param insert_i:
        :param column_name:
        :param output_path:
        :param print_on:
        :param to_csv_on:
        :return:
        """
        logging.info('Extract index: Retry_ID')
        ID_DISTANCE = 32768
        id_max = int(data.iloc[0, 2][:-4], 16)
        id_list = data['ID&Fragment'].values.tolist()
        res_list = [0] * len(id_list)
        # IP id分析
        for idx in bar.progressbar(range(len(id_list))):
            id_now = int(id_list[idx][:-4], 16)
            if (id_now > id_max and id_now - id_max < ID_DISTANCE) or (
                    id_now < id_max and id_max - id_now > ID_DISTANCE):
                id_max = id_now
            else:
                res_list[idx] = 1
        data.insert(insert_i, column_name, res_list)

        if print_on:
            print(data)
        if to_csv_on:
            data.to_csv(path_or_buf=output_path, index=False, float_format="%.6f")
        return data

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
                                       names=self.columnApTimestamp)
        if print_on:
            print(router_timestamp)
        return router_timestamp

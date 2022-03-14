import os.path
import sys

import numpy as np
import pandas as pd
from pyshark import FileCapture

import progressbar as bar
import program_logging
import logging


cPktId = ['Source', 'Destination', 'ID&Fragment', 'Protocol']
cApLog = ['Tn'] + cPktId + ['Time_Sync', 'Time_Local']
cCapWan = cPktId + ['Sniff_Timestamp', 'Length']
cCapWifi = cCapWan + ['Retry']
cCapProto = ['ProtoParam' + str(i) for i in range(1, 11)] + ['ProtoType']
cApDownPkt = cPktId + ['T0', 'T1', 'T2', 'T0d', 'T1d', 'T2d']
cApUpPkt = cPktId + ['TU0', 'TU1', 'TU0d', 'TU1d']
cFeatDown = cApDownPkt + ['T1_T0', 'T2_T1', 'T2_T0', 'Len_Wifi', 'Retry']


def extract_cap_info(path_cap: str, cap_filter: str, path_output: str, isWifi: bool):
    logging.info('Capture info extracting...')
    logging.info('[File] ' + path_cap)
    logging.info('[Filter] ' + cap_filter)
    capture = FileCapture(input_file=path_cap,
                          display_filter=cap_filter,
                          only_summaries=False,
                          keep_packets=False,
                          use_json=True)
    info_list = []
    error_list = []
    pgb = bar.ProgressBar()
    i = [0]

    def ip2int(ip: str):
        return sum([256 ** m * int(n) for m, n in enumerate(ip.split('.')[::1])])

    def combine_id_flags(ipid: str, flags: str):
        return '0x' + ipid[-4:] + flags[-2:] + flags[-4:-2]

    def get_wifi_packet_info(packet):
            if 'wlan_aggregate' in dir(packet):
                if isinstance(packet.wlan_aggregate._all_fields['wlan_aggregate.a_mdsu.subframe'], dict):
                    a_msdu_subframe = [packet.wlan_aggregate._all_fields['wlan_aggregate.a_mdsu.subframe']]
                elif isinstance(packet.wlan_aggregate._all_fields['wlan_aggregate.a_mdsu.subframe'], list):
                    a_msdu_subframe = packet.wlan_aggregate._all_fields['wlan_aggregate.a_mdsu.subframe']
                else:
                    error_list.append([i[0], 'wlan_aggregate'])
                    return
                for subframe in a_msdu_subframe:
                    if 'ip' in subframe:
                        info = [ip2int(subframe['ip']['ip.src']),
                                ip2int(subframe['ip']['ip.dst']),
                                combine_id_flags(subframe['ip']['ip.id'], subframe['ip']['ip.flags']),
                                int(subframe['ip']['ip.proto']),
                                float(packet.sniff_timestamp),
                                int(subframe['wlan_aggregate.a_mdsu.length']),
                                int(packet.wlan.fc_tree.flags_tree.retry)]
                        info_list.append(info)
                    else:
                        error_list.append([i[0], 'subframe'])
            elif 'ip' in packet:
                info = [ip2int(packet.ip.src),
                        ip2int(packet.ip.dst),
                        combine_id_flags(packet.ip.id, packet.ip.flags),
                        int(packet.ip.proto),
                        float(packet.sniff_timestamp),
                        int(packet.length),
                        int(packet.wlan.fc_tree.flags_tree.retry)]
                info_list.append(info)
            else:
                error_list.append([i[0], 'packet'])
            i[0] += 1
            pgb.update(i[0])

    def get_wan_packet_info(packet):
        if 'ip' in packet:
            proto = int(packet.ip.proto)
            info = [ip2int(packet.ip.src),
                    ip2int(packet.ip.dst),
                    combine_id_flags(packet.ip.id, packet.ip.flags),
                    proto,
                    float(packet.sniff_timestamp),
                    int(packet.length)]
            if proto == 17:
                if 'udt' in dir(packet):
                    iscontrol = int(packet.udt.iscontrol)
                    info.append(iscontrol)
                    if iscontrol:
                        udt_type = int(packet.udt.type, 0)
                        info.append(udt_type)
                        if udt_type == 2:
                            info.extend([int(packet.udt.ack_seqno),
                                         int(packet.udt.ackno),
                                         int(packet.udt.rtt),
                                         int(packet.udt.rttvar),
                                         int(packet.udt.buf)])
                            if 'rate' in dir(packet.udt):
                                info.append(int(packet.udt.rate))
                            if 'linkcap' in dir(packet.udt):
                                info.append(int(packet.udt.linkcap))
                        elif udt_type == 6:
                            info.append(int(packet.udt.ackno))
                        elif udt_type == 3:
                            if 'missing_sequence_number_' in dir(packet.udt):
                                msg = str(packet.udt.missing_sequence_number_.expert.message)
                                msg = int(msg[(msg.rfind('[') + 1):msg.rfind(']')])
                                info.append(msg)
                    else:
                        info.append(packet.udt.msgno)
                    proto_type = 'udt'
                else:
                    info.append(packet.udp.length)
                    proto_type = 'udp'
            elif proto == 6:
                info.extend([int(packet.tcp.seq_raw),
                             int(packet.tcp.seq),
                             int(packet.tcp.ack_raw),
                             int(packet.tcp.ack),
                             str(packet.tcp.flags),
                             int(packet.tcp.window_size),
                             int(packet.tcp.len)])
                if 'options_tree' in dir(packet.tcp) and 'sack' in dir(packet.tcp.options_tree):
                    cnt_sack = int(packet.tcp.options_tree.sack_tree.count)
                    if cnt_sack <= 1:
                        info.extend([cnt_sack,
                                     int(packet.tcp.options_tree.sack_tree.sack_le),
                                     int(packet.tcp.options_tree.sack_tree.sack_re)])
                    else:
                        info.extend([cnt_sack,
                                     int(packet.tcp.options_tree.sack_tree.sack_le[0]),
                                     int(packet.tcp.options_tree.sack_tree.sack_re[0])])
                proto_type = 'tcp'
            else:
                proto_type = 'other'
            info.extend([None] * (len_prt - len(info)))
            info.append(proto_type)
            info_list.append(info)
        else:
            error_list.append([i[0], 'packet'])
        i[0] += 1
        pgb.update(i[0])

    if isWifi:
        column = cCapWifi
        capture.apply_on_packets(get_wifi_packet_info)
    else:
        column = cCapWan + cCapProto
        len_prt = column.index('ProtoType')
        capture.apply_on_packets(get_wan_packet_info)
    info_list = pd.DataFrame(data=info_list, columns=column)
    info_list.to_csv(path_or_buf=path_output, index=False)
    if len(error_list) > 1:
        error_list = pd.DataFrame(data=error_list, columns=['No', 'Error_Type'])
        error_list.to_csv(path_or_buf=path_output + '_error.txt', index=False)
    pgb.finish()
    logging.info('[Output] ' + path_output)
    return info_list


def combine_cap_info(path_cap_info_wifi: str, path_cap_info_wan: str, path_output: str):
    logging.info('Capture info combining...')
    logging.info('[Wifi] ' + path_cap_info_wifi)
    logging.info('[Wan] ' + path_cap_info_wan)
    cap_info_wifi = pd.read_csv(filepath_or_buffer=path_cap_info_wifi)
    cap_info_wifi.rename(columns={'Sniff_Timestamp': 'Timestamp_Wifi', 'Length': 'Len_Wifi'}, inplace=True)
    cap_info_wan = pd.read_csv(filepath_or_buffer=path_cap_info_wan)
    cap_info_wan.rename(columns={'Sniff_Timestamp': 'Timestamp_Wan', 'Length': 'Len_Wan'}, inplace=True)
    column = cap_info_wifi.columns.values.tolist() + cap_info_wan.columns.values.tolist()[4:]
    cap_info_wan['Read'] = 0
    len_wifi = cap_info_wifi.shape[0]
    len_wan = cap_info_wan.shape[0]
    iWifiTs = cap_info_wifi.columns.tolist().index('Timestamp_Wifi')
    iWanTs = cap_info_wan.columns.tolist().index('Timestamp_Wan')
    iWanRd = cap_info_wan.columns.tolist().index('Read')
    cap_info_wifi = cap_info_wifi.values.tolist()
    cap_info_wan = cap_info_wan.values.tolist()
    bgnWifi = 0
    bgnWan = 0
    for i in bar.progressbar(range(bgnWifi, len_wifi)):
        packet_id = cap_info_wifi[i][0:4]
        flag = True
        for j in range(bgnWan, len_wan):
            if not cap_info_wan[j][iWanRd]:
                if flag:
                    bgnWan = j
                    flag = False
                if cap_info_wan[j][0:4] == packet_id:
                    if not -15 < cap_info_wan[j][iWanTs] - cap_info_wifi[i][iWifiTs] < 15:
                        break
                    cap_info_wifi[i].extend(cap_info_wan[j][4:iWanRd])
                    cap_info_wan[j][iWanRd] = 1
                    break
    cap_info_wifi = pd.DataFrame(data=cap_info_wifi, columns=column)
    cap_info_wifi.to_csv(path_or_buf=path_output, index=False)
    logging.info('[Output] ' + path_output)


def combine_ap_log_down(dir_ap_log: str):
    path_log_t0 = os.path.join(dir_ap_log, 'log_t0')
    path_log_t1 = os.path.join(dir_ap_log, 'log_t1')
    path_log_t2 = os.path.join(dir_ap_log, 'log_t2')
    log_tall = pd.read_csv(filepath_or_buffer=path_log_t0, header=None, names=cApLog)
    log_temp = pd.read_csv(filepath_or_buffer=path_log_t1, header=None, names=cApLog)
    log_tall = pd.concat(objs=[log_tall, log_temp])
    log_temp = pd.read_csv(filepath_or_buffer=path_log_t2, header=None, names=cApLog)
    log_tall = pd.concat(objs=[log_tall, log_temp])
    log_tall.sort_values(by=['Time_Sync', 'Time_Local'], inplace=True)
    log_tall.to_csv(path_or_buf=os.path.join(dir_ap_log, 'log_tall'), index=False)
    return log_tall


def combine_ap_log_up(dir_ap_log: str):
    path_log_tu0 = os.path.join(dir_ap_log, 'log_tu0')
    path_log_tu1 = os.path.join(dir_ap_log, 'log_tu1')
    log_tuall = pd.read_csv(filepath_or_buffer=path_log_tu0, header=None, names=cApLog)
    log_temp = pd.read_csv(filepath_or_buffer=path_log_tu1, header=None, names=cApLog)
    log_tuall = pd.concat(objs=[log_tuall, log_temp])
    log_tuall.sort_values(by=['Time_Sync', 'Time_Local'], inplace=True)
    log_tuall.to_csv(path_or_buf=os.path.join(dir_ap_log, 'log_tuall'), index=False)
    return log_tuall


def combine_ap_log(dir_cap: str):
    logging.info('AP log combining... (' + dir_cap + ')')
    dir_ap_log = os.path.join(dir_cap, 'ap_log')
    ap_log_down = combine_ap_log_down(dir_ap_log)
    ap_log_up = combine_ap_log_up(dir_ap_log)
    return ap_log_down, ap_log_up


def extract_ap_pkt_down(dir_ap_log: str):
    column = cApDownPkt
    obj_curr = [None] * len(column)
    res = []
    error_list = []
    path_log_tall = dir_ap_log + '/log_tall'
    data = pd.read_csv(filepath_or_buffer=path_log_tall)
    len_data = len(data)
    data['Flag_Read'] = data['Tn'] * 0
    iTn = data.columns.tolist().index('Tn')  # 0
    iSrc = data.columns.tolist().index('Source')  # 1
    iDst = data.columns.tolist().index('Destination')  # 2
    iIdf = data.columns.tolist().index('ID&Fragment')  # 3
    iPrt = data.columns.tolist().index('Protocol')  # 4
    iTsy = data.columns.tolist().index('Time_Sync')  # 5
    iTlc = data.columns.tolist().index('Time_Local')  # 6
    iFlg = data.columns.tolist().index('Flag_Read')  # 7
    iPed = iPrt + 1
    oSrc = column.index('Source')  # 0
    oDst = column.index('Destination')  # 1
    oIdf = column.index('ID&Fragment')  # 2
    oPrt = column.index('Protocol')  # 3
    oT0 = column.index('T0')  # 4
    oT1 = column.index('T1')  # 5
    oT2 = column.index('T2')  # 6
    oT0d = column.index('T0d')  # 7
    oT1d = column.index('T1d')  # 8
    oT2d = column.index('T2d')  # 9
    oPed = oPrt + 1
    oTed = oT2 + 1
    data = data.values.tolist()

    def get_ap_pkt_data(idx_start: int):
        # 判断索引是否越界
        if idx_start >= len_data:
            return
        # 开始遍历匹配
        for idx_cal in range(idx_start, len_data):
            # 判断包标识项是否匹配
            if data[idx_cal][iFlg] == 0 and obj_curr[oSrc:oPed] == data[idx_cal][iSrc:iPed]:
                # 判断当前log行Tn项的值
                tn = data[idx_cal][iTn]
                if tn == 0:
                    # 判断缓存变量中的T0, T1, T2
                    if not any(obj_curr[oT0:oTed]):
                        obj_curr[oT0] = data[idx_cal][iTsy]
                        obj_curr[oT0d] = data[idx_cal][iTlc]
                        data[idx_cal][iFlg] = 1
                    # else:
                    #     break
                elif tn == 1:
                    # 判断缓存变量中的T1, T2
                    if not any(obj_curr[oT1:oTed]):
                        obj_curr[oT1] = data[idx_cal][iTsy]
                        obj_curr[oT1d] = data[idx_cal][iTlc]
                        data[idx_cal][iFlg] = 1
                    # elif obj_curr[oT1] is not None and obj_curr[oT2] is None:
                    #     data[idx_cal, iFlg] = 1
                    # else:
                    #     break
                elif tn == 2:
                    # 判断缓存变量中的T2
                    if not obj_curr[oT2]:
                        obj_curr[oT2] = data[idx_cal][iTsy]
                        obj_curr[oT2d] = data[idx_cal][iTlc]
                        data[idx_cal][iFlg] = 1
                        break
                # else:
                #     error_list.append([idx_cal, tn])
                #     # logging.error('Invalid data: [%d]Tn=%d', idx_cal, tn)
                #     break

    for idx_ex in bar.progressbar(range(0, len_data)):
        # 判断当前log行是否已读取
        if data[idx_ex][iFlg] == 0:
            # 将当前包标识项写入缓存变量
            obj_curr[oSrc] = data[idx_ex][iSrc]
            obj_curr[oDst] = data[idx_ex][iDst]
            obj_curr[oIdf] = data[idx_ex][iIdf]
            obj_curr[oPrt] = data[idx_ex][iPrt]
            # 获取当前包数据
            get_ap_pkt_data(idx_start=idx_ex)
            # 将缓存变量写入结果，清空缓存变量
            res.append(obj_curr)
            obj_curr = [None] * len(column)
    res = pd.DataFrame(data=res, columns=column)
    res.to_csv(path_or_buf=os.path.join(dir_ap_log, 'rawdata_ap_down'), index=False)
    if error_list:
        error_list = pd.DataFrame(data=error_list, columns=['No', 'Tn'])
        error_list.to_csv(path_or_buf=os.path.join(dir_ap_log, 'error_rawdata_ap_down'), index=False)
    return res


def extract_ap_pkt_up(dir_ap_log: str):
    column = cApUpPkt
    obj_curr = [None] * len(column)
    res = []
    error_list = []
    path_log_tuall = dir_ap_log + '/log_tuall'
    data = pd.read_csv(filepath_or_buffer=path_log_tuall)
    len_data = len(data)
    data['Flag_Read'] = data['Tn'] * 0
    iTn = data.columns.tolist().index('Tn')  # 0
    iSrc = data.columns.tolist().index('Source')  # 1
    iDst = data.columns.tolist().index('Destination')  # 2
    iIdf = data.columns.tolist().index('ID&Fragment')  # 3
    iPrt = data.columns.tolist().index('Protocol')  # 4
    iTsy = data.columns.tolist().index('Time_Sync')  # 5
    iTlc = data.columns.tolist().index('Time_Local')  # 6
    iFlg = data.columns.tolist().index('Flag_Read')  # 7
    iPed = iPrt + 1
    oSrc = column.index('Source')  # 0
    oDst = column.index('Destination')  # 1
    oIdf = column.index('ID&Fragment')  # 2
    oPrt = column.index('Protocol')  # 3
    oTU0 = column.index('TU0')  # 4
    oTU1 = column.index('TU1')  # 5
    oTU0d = column.index('TU0d')  # 6
    oTU1d = column.index('TU1d')  # 7
    oPed = oPrt + 1
    oTed = oTU1 + 1
    data = data.values.tolist()

    def get_ap_pkt_data(idx_start: int):
        # 判断索引是否越界
        if idx_start >= len_data:
            return
        # 开始遍历匹配
        for idx_cal in range(idx_start, len_data):
            # 判断包标识项是否匹配
            if data[idx_cal][iFlg] == 0 and obj_curr[oSrc:oPed] == data[idx_cal][iSrc:iPed]:
                # 判断当前log行Tn项的值
                tn = data[idx_cal][iTn]
                if tn == 0:
                    # 判断缓存变量中的T0, T1
                    if not any(obj_curr[oTU0:oTed]):
                        obj_curr[oTU0] = data[idx_cal][iTsy]
                        obj_curr[oTU0d] = data[idx_cal][iTlc]
                        data[idx_cal][iFlg] = 1
                    # else:
                    #     break
                elif tn == 1:
                    # 判断缓存变量中的T1
                    if not obj_curr[oTU1]:
                        obj_curr[oTU1] = data[idx_cal][iTsy]
                        obj_curr[oTU1d] = data[idx_cal][iTlc]
                        data[idx_cal][iFlg] = 1
                        break
                    # elif obj_curr[oTU1] is not None and obj_curr[oT2] is None:
                    #     data[idx_cal, iFlg] = 1
                    # else:
                    #     break
                # else:
                #     error_list.append([idx_cal, tn])
                #     # logging.error('Invalid data: [%d]Tn=%d', idx_cal, tn)
                #     break

    for idx_ex in bar.progressbar(range(0, len_data)):
        # 判断当前log行是否已读取
        if data[idx_ex][iFlg] == 0:
            # 将当前包标识项写入缓存变量
            obj_curr[oSrc] = data[idx_ex][iSrc]
            obj_curr[oDst] = data[idx_ex][iDst]
            obj_curr[oIdf] = data[idx_ex][iIdf]
            obj_curr[oPrt] = data[idx_ex][iPrt]
            # 获取当前包数据
            get_ap_pkt_data(idx_start=idx_ex)
            # 将缓存变量写入结果，清空缓存变量
            res.append(obj_curr)
            obj_curr = [None] * len(column)
    res = pd.DataFrame(data=res, columns=column)
    res.to_csv(path_or_buf=os.path.join(dir_ap_log, 'rawdata_ap_up'), index=False)
    if error_list:
        error_list = pd.DataFrame(data=error_list, columns=['No', 'Tn'])
        error_list.to_csv(path_or_buf=os.path.join(dir_ap_log, 'error_rawdata_ap_up'), index=False)
    return res


def extract_ap_pkt(dir_cap: str):
    logging.info('AP packet data extracting... (' + dir_cap + ')')
    dir_ap_log = dir_cap + '/ap_log'
    ap_pkt_down = extract_ap_pkt_down(dir_ap_log)
    ap_pkt_up = extract_ap_pkt_up(dir_ap_log)
    return ap_pkt_down, ap_pkt_up


def combine_rawdata(path_data_ap: str, path_data_cap: str, path_output: str, isDown=True):
    logging.info('Raw data combining...')
    logging.info('[AP Log] ' + path_data_ap)
    logging.info('[Cap Info] ' + path_data_cap)
    rawdata_ap = pd.read_csv(filepath_or_buffer=path_data_ap)
    cap_info = pd.read_csv(filepath_or_buffer=path_data_cap)
    column = rawdata_ap.columns.values.tolist() + cap_info.columns.values.tolist()[4:]
    cap_info['Read'] = 0
    len_ap = rawdata_ap.shape[0]
    len_cap = cap_info.shape[0]
    if isDown:
        iApT0 = rawdata_ap.columns.tolist().index('T0')
    else:
        iApT0 = rawdata_ap.columns.tolist().index('TU0')
    iCapTs = cap_info.columns.tolist().index('Timestamp_Wan')
    iCapRd = cap_info.columns.tolist().index('Read')
    rawdata_ap = rawdata_ap.values.tolist()
    cap_info = cap_info.values.tolist()
    bgnAp = 0
    bgnCap = 0
    for i in bar.progressbar(range(bgnAp, len_ap)):
        packet_id = rawdata_ap[i][0:4]
        flag = True
        ap_t0 = rawdata_ap[i][iApT0]
        for j in range(bgnCap, len_cap):
            if not cap_info[j][iCapRd]:
                if flag:
                    if -15 < ap_t0 - cap_info[j][iCapTs] < 15:
                        bgnCap = j
                        flag = False
                else:
                    if cap_info[j][0:4] == packet_id:
                        if -15 < ap_t0 - cap_info[j][iCapTs] < 15:
                            rawdata_ap[i].extend(cap_info[j][4:iCapRd])
                            cap_info[j][iCapRd] = 1
                        break
    rawdata_ap = pd.DataFrame(data=rawdata_ap, columns=column)
    rawdata_ap.to_csv(path_or_buf=path_output, index=False)
    logging.info('[Output] ' + path_output)
    return rawdata_ap


def preprocess_down(path_rawdata: str):
    logging.info('Preprocessing...(' + path_rawdata + ')')
    data = pd.read_csv(filepath_or_buffer=path_rawdata)
    iT0 = data.columns.values.tolist().index('T0')
    iT1 = data.columns.values.tolist().index('T1')
    iT2 = data.columns.values.tolist().index('T2')
    iT0d = data.columns.values.tolist().index('T0d')
    iT1d = data.columns.values.tolist().index('T1d')
    iT2d = data.columns.values.tolist().index('T2d')
    idxs = data[data[['T0', 'T1', 'T2']].isnull().T.all()].index.values.tolist()
    data.drop(index=idxs, inplace=True)
    data.reset_index(drop=True, inplace=True)
    idxs = data[data[['T0', 'T1', 'T2']].isnull().T.any()].index.values.tolist()
    pidxs = [i - 1 for i in idxs]
    if pidxs[0] < 0:
        pidxs[0] = data[data[['T0', 'T1', 'T2']].notnull().T.all()].index.values.tolist()[0]
    column = data.columns.values.tolist()
    data = data.values.tolist()
    for i in bar.progressbar(range(len(idxs))):
        idx = idxs[i]
        pidx = pidxs[i]
        if np.isnan(data[idx][iT0]):
            if np.isnan(data[idx][iT1]):
                data[idx][iT1] = data[pidx][iT1] - data[pidx][iT2] + data[idx][iT2]
                data[idx][iT1d] = data[pidx][iT1d] - data[pidx][iT2d] + data[idx][iT2d]
                data[idx][iT0] = data[pidx][iT0] - data[pidx][iT1] + data[idx][iT1]
                data[idx][iT0d] = data[pidx][iT0d] - data[pidx][iT1d] + data[idx][iT1d]
            else:
                data[idx][iT0] = data[pidx][iT0] - data[pidx][iT1] + data[idx][iT1]
                data[idx][iT0d] = data[pidx][iT0d] - data[pidx][iT1d] + data[idx][iT1d]
                if np.isnan(data[idx][iT2]):
                    data[idx][iT2] = data[pidx][iT2] - data[pidx][iT1] + data[idx][iT1]
                    data[idx][iT2d] = data[pidx][iT2d] - data[pidx][iT1d] + data[idx][iT1d]
        elif np.isnan(data[idx][iT1]):
            data[idx][iT1] = data[pidx][iT1] - data[pidx][iT0] + data[idx][iT0]
            data[idx][iT1d] = data[pidx][iT1d] - data[pidx][iT0d] + data[idx][iT0d]
            if np.isnan(data[idx][iT2]):
                data[idx][iT2] = data[pidx][iT2] - data[pidx][iT1] + data[idx][iT1]
                data[idx][iT2d] = data[pidx][iT2d] - data[pidx][iT1d] + data[idx][iT1d]
        else:
            data[idx][iT2] = data[pidx][iT2] - data[pidx][iT1] + data[idx][iT1]
            data[idx][iT2d] = data[pidx][iT2d] - data[pidx][iT1d] + data[idx][iT1d]
    data = pd.DataFrame(data=data, columns=column)
    data.to_csv(path_or_buf=path_rawdata + '_preprocess', index=False)
    return data


def preprocess_up(path_rawdata: str):
    logging.info('Preprocessing...(' + path_rawdata + ')')
    data = pd.read_csv(filepath_or_buffer=path_rawdata)
    iTU0 = data.columns.values.tolist().index('TU0')
    iTU1 = data.columns.values.tolist().index('TU1')
    iTU0d = data.columns.values.tolist().index('TU0d')
    iTU1d = data.columns.values.tolist().index('TU1d')
    idxs = data[data[['TU0', 'TU1']].isnull().T.all()].index.values.tolist()
    data.drop(index=idxs, inplace=True)
    data.reset_index(drop=True, inplace=True)
    idxs = data[data[['TU0', 'TU1']].isnull().T.any()].index.values.tolist()
    pidxs = [i - 1 for i in idxs]
    if pidxs[0] < 0:
        pidxs[0] = data[data[['TU0', 'TU1']].notnull().T.all()].index.values.tolist()[0]
    column = data.columns.values.tolist()
    data = data.values.tolist()
    for i in bar.progressbar(range(len(idxs))):
        idx = idxs[i]
        pidx = pidxs[i]
        if np.isnan(data[idx][iTU0]):
            data[idx][iTU0] = data[pidx][iTU0] - data[pidx][iTU1] + data[idx][iTU1]
            data[idx][iTU0d] = data[pidx][iTU0d] - data[pidx][iTU1d] + data[idx][iTU1d]
        else:
            data[idx][iTU1] = data[pidx][iTU1] - data[pidx][iTU0] + data[idx][iTU0]
            data[idx][iTU1d] = data[pidx][iTU1d] - data[pidx][iTU0d] + data[idx][iTU0d]
    data = pd.DataFrame(data=data, columns=column)
    data.to_csv(path_or_buf=path_rawdata + '_preprocess', index=False)
    return data


def extract_feature(path_data: str, path_feature: str):
    logging.info('Feature extracting...(' + path_data + ')')
    data = pd.read_csv(filepath_or_buffer=path_data)
    data.dropna(subset=['Retry', 'Len_Wifi'], inplace=True)
    src = data['Source'].value_counts().idxmax()
    data = data[data['Source'] == src]
    data['T1_T0'] = data['T1'] - data['T0']
    data['T2_T1'] = data['T2'] - data['T1']
    data['T2_T0'] = data['T2'] - data['T0']
    data = data[cFeatDown]
    len_data = data.shape[0]
    region = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    apTsp = ['T0', 'T1', 'T2']
    pktIndex = ['T1_T0', 'T2_T1', 'T2_T0', 'Retry', 'Len_Wifi']
    # 添加列索引
    column_rgn = []
    for rgn in region:
        # 时间跨度特征
        for at in apTsp:
            name_rgn_at = at + '_' + str(rgn)
            column_rgn.append(name_rgn_at)
        # 区间统计特征
        for pidx in pktIndex:
            name_rgn_pidx_avg = pidx + '_Avg_' + str(rgn)
            name_rgn_pidx_std = pidx + '_Std_' + str(rgn)
            column_rgn.extend([name_rgn_pidx_avg, name_rgn_pidx_std])
    data = pd.concat([data, pd.DataFrame(columns=column_rgn)], sort=False)
    data_arr = data.values.tolist()
    # 计算特征
    pbar = bar.ProgressBar(max_value=(len(apTsp) + len(pktIndex)) * len_data * len(region))
    ibar = [0]
    for rgn in region:
        # 时间跨度特征
        for at in apTsp:
            name_rgn_at = at + '_' + str(rgn)
            iPktIndex = data.columns.tolist().index(at)
            iRgnIndex = data.columns.tolist().index(name_rgn_at)
            for i in range(0, rgn - 1):
                pbar.update(ibar[0])
                ibar[0] += 1
                data_arr[i][iRgnIndex] = np.around(data_arr[i][iPktIndex] - data_arr[0][iPktIndex], 6)
            for i in range(rgn - 1, len_data):
                pbar.update(ibar[0])
                ibar[0] += 1
                data_arr[i][iRgnIndex] = np.around(data_arr[i][iPktIndex] - data_arr[i - rgn + 1][iPktIndex], 6)
        # 区间统计特征
        for pidx in pktIndex:
            name_rgn_pidx_avg = pidx + '_Avg_' + str(rgn)
            name_rgn_pidx_std = pidx + '_Std_' + str(rgn)
            iPktIndex = data.columns.tolist().index(pidx)
            iRgnAvg = data.columns.tolist().index(name_rgn_pidx_avg)
            iRgnStd = data.columns.tolist().index(name_rgn_pidx_std)
            pbar.update(ibar[0])
            ibar[0] += 1
            data_arr[0][iRgnAvg] = data_arr[0][iPktIndex]
            data_arr[0][iRgnStd] = 0
            for i in range(1, rgn - 1):
                pbar.update(ibar[0])
                ibar[0] += 1
                data_rgn = [d[iPktIndex] for d in data_arr[0:(i + 1)]]
                data_arr[i][iRgnAvg] = np.around(np.nanmean(data_rgn), 6)
                data_arr[i][iRgnStd] = np.around(np.nanstd(data_rgn, ddof=1), 6)
            for i in range(rgn - 1, len_data):
                pbar.update(ibar[0])
                ibar[0] += 1
                data_rgn = [d[iPktIndex] for d in data_arr[(i - rgn + 1):(i + 1)]]
                data_arr[i][iRgnAvg] = np.around(np.nanmean(data_rgn), 6)
                data_arr[i][iRgnStd] = np.around(np.nanstd(data_rgn, ddof=1), 6)
    data = pd.DataFrame(data=data_arr, columns=data.columns)
    pbar.finish()
    data.to_csv(path_or_buf=path_feature, index=False)
    return data


def set_label(path_data: str, path_timestamp: str):
    logging.info('Label setting...(' + path_data + ')')
    data = pd.read_csv(filepath_or_buffer=path_data)
    timestamp = pd.read_csv(filepath_or_buffer=path_timestamp)
    data['Label'] = 0
    for i, row in timestamp.iterrows():
        if row['Type'] == 'blackborder':
            start_time = row['Start_Time']
            end_time = row['End_Time']
            idx_true = ((data['T1'] + data['T2']) / 2 >= start_time) & ((data['T1'] + data['T2']) / 2 <= end_time)
            data.loc[idx_true, 'Label'] = 1
        elif row['Type'] == 'whole':
            start_time = row['Start_Time']
            end_time = row['End_Time']
            idx_true = ((data['T1'] + data['T2']) / 2 >= start_time) & ((data['T1'] + data['T2']) / 2 <= end_time)
            data = data[idx_true]
    data.to_csv(path_or_buf=path_data + '_label', index=False)


def data_processing(path_params: list, processes: list):
    # 文件路径
    if len(sys.argv) > 4:
        objs = [int(sys.argv[1])]
        envs = path_params[1] if sys.argv[2] == 'n' else [int(sys.argv[2])]
        idxs = path_params[2] if sys.argv[3] == 'n' else [int(sys.argv[3])]
    else:
        objs = path_params[0]
        envs = path_params[1]
        idxs = path_params[2]
    dirs_cap = []
    paths_pcap_wan = []
    paths_pcap_wifi = []
    for obj in objs:
        for env in envs:
            for idx in idxs:
                root = 'data/data_game'
                dtype = 'g'
                name_to = dtype + str(obj)
                name_toe = name_to + '_' + str(env)
                name_toei = name_toe + '_' + str(idx)
                dir_cap = root + '/' + name_to + '/' + name_toe + '/' + name_toei
                path_pcap_wan = dir_cap + '/pcap_wan_' + name_toei + '.pcapng'
                path_pcap_wifi = dir_cap + '/pcap_wifi_' + name_toei + '.pcap'
                if os.path.exists(dir_cap):
                    dirs_cap.append(dir_cap)
                    paths_pcap_wan.append(path_pcap_wan)
                    paths_pcap_wifi.append(path_pcap_wifi)
    # 数据处理
    if len(sys.argv) > 4:
        processes = list(map(int, sys.argv[4:]))
    for process in processes:
        # 提取wifi抓包下行数据 --> capture_info_wifi_down
        if int(process) == 1:
            for path_pcap_wifi in paths_pcap_wifi:
                extract_cap_info(path_cap=path_pcap_wifi,
                                 cap_filter='ip.dst == 192.168.137.160',
                                 path_output=os.path.dirname(path_pcap_wifi) + '/capture_info_wifi_down',
                                 isWifi=True)
        # 提取wifi抓包上行数据 --> capture_info_wifi_up
        if int(process) == 2:
            for path_pcap_wifi in paths_pcap_wifi:
                extract_cap_info(path_cap=path_pcap_wifi,
                                 cap_filter='ip.src == 192.168.137.160',
                                 path_output=os.path.dirname(path_pcap_wifi) + '/capture_info_wifi_up',
                                 isWifi=True)
        # 提取wan抓包下行数据 --> capture_info_wan_down
        if int(process) == 3:
            for path_pcap_wan in paths_pcap_wan:
                extract_cap_info(path_cap=path_pcap_wan,
                                 cap_filter='ip.dst == 192.168.137.160',
                                 path_output=os.path.dirname(path_pcap_wan) + '/capture_info_wan_down',
                                 isWifi=False)
        # 提取wan抓包上行数据 --> capture_info_wan_up
        if int(process) == 4:
            for path_pcap_wan in paths_pcap_wan:
                extract_cap_info(path_cap=path_pcap_wan,
                                 cap_filter='ip.src == 192.168.137.160',
                                 path_output=os.path.dirname(path_pcap_wan) + '/capture_info_wan_up',
                                 isWifi=False)
        # 合并抓包下行数据 --> capture_info_down
        if int(process) == 5:
            for dir_cap in dirs_cap:
                path_cap_info_wifi = dir_cap + '/capture_info_wifi_down'
                path_cap_info_wan = dir_cap + '/capture_info_wan_down'
                path_cap_info = path_cap_info_wifi.replace('_wifi', '')
                if os.path.exists(path_cap_info):
                    continue
                combine_cap_info(path_cap_info_wifi=path_cap_info_wifi,
                                 path_cap_info_wan=path_cap_info_wan,
                                 path_output=path_cap_info)
        # 合并抓包上行数据 --> capture_info_up
        if int(process) == 6:
            for dir_cap in dirs_cap:
                path_cap_info_wifi = dir_cap + '/capture_info_wifi_up'
                path_cap_info_wan = dir_cap + '/capture_info_wan_up'
                path_cap_info = path_cap_info_wifi.replace('_wifi', '')
                if os.path.exists(path_cap_info):
                    continue
                combine_cap_info(path_cap_info_wifi=path_cap_info_wifi,
                                 path_cap_info_wan=path_cap_info_wan,
                                 path_output=path_cap_info)
        # 合并ap_log --> log_tall, log_tuall
        if int(process) == 7:
            for dir_cap in dirs_cap:
                combine_ap_log(dir_cap)
        # 提取AP单包数据 --> rawdata_ap_down, rawdata_ap_up
        if int(process) == 8:
            for dir_cap in dirs_cap:
                extract_ap_pkt(dir_cap)
        # 合并AP和抓包下行数据 --> rawdata_down
        if int(process) == 9:
            for dir_cap in dirs_cap:
                path_rawdata = dir_cap + '/ap_log/rawdata_down'
                if os.path.exists(path_rawdata):
                    continue
                combine_rawdata(path_data_ap=dir_cap + '/ap_log/rawdata_ap_down',
                                path_data_cap=dir_cap + '/capture_info_down',
                                path_output=path_rawdata)
        # 合并AP和抓包上行数据 --> rawdata_up
        if int(process) == 10:
            for dir_cap in dirs_cap:
                path_rawdata = dir_cap + '/ap_log/rawdata_up'
                if os.path.exists(path_rawdata):
                    continue
                combine_rawdata(path_data_ap=dir_cap + '/ap_log/rawdata_ap_up',
                                path_data_cap=dir_cap + '/capture_info_up',
                                path_output=path_rawdata,
                                isDown=False)
        # 下行数据预处理 --> rawdata_down_preprocess
        if int(process) == 11:
            for dir_cap in dirs_cap:
                preprocess_down(path_rawdata=dir_cap + '/ap_log/rawdata_down')
        # 上行数据预处理 --> rawdata_up_preprocess
        if int(process) == 12:
            for dir_cap in dirs_cap:
                preprocess_up(path_rawdata=dir_cap + '/ap_log/rawdata_up')
        # 提取下行特征 --> feature_down
        if int(process) == 13:
            for dir_cap in dirs_cap:
                extract_feature(path_data=dir_cap + '/ap_log/rawdata_down_preprocess',
                                path_feature=dir_cap + '/ap_log/feature_down')
        # 提取上行特征 --> feature_up
        if int(process) == 14:
            for dir_cap in dirs_cap:
                extract_feature(path_data=dir_cap + '/ap_log/rawdata_up_preprocess',
                                path_feature=dir_cap + '/ap_log/feature_up')
        # 下行数据标注 --> feature_down_label
        if int(process) == 15:
            for dir_cap in dirs_cap:
                set_label(path_data=dir_cap + '/ap_log/feature_down',
                          path_timestamp=dir_cap + '/ScreenRecorder/result_detection_timestamp.txt')
        # 上行数据标注 --> feature_up_label
        if int(process) == 16:
            for dir_cap in dirs_cap:
                set_label(path_data=dir_cap + '/ap_log/feature_up',
                          path_timestamp=dir_cap + '/ScreenRecorder/result_detection_timestamp.txt')


if __name__ == '__main__':
    objs = [2]      # 游戏编号: 2, 3, 5, 6
    envs = [0]      # 环境编号: 0, 1, 2, 3
    idxs = [1]      # 样本编号: 1, 2, 3, 4, 5
    processes = [12]    # 处理编号
    path_params = [objs, envs, idxs]
    data_processing(path_params=path_params, processes=processes)

    # 测试
    # extract_cap_info(path_cap='data/data_game/g2/g2_0/g2_0_1/pcap_wan_g2_0_1_10000.pcapng',
    #                  cap_filter='ip.addr == 192.168.137.160',
    #                  path_output='data/data_game/g2/g2_0/g2_0_1/capture_info_proto',
    #                  isWifi=False)
    # path_cap_info_wifi = 'data/data_game/g2/g2_0/g2_0_1/test_capture_info_wifi_down'
    # path_cap_info_wan = 'data/data_game/g2/g2_0/g2_0_1/test_capture_info_wan_down'
    # combine_cap_info(path_cap_info_wifi=path_cap_info_wifi,
    #                  path_cap_info_wan=path_cap_info_wan)

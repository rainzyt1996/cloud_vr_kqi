
import os
import sys

import logging
import pandas as pd
import program_logging
import progressbar as bar
from pymrmre import mrmr
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import warnings

warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice.')


REGION = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def evaluate(labels, predict):
    accuracy = accuracy_score(labels, predict)
    precision = precision_score(labels, predict)
    recall = recall_score(labels, predict)
    f1 = f1_score(labels, predict)
    fpr, tpr, thresholds = roc_curve(labels, predict)
    thresholds = thresholds[np.where(tpr == recall)][0]
    fpr = fpr[np.where(tpr == recall)][0]
    tpr = tpr[np.where(tpr == recall)][0]
    return accuracy, precision, recall, f1, fpr, tpr, thresholds


def extract_ts_stm_info(dir_cap: str):
    logging.info('TS stream info extract.(' + dir_cap + ')')
    path_http = os.path.join(dir_cap, 'capture_info_wan_http')
    path_strm = os.path.join(dir_cap, 'capture_info_wan_stream')
    path_ts_stm_info = os.path.join(dir_cap, 'ts_stm_info')
    path_stm_ts = os.path.join(dir_cap, 'capture_info_wan_ts')

    data = pd.read_csv(filepath_or_buffer=path_http)
    len_data = data.shape[0]
    iTst = data.columns.tolist().index('Sniff_Timestamp')
    column = ['TS_No', 'Sniff_Timestamp', 'Interval_TS', 'Num_Packet', 'Sum_Length', 'PacketRate_TS', 'DataRate_TS']
    tsStmInfo = []
    for i in range(0, len_data - 1):
        tsStmInfo.append([i,
                          data.iloc[i, iTst],
                          data.iloc[i + 1, iTst] - data.iloc[i, iTst]])
    tsStmInfo.append([len_data - 1,
                      data.iloc[len_data - 1, iTst],
                      data.iloc[len_data - 1, iTst]])

    data = pd.read_csv(filepath_or_buffer=path_strm)
    len_data = data.shape[0]
    iTst = data.columns.tolist().index('Sniff_Timestamp')
    iNum = data.columns.tolist().index('Number')
    iSml = data.columns.tolist().index('SumLength')
    stm_ts = data.loc[:, ['Source', 'Destination', 'ID&Fragment', 'Protocol',
                          'Sniff_Timestamp', 'PacketRate', 'DataRate']]
    stm_ts['TS_No'] = stm_ts['Protocol'] * 0
    oNo = stm_ts.columns.tolist().index('TS_No')
    stm_ts = stm_ts.values.tolist()
    data = data.values.tolist()
    num_ts = len(tsStmInfo)
    tsStmInfo[num_ts - 1][2] = data[len_data - 1][iTst] - tsStmInfo[num_ts - 1][2]
    j = 0
    for i in bar.progressbar(range(1, len_data - 1)):
        if data[i][iNum] == 1:
            tsStmInfo[j].extend([data[i - 1][iNum],
                                 data[i - 1][iSml],
                                 data[i - 1][iNum] / tsStmInfo[j][2],
                                 data[i - 1][iSml] / tsStmInfo[j][2]])
            j += 1
        stm_ts[i][oNo] = j
    tsStmInfo[j].extend([data[len_data - 1][iNum],
                         data[len_data - 1][iSml],
                         data[len_data - 1][iNum] / tsStmInfo[j][2],
                         data[len_data - 1][iSml] / tsStmInfo[j][2]])
    stm_ts[len_data - 1][oNo] = j
    tsStmInfo = pd.DataFrame(data=tsStmInfo, columns=column)
    tsStmInfo.to_csv(path_or_buf=path_ts_stm_info, index=False)
    stm_ts = pd.DataFrame(data=stm_ts, columns=['Source', 'Destination', 'ID&Fragment', 'Protocol',
                                                'Sniff_Timestamp', 'PacketRate', 'DataRate', 'TS_No'])
    stm_ts.to_csv(path_or_buf=path_stm_ts, index=False)


def extract_ts_pkt_info(dir_ap: str):
    dir_cap = os.path.dirname(dir_ap)
    logging.info('TS packet info extract.(' + dir_cap + ')')
    dir_scr = os.path.join(dir_cap, 'ScreenRecorder')
    path_index_pkt = os.path.join(dir_ap, 'index_pkt')
    path_timestamp = os.path.join(dir_scr, 'result_detection_timestamp.txt')
    path_stm_ts = os.path.join(dir_cap, 'capture_info_wan_ts')
    path_ts_stm_info = os.path.join(dir_cap, 'ts_stm_info')
    path_index_pkt_ts = os.path.join(dir_ap, 'index_pkt_ts_label')
    path_ts_feat_label = os.path.join(dir_cap, 'ts_feature_label')

    index_pkt = pd.read_csv(filepath_or_buffer=path_index_pkt)
    timestamp = pd.read_csv(filepath_or_buffer=path_timestamp)
    index_pkt['Label'] = 0
    for i, row in timestamp.iterrows():
        if row['Type'] == 'caton':
            start_time = row['Start_Time']
            end_time = row['End_Time']
            idx_true = ((index_pkt['T1'] + index_pkt['T2']) / 2 >= start_time) & ((index_pkt['T1'] + index_pkt['T2']) / 2 <= end_time)
            index_pkt.loc[idx_true, 'Label'] = 1
        elif row['Type'] == 'whole':
            start_time = row['Start_Time']
            end_time = row['End_Time']
            idx_true = ((index_pkt['T1'] + index_pkt['T2']) / 2 >= start_time) & ((index_pkt['T1'] + index_pkt['T2']) / 2 <= end_time)
            index_pkt = index_pkt[idx_true]

    iIdxT0 = index_pkt.columns.tolist().index('T0')
    len_idx = index_pkt.shape[0]
    stm_ts = pd.read_csv(filepath_or_buffer=path_stm_ts)
    stm_ts['Read'] = stm_ts['Protocol'] * 0
    iStmSts = stm_ts.columns.tolist().index('Sniff_Timestamp')
    iStmPr = stm_ts.columns.tolist().index('PacketRate')
    iStmDr = stm_ts.columns.tolist().index('DataRate')
    iStmNo = stm_ts.columns.tolist().index('TS_No')
    iStmRd = stm_ts.columns.tolist().index('Read')
    len_stm = stm_ts.shape[0]
    index_pkt = index_pkt.values.tolist()
    stm_ts = stm_ts.values.tolist()
    iIdxBgn = 0
    iStmBgn = 0

    while iIdxBgn < len_idx and index_pkt[iIdxBgn][2] == '0x00004000':
        iIdxBgn += 1
    while iStmBgn < len_stm and not index_pkt[iIdxBgn][0:4] == stm_ts[iStmBgn][0:4]:
        iStmBgn += 1
    for i in bar.progressbar(range(iIdxBgn, len_idx)):
        flag = True
        for j in range(iStmBgn, len_stm):
            if stm_ts[j][iStmRd] == 0:
                if flag:
                    iStmBgn = j
                    flag = False
                if index_pkt[i][0:4] == stm_ts[j][0:4]:
                    if not -15 < stm_ts[j][iStmSts] - index_pkt[i][iIdxT0] < 15:
                        index_pkt[i].extend([None, None, None])
                        break
                    index_pkt[i].extend([stm_ts[j][iStmPr], stm_ts[j][iStmDr], stm_ts[j][iStmNo]])
                    stm_ts[j][iStmRd] = 1
                    break
    index_pkt = pd.DataFrame(data=index_pkt,
                             columns=['Source', 'Destination', 'ID&Fragment', 'Protocol',
                                      'T0', 'T1', 'T2', 'Length', 'Retry',
                                      'T1_T0', 'T2_T1', 'T2_T0', 'Label', 'PacketRate', 'DataRate', 'TS_No'])
    index_pkt.to_csv(path_or_buf=path_index_pkt_ts, index=False)

    ts_stm_info = pd.read_csv(filepath_or_buffer=path_ts_stm_info)
    len_ts = ts_stm_info.shape[0]
    iTsNo = ts_stm_info.columns.tolist().index('TS_No')
    column = ts_stm_info.columns.tolist()
    column.extend(['T1_T0_Avg_TS', 'T1_T0_Std_TS',
                   'T2_T1_Avg_TS', 'T2_T1_Std_TS',
                   'T2_T0_Avg_TS', 'T2_T0_Std_TS',
                   'Length_Avg_TS', 'Length_Std_TS',
                   'Retry_Avg_TS', 'Retry_Std_TS',
                   'Label'])
    ts_stm_info = ts_stm_info.values.tolist()
    for i in range(0, len_ts):
        ts_pkt = index_pkt[index_pkt['TS_No'] == ts_stm_info[i][iTsNo]]
        ts_stm_info[i].extend([ts_pkt['T1_T0'].mean(), ts_pkt['T1_T0'].std(),
                               ts_pkt['T2_T1'].mean(), ts_pkt['T2_T1'].std(),
                               ts_pkt['T2_T0'].mean(), ts_pkt['T2_T0'].std(),
                               ts_pkt['Length'].mean(), ts_pkt['Length'].std(),
                               ts_pkt['Retry'].mean(), ts_pkt['Retry'].std()])
        if ts_pkt['Label'].sum() > 0:
            ts_stm_info[i].extend([1])
        else:
            ts_stm_info[i].extend([0])
    ts_stm_info = pd.DataFrame(data=ts_stm_info, columns=column)
    ts_stm_info.to_csv(path_or_buf=path_ts_feat_label, index=False)


def extract_index_pkt_region(path_index_pkt: str, path_output: str, region):
    logging.info('Region index extracting...(' + path_index_pkt + ')')
    data = pd.read_csv(filepath_or_buffer=path_index_pkt)
    data = data.dropna(subset=['TS_No'])
    len_data = data.shape[0]
    apTsp = ['T0', 'T1', 'T2']
    pktIndex = ['T1_T0', 'T2_T1', 'T2_T0', 'Length', 'Retry']

    # 添加列索引
    column_rgn = []
    for rgn in region:
        # 时间跨度特征
        for at in apTsp:
            name_rgn_at = at + '_' + str(rgn)
            column_rgn.append(name_rgn_at)
            # data.insert(len(data.columns), name_rgn_at, None)
        # 区间统计特征
        for pidx in pktIndex:
            name_rgn_pidx_avg = pidx + '_Avg_' + str(rgn)
            # data.insert(len(data.columns), name_rgn_pidx_avg, None)
            name_rgn_pidx_std = pidx + '_Std_' + str(rgn)
            # data.insert(len(data.columns), name_rgn_pidx_std, None)
            column_rgn.extend([name_rgn_pidx_avg, name_rgn_pidx_std])
    data = pd.concat([data, pd.DataFrame(columns=column_rgn)], sort=False)
    data_arr = data.values
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
                data_arr[i, iRgnIndex] = np.around(data_arr[i, iPktIndex] - data_arr[0, iPktIndex], 6)
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
            pbar.update(ibar[0])
            ibar[0] += 1
            data_arr[0, iRgnAvg] = data_arr[0, iPktIndex]
            data_arr[0, iRgnStd] = 0
            for i in range(1, rgn - 1):
                pbar.update(ibar[0])
                ibar[0] += 1
                data_rgn = [d[iPktIndex] for d in data_arr[0:(i + 1)]]
                data_arr[i, iRgnAvg] = np.around(np.nanmean(data_rgn), 6)
                data_arr[i, iRgnStd] = np.around(np.nanstd(data_rgn, ddof=1), 6)
            for i in range(rgn - 1, len_data):
                pbar.update(ibar[0])
                ibar[0] += 1
                data_rgn = [d[iPktIndex] for d in data_arr[(i - rgn + 1):(i + 1)]]
                data_arr[i, iRgnAvg] = np.around(np.nanmean(data_rgn), 6)
                data_arr[i, iRgnStd] = np.around(np.nanstd(data_rgn, ddof=1), 6)
    data = pd.DataFrame(data=data_arr, columns=data.columns)
    pbar.finish()
    data.to_csv(path_or_buf=path_output, index=False, float_format="%.6f")


def extract_index_pkt_add_disorder(path_index_pkt: str, path_output: str, region):
    # 提取并增加乱序情况（单包，区间）
    logging.info('Disorder extracting...(' + path_index_pkt + ')')
    data = pd.read_csv(filepath_or_buffer=path_index_pkt)
    if 'Disorder' in data.columns:
        return
    len_data = data.shape[0]
    column_add = ['Disorder']
    for rgn in region:
        name_avg = 'Disorder_Avg_' + str(rgn)
        name_std = 'Disorder_Std_' + str(rgn)
        column_add.extend([name_avg, name_std])
    data = pd.concat([data, pd.DataFrame(columns=column_add)], sort=False)
    column = data.columns.tolist()
    iDsd = column.index('Disorder')
    data = data.values
    # 乱序
    mid = int(data[0][2][2:6], 16) - 1
    for i in bar.progressbar(range(len_data)):
        ipid = int(data[i][2][2:6], 16)
        did = ipid - mid
        if 0 < did < 32768 or did < -64000:
            mid = ipid
            data[i][iDsd] = 0
        else:
            data[i][iDsd] = 1
    # 区间统计特征
    pbar = bar.ProgressBar(max_value=(len_data - 1) * len(region))
    ibar = [0]
    for rgn in region:
        name_avg = 'Disorder_Avg_' + str(rgn)
        name_std = 'Disorder_Std_' + str(rgn)
        iRgnAvg = column.index(name_avg)
        iRgnStd = column.index(name_std)
        for i in range(1, rgn - 1):
            pbar.update(ibar[0])
            ibar[0] += 1
            data_rgn = [d[iDsd] for d in data[0:(i + 1)]]
            data[i, iRgnAvg] = np.around(np.nanmean(data_rgn), 6)
            data[i, iRgnStd] = np.around(np.nanstd(data_rgn, ddof=1), 6)
        for i in range(rgn - 1, len_data):
            pbar.update(ibar[0])
            ibar[0] += 1
            data_rgn = [d[iDsd] for d in data[(i - rgn + 1):(i + 1)]]
            data[i, iRgnAvg] = np.around(np.nanmean(data_rgn), 6)
            data[i, iRgnStd] = np.around(np.nanstd(data_rgn, ddof=1), 6)
    pbar.finish()
    data = pd.DataFrame(data=data, columns=column)
    data.to_csv(path_or_buf=path_output, index=False)
    return


def extract_index_pkt_add_dt(path_index_pkt: str, path_output: str, region):
    # 提取并增加包间节点时间差（单包&区间）
    logging.info('DT extracting...(' + path_index_pkt + ')')
    data = pd.read_csv(filepath_or_buffer=path_index_pkt)
    if 'DT0' in data.columns:
        return
    len_data = data.shape[0]
    column_dt = ['DT0', 'DT1', 'DT2']
    column_add = column_dt.copy()
    for rgn in region:
        for cname in column_dt:
            name_avg = cname + '_Avg_' + str(rgn)
            name_std = cname + '_Std_' + str(rgn)
            column_add.extend([name_avg, name_std])
    data = pd.concat([data, pd.DataFrame(columns=column_add)], sort=False)
    column = data.columns.tolist()
    iT0 = column.index('T0')
    iT1 = column.index('T1')
    iT2 = column.index('T2')
    iDT0 = column.index('DT0')
    iDT1 = column.index('DT1')
    iDT2 = column.index('DT2')
    data = data.values
    # 包间节点时间差
    dt0 = data[1:len_data, iT0] - data[0:(len_data-1), iT0]
    dt0 = np.around(dt0.tolist(), 6)
    dt0 = np.insert(dt0, 0, 0, axis=0)
    data[0:len_data, iDT0] = dt0
    dt1 = data[1:len_data, iT1] - data[0:(len_data-1), iT1]
    dt1 = np.around(dt1.tolist(), 6)
    dt1 = np.insert(dt1, 0, 0, axis=0)
    data[0:len_data, iDT1] = dt1
    dt2 = data[1:len_data, iT2] - data[0:(len_data-1), iT2]
    dt2 = np.around(dt2.tolist(), 6)
    dt2 = np.insert(dt2, 0, 0, axis=0)
    data[0:len_data, iDT2] = dt2
    # 区间统计特征
    pbar = bar.ProgressBar(max_value=(len_data - 1) * len(region) * len(column_dt))
    ibar = [0]
    for rgn in region:
        for cname in column_dt:
            name_avg = cname + '_Avg_' + str(rgn)
            name_std = cname + '_Std_' + str(rgn)
            iDT = column.index(cname)
            iRgnAvg = column.index(name_avg)
            iRgnStd = column.index(name_std)
            for i in range(1, rgn - 1):
                pbar.update(ibar[0])
                ibar[0] += 1
                data_rgn = [d[iDT] for d in data[1:(i + 1)]]
                data[i, iRgnAvg] = np.around(np.nanmean(data_rgn), 6)
                data[i, iRgnStd] = np.around(np.nanstd(data_rgn, ddof=1), 6)
            for i in range(rgn - 1, len_data):
                pbar.update(ibar[0])
                ibar[0] += 1
                data_rgn = [d[iDT] for d in data[(i - rgn + 2):(i + 1)]]
                data[i, iRgnAvg] = np.around(np.nanmean(data_rgn), 6)
                data[i, iRgnStd] = np.around(np.nanstd(data_rgn, ddof=1), 6)
    pbar.finish()
    data = pd.DataFrame(data=data, columns=column)
    data.to_csv(path_or_buf=path_output, index=False)
    return


def extract_ts_pkt_add(path_index_pkt: str, path_ts_feature: str, path_output: str):
    # 提取并增加分片包间特征（节点时间差统计量，片内乱序统计量，包速率，码速率）
    logging.info('TS feature adding...(' + path_ts_feature + ')')
    index_pkt = pd.read_csv(filepath_or_buffer=path_index_pkt,
                            usecols=['TS_No', 'T0', 'T1', 'T2', 'Disorder', 'Length'])
    ts_feat = pd.read_csv(filepath_or_buffer=path_ts_feature)
    column_add = ['Disorder_Avg_TS', 'Disorder_Std_TS',
                  'DT0_Avg_TS', 'DT0_Std_TS',
                  'DT1_Avg_TS', 'DT1_Std_TS',
                  'DT2_Avg_TS', 'DT2_Std_TS',
                  'PktRate_T0_TS', 'BitRate_T0_TS',
                  'PktRate_T1_TS', 'BitRate_T1_TS',
                  'PktRate_T2_TS', 'BitRate_T2_TS']
    ts_feat = pd.concat([ts_feat, pd.DataFrame(columns=column_add)], sort=False)
    ts_nos = ts_feat['TS_No'].values.tolist()
    pbar = bar.ProgressBar(max_value=len(ts_nos))
    ibar = [0]
    for ts_no in ts_nos:
        pbar.update(ibar[0])
        ibar[0] += 1
        index_pkt_ts = index_pkt[index_pkt['TS_No'] == ts_no][['T0', 'T1', 'T2', 'Disorder', 'Length']].copy().values
        num_pkt = index_pkt_ts.shape[0]
        # Disorder
        disorder_ts = index_pkt_ts[:, 3]
        ts_feat.loc[ts_feat['TS_No'] == ts_no, ['Disorder_Avg_TS']] = np.around(np.nanmean(disorder_ts), 6)
        ts_feat.loc[ts_feat['TS_No'] == ts_no, ['Disorder_Std_TS']] = np.around(np.nanstd(disorder_ts, ddof=1), 6)
        if num_pkt > 1:
            # DT0
            dt0s = index_pkt_ts[1:num_pkt, 0] - index_pkt_ts[0:(num_pkt-1), 0]
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['DT0_Avg_TS']] = np.around(np.nanmean(dt0s), 6)
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['DT0_Std_TS']] = np.around(np.nanstd(dt0s, ddof=1), 6)
            # DT1
            dt1s = index_pkt_ts[1:num_pkt, 1] - index_pkt_ts[0:(num_pkt-1), 1]
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['DT1_Avg_TS']] = np.around(np.nanmean(dt1s), 6)
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['DT1_Std_TS']] = np.around(np.nanstd(dt1s, ddof=1), 6)
            # DT2
            dt2s = index_pkt_ts[1:num_pkt, 2] - index_pkt_ts[0:(num_pkt-1), 2]
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['DT2_Avg_TS']] = np.around(np.nanmean(dt2s), 6)
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['DT2_Std_TS']] = np.around(np.nanstd(dt2s, ddof=1), 6)
            # PktRate
            dt0_ts = index_pkt_ts[num_pkt-1, 0] - index_pkt_ts[0, 0]
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['PktRate_T0_TS']] = np.around((num_pkt-1)/dt0_ts, 6)
            dt1_ts = index_pkt_ts[num_pkt-1, 1] - index_pkt_ts[0, 1]
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['PktRate_T1_TS']] = np.around((num_pkt-1)/dt1_ts, 6)
            dt2_ts = index_pkt_ts[num_pkt-1, 2] - index_pkt_ts[0, 2]
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['PktRate_T2_TS']] = np.around((num_pkt-1)/dt2_ts, 6)
            # BitRate
            sumlen = np.nansum(index_pkt_ts[0:(num_pkt-1), 4])
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['BitRate_T0_TS']] = np.around(sumlen/dt0_ts, 6)
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['BitRate_T1_TS']] = np.around(sumlen/dt1_ts, 6)
            ts_feat.loc[ts_feat['TS_No'] == ts_no, ['BitRate_T2_TS']] = np.around(sumlen/dt2_ts, 6)
    pbar.finish()
    ts_feat.to_csv(path_or_buf=path_output, index=False)
    return


def extract_region_feature(path_index_pkt: str, region):
    # 提取区间特征（分离）及标注
    logging.info('Region feature extracting...(' + path_index_pkt + ')')
    data = pd.read_csv(filepath_or_buffer=path_index_pkt)
    data.rename(columns={'Label': 'Label_Pkt'}, inplace=True)
    len_data = data.shape[0]
    info_pkt = ['Source', 'Destination', 'ID&Fragment', 'Protocol', 'Label_Pkt', 'TS_No']
    index_pkt = ['T1_T0', 'T2_T1', 'T2_T0', 'Length', 'Retry', 'Disorder',
                 'DT0', 'DT1', 'DT2']
    index_rgn_name = []
    for idx in index_pkt:
        index_rgn_name.extend([idx+'_Avg_', idx+'_Std_'])
    for rgn in region:
        logging.info('Region: ' + str(rgn))
        index_rgn = info_pkt + [name+str(rgn) for name in index_rgn_name]
        data_rgn = data[index_rgn].copy()
        data_rgn['Rgn_No'] = data_rgn.index - rgn + 1
        data_rgn['Rgn_TS_No'] = data_rgn['TS_No']
        data_rgn['Label'] = 0
        iLP = data_rgn.columns.tolist().index('Label_Pkt')
        iLabel = data_rgn.columns.tolist().index('Label')
        ts_nos = data_rgn['TS_No'].drop_duplicates().values.tolist()
        for ts_no in ts_nos:
            rgn_ts_no = data_rgn.loc[data_rgn['Rgn_TS_No'] == ts_no, ['Rgn_TS_No']].values
            rgn_ts_no[0:(rgn-1)] = -1
            data_rgn.loc[data_rgn['Rgn_TS_No'] == ts_no, ['Rgn_TS_No']] = rgn_ts_no
        for i in bar.progressbar(range(rgn-1, len_data)):
            if data_rgn.iloc[(i-rgn+1):(i+1), iLP].sum() > 0:
                data_rgn.iloc[i, iLabel] = 1
        data_rgn.to_csv(path_or_buf=os.path.dirname(path_index_pkt) + '/region_feature_' + str(rgn), index=False)
    return


def result_region_to_pkt(path_region_feature: str,
                         path_result_region: str,
                         path_pkt_feature: str,
                         path_output: str):
    # 识别结果转换：区间->数据包
    return


def combine_ts_feature(dir_cap_list: list, path_output: str):
    # 合并样本分片特征
    logging.info('TS feature combine.')
    alldata = []
    columns = []
    for _, dir_cap in bar.progressbar(enumerate(dir_cap_list)):
        path_ts_feat_label = os.path.join(dir_cap, 'ts_feature_label')
        data = pd.read_csv(filepath_or_buffer=path_ts_feat_label)
        data.insert(0, 'Sample_Dir', dir_cap)
        columns = data.columns.tolist()
        data.replace(np.inf, np.nan, inplace=True)
        data.dropna(inplace=True)
        alldata.extend(data.values.tolist())
    alldata = pd.DataFrame(data=alldata, columns=columns)
    alldata.to_csv(path_or_buf=path_output, index=False)
    return


def combine_region_feature(dirs_ap: list, region: int, path_output: str):
    # 合并样本区间特征
    logging.info('Region feature combining...(region=' + str(region) + ')')
    alldata = []
    columns = []
    for _, dir_ap in bar.progressbar(enumerate(dirs_ap)):
        path_ts_feat_label = os.path.join(dir_ap, 'region_feature_' + str(region))
        data = pd.read_csv(filepath_or_buffer=path_ts_feat_label)
        data.insert(0, 'Sample_Dir', dir_ap)
        columns = data.columns.tolist()
        data.replace(np.inf, np.nan, inplace=True)
        data.dropna(inplace=True)
        data = data[data['Rgn_No'] >= 0]
        alldata.extend(data.values.tolist())
    alldata = pd.DataFrame(data=alldata, columns=columns)
    alldata.to_csv(path_or_buf=path_output, index=False)
    return


def combine_result(paths_result: list, path_output: str):
    result = pd.DataFrame()
    for i, path_result in enumerate(paths_result):
        logging.info('Result combining... (' + path_result + ')')
        res = pd.read_csv(filepath_or_buffer=path_result, usecols=['Predict'])  # , header=0, names=['Predict' + str(i)])
        result = pd.concat(objs=[result, res], axis=1)
    # result['Predict'] = result.apply(lambda x: x.sum(), axis=1)
    result['PredictAvg'] = result.mean(axis=1)
    result['PredictAvg'].to_csv(path_or_buf=path_output, index=False)
    return result


def result_disperse(path_result: str):
    result = pd.read_csv(filepath_or_buffer=path_result)
    for threshold in np.arange(0, 1.1, 0.1):
        result.loc[result['PredictAvg'] < threshold, 'Predict'] = 0
        result.loc[result['PredictAvg'] >= threshold, 'Predict'] = 1
        path_output = path_result + '_' + str(threshold)
        result['Predict'].to_csv(path_or_buf=path_output, index=False)


def result_ts_pkt_info(path_result_ts_pkt: str):
    data = pd.read_csv(filepath_or_buffer=path_result_ts_pkt)
    path_info = os.path.dirname(path_result_ts_pkt) + '/info_statistic.txt'
    file = open(path_info, 'w')
    file.write('Label,Predict,Number\n')
    file.write('0,0,' + str(len(data[(data['Label'] == 0) & (data['Predict'] == 0)])) + '\n')
    file.write('0,1,' + str(len(data[(data['Label'] == 0) & (data['Predict'] == 1)])) + '\n')
    file.write('1,0,' + str(len(data[(data['Label'] == 1) & (data['Predict'] == 0)])) + '\n')
    file.write('1,1,' + str(len(data[(data['Label'] == 1) & (data['Predict'] == 1)])) + '\n')
    file.close()


def result_ts_to_pkt(path_ts_feature: str,
                     path_result_ts: str,
                     path_pkt_feature: str,
                     path_output: str,
                     path_result_pkt=None):
    logging.info('Converting result from ts to pkt.(' + str(path_result_pkt) + ')')
    ts_data = pd.read_csv(filepath_or_buffer=path_ts_feature)
    data = pd.read_csv(filepath_or_buffer=path_result_ts)
    if not ts_data.shape[0] == data.shape[0]:
        logging.info('TS number mismatch!')
        return

    ts_data = ts_data.join(data)
    # dir_cap_list = ts_data['Sample_Dir'].drop_duplicates().values.tolist()
    pkt_feature = pd.read_csv(filepath_or_buffer=path_pkt_feature,
                              usecols=['Sample_Dir', 'TS_No',
                                       'Source', 'Destination', 'ID&Fragment', 'Protocol',
                                       'Label'])
    if path_result_pkt is None:
        pkt_feature['Predict'] = 1
    else:
        result_pkt = pd.read_csv(filepath_or_buffer=path_result_pkt, header=0, names=['Predict'])
        pkt_feature = pkt_feature.join(result_pkt)

    result = []
    for idx, row in ts_data.iterrows():
        ts_dir = row['Sample_Dir']
        smp_dir = ts_dir + '/ap_log_' + os.path.basename(ts_dir)
        ts_no = row['TS_No']
        ts_predict = row['Predict']
        ts_pkt_data = pkt_feature[(pkt_feature['Sample_Dir'] == smp_dir) & (pkt_feature['TS_No'] == ts_no)].copy()
        if ts_predict == 0:
            ts_pkt_data['Predict'] = 0
        result.extend(ts_pkt_data.loc[:, ['Label', 'Predict']].values.tolist())

    # result = []
    # for dir_cap in dir_cap_list:
    #     ts_list = ts_data[ts_data['Sample_Dir'] == dir_cap]
    #     # path_data_pkt = dir_cap + '/ap_log_' + os.path.split(dir_cap)[1] + '/index_pkt_ts_label'
    #     # data_pkt = pd.read_csv(filepath_or_buffer=path_data_pkt)
    #     for idx, row in ts_list.iterrows():
    #         ts_no = row['TS_No']
    #         ts_predict = row['Predict']
    #         if ts_predict == 0:
    #             pkt_feature.loc[(pkt_feature['Sample_Dir'] == dir_cap) & (pkt_feature['TS_No'] == ts_no), ['Predict']] = 0
        # if path_result_pkt is None:
        #     for idx, row in ts_list.iterrows():
        #         ts_no = row['TS_No']
        #         ts_predict = row['Predict']
        #         if ts_predict:
        #             pkt_feature[pkt_feature['Sample_Dir'] == dir_cap & pkt_feature['TS_No'] == ts_no] = 1
        #         data_pkt_ts = data_pkt[data_pkt['TS_No'] == ts_no]
        #         data_pkt_ts.insert(1, 'Predict', ts_predict)
        #         result.extend(data_pkt_ts.loc[:, ['Label', 'Predict']].values.tolist())
        # else:
        #     result_pkt = pd.read_csv(filepath_or_buffer=path_result_pkt, header=0, names=['Predict'])
        #     data_pkt = data_pkt.join(result_pkt)
        #     for idx, row in ts_list.iterrows():
        #         ts_no = row['TS_No']
        #         ts_predict = row['Predict']
        #         data_pkt_ts = data_pkt[data_pkt['TS_No'] == ts_no].copy()
        #         if ts_predict == 0:
        #             data_pkt_ts['Predict'] = ts_predict
        #         result.extend(data_pkt_ts.loc[:, ['Label', 'Predict']].values.tolist())

    if path_result_pkt is None:
        result_name = os.path.split(path_result_ts)[1]
    else:
        result_name = os.path.split(path_result_pkt)[1]
    dir_output = os.path.dirname(os.path.dirname(path_result_ts)) + '/result_predict_pkt'
    os.makedirs(name=dir_output, exist_ok=True)
    path_output = dir_output + '/' + result_name
    result = pd.DataFrame(data=result, columns=['Label', 'Predict'])
    result.to_csv(path_or_buf=path_output, index=False)
    # pkt_feature.to_csv(path_or_buf=path_output, index=False, columns=['Label', 'Predict'])

    dir_output = os.path.dirname(path_output)
    path_output = dir_output + '/info.txt'
    if not os.path.exists(path_output):
        file = open(path_output, 'w')
        file.write('TS Feature: ' + path_ts_feature + '\n')
        file.write('TS Result: ' + path_result_ts + '\n')
        file.close()
    path_output = dir_output + '/evaluation.csv'
    evaluation = evaluate(result['Label'].values, result['Predict'].values)
    # evaluation = evaluate(pkt_feature['Label'].values, pkt_feature['Predict'].values)
    evaluation = [result_name] + list(evaluation)
    if not os.path.exists(path_output):
        file = open(path_output, 'a')
        column = ['Name', 'Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'TPR', 'Thresholds']
        file.write(','.join([x for x in column]) + '\n')
    else:
        file = open(path_output, 'a')
    file.write(','.join([str(x) for x in evaluation]) + '\n')
    file.close()
    print(evaluation)


def mrmr_analysis(data, path_output: str):
    if type(data) is str:
        # logging.info('mRMR analysis.(' + data + ')')
        data = pd.read_csv(filepath_or_buffer=data)
    elif type(data) is not pd.DataFrame:
        print('Data type invalid!')
        return
    data.drop(columns=['Source', 'Destination', 'ID&Fragment', 'Protocol', 'T0', 'T1', 'T2', 'TS_No'],
              inplace=True)
    iLabel = data.columns.tolist().index('Label')
    label = data.iloc[:, iLabel:(iLabel + 1)]
    data.drop(columns=['Label'], inplace=True)
    # data = data.iloc[:, 0:3]
    # print(data)
    res = mrmr.mrmr_ensemble(features=data,
                             targets=label,
                             solution_length=data.shape[1])
    res = res.values.tolist()[0][0]
    file = open(path_output, 'w')
    file.write(','.join([x for x in res]))
    file.write('\n')
    file.write(','.join([str(res.index(x) + 1) for x in res]))
    file.close()
    print(res)


def mrmr_statistic():
    features = pd.read_csv(filepath_or_buffer='data/data_video/v1/v1_1/v1_1_1/ap_log_v1_1_1/index_ts_label')
    features.drop(columns=['Source', 'Destination', 'ID&Fragment', 'Protocol', 'T0', 'T1', 'T2', 'TS_No', 'Label'],
                  inplace=True)
    features = features.columns.tolist()
    features = pd.DataFrame(columns=features)
    objs = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13]   # 1, 5, 6, 7, 8, 9, 10, 11, 12, 13
    for obj in objs:
        path_mrmr = 'data/data_video/vall/mrmr/mrmr_v' + str(obj) + '_0&1_1&2.txt'
        mrmr = pd.read_csv(filepath_or_buffer=path_mrmr)
        mrmr.index = ['v' + str(obj)]
        features = pd.concat([features, mrmr])
    features.T.to_csv(path_or_buf='data/data_video/vall/mrmr/mrmr_statistic.csv')
    # print(type(features), features)


def feature_selection():
    path_mrmr = 'data/data_video/vall/mrmr/mrmr_statistic.csv'
    mrmr = pd.read_csv(filepath_or_buffer=path_mrmr, index_col=0)
    mrmr['avg'] = mrmr.mean(axis=1)
    features = mrmr[mrmr['avg'] < 50].index.values.tolist()
    path_output = os.path.join(os.path.dirname(path_mrmr), 'feature_selection.txt')
    file = open(path_output, 'w')
    file.write(','.join([x for x in features]))
    file.close()
    print(features)


def data_selection(dir_ap_list: list, path_output: str, ts_on=True):
    logging.info('Data selection.')
    path_feat_select = 'data/data_video/vall/mrmr/feature_selection.txt'
    features = pd.read_csv(filepath_or_buffer=path_feat_select, header=None).values.tolist()[0]
    columns = ['TS_No', 'Source', 'Destination', 'ID&Fragment', 'Protocol'] + features + ['Label']
    data = pd.DataFrame()
    for dir_ap in dir_ap_list:
        logging.info('Data selecting...(' + dir_ap + ')')
        path_index_ts = os.path.join(dir_ap, 'index_ts_label')
        pkt_data = pd.read_csv(filepath_or_buffer=path_index_ts)[columns]
        if ts_on:
            path_ts_label = os.path.join(os.path.dirname(dir_ap), 'ts_feature_label')
            ts_data = pd.read_csv(filepath_or_buffer=path_ts_label)[['TS_No', 'Label']]
            ts_data = ts_data[ts_data['Label'] == 1]['TS_No'].values.tolist()
            pkt_data = pkt_data[pkt_data['TS_No'].isin(ts_data)]
        pkt_data.insert(loc=0, column='Sample_Dir', value=dir_ap)
        data = pd.concat([data, pkt_data])
    data.to_csv(path_or_buf=path_output, index=False)


def result_process():
    path_ts_feature = 'data/data_video/vall/ts_feature_label_v11_v13_0&1_1&2'
    path_pkt_feature = 'data/data_video/vall/index_ts_label0_v11_v13_0&1_1&2'

    # path_result_ts = 'train/15_ts/SVM/20220221185946/result_predict/result_rbf_13_0.017_3'
    # name_output = 's20220221185946p13p0017'

    path_result_ts = 'train/15_ts/RF/20220221160708/result_predict/result_31_None_2_1_auto'
    name_output = 'r20220221160708p31'

    # path_result_ts = 'train/15_ts/AdaBoost/20220221155100/result_predict/result_73_0.9499999999999997_None_2_1_None'
    # name_output = 'a20220221155100p73p095'

    # result_ts_pkt_info(path_result_ts_pkt='train/17_ts_pkt/direct/result_predict_pkt/' + name_output + '/' + os.path.basename(path_result_ts))

    # result_ts_to_pkt(path_ts_feature='data/data_video/vall/ts_feature_label_v11_v13_0&1_1&2',
    #                  path_pkt_feature='data/data_video/vall/index_ts_label0_v11_v13_0&1_1&2',
    #                  path_result_ts=path_result_ts,
    #                  path_result_pkt=None,
    #                  dir_output='train/17_ts_pkt/direct/result_predict_pkt/' + name_output)

    dir_result = 'train/17_ts_pkt/AdaBoost/20220309213350'
    dir_result_pkt = dir_result + '/result_predict'
    filenames = os.listdir(dir_result_pkt)
    for filename in filenames:
        path_result_pkt = os.path.join(dir_result_pkt, filename)
        if path_result_pkt is None:
            result_name = os.path.split(path_result_ts)[1]
        else:
            result_name = filename
        dir_output = dir_result + '/result_predict_pkt/' + name_output
        os.makedirs(name=dir_output, exist_ok=True)
        path_output = dir_output + '/' + result_name
        if not os.path.exists(path_output):
            result_ts_to_pkt(path_ts_feature=path_ts_feature,
                             path_pkt_feature=path_pkt_feature,
                             path_result_ts=path_result_ts,
                             path_result_pkt=path_result_pkt,
                             path_output=path_output)
    # dir_res = 'train/17_ts_pkt/SVM'
    # paths_result = [dir_res + '/v1/20220225134439/result_predict/result_rbf_0.01_0.1_3',
    #                 dir_res + '/v5/20220227013403/result_predict/result_rbf_0.01_0.001_3',
    #                 dir_res + '/v6/20220227201949/result_predict/result_rbf_100.0_1.0_3',
    #                 dir_res + '/v7/20220227013504/result_predict/result_rbf_100.0_0.001_3',
    #                 dir_res + '/v8/20220227013732/result_predict/result_rbf_10.0_1.0_3',
    #                 dir_res + '/v9/20220227202118/result_predict/result_rbf_10.0_0.1_3',
    #                 dir_res + '/v10/20220227202258/result_predict/result_rbf_0.1_0.1_3']
    # path_results = dir_res + '/combine/result_v1_v10'
    # dir_res = 'train/17_ts_pkt'
    # paths_result = [dir_res + '/SVM/combine/result_predict_pkt/r20220221161311p28th0.2/result_v1_v10_0.2',
    #                 dir_res + '/RF/20220225134952/result_predict/result_22_None_2_1_auto',
    #                 dir_res + '/AdaBoost/20220225135324/result_predict/result_24_1.0_None_2_1_None']
    # path_results = dir_res + '/combine/result_svm_rf_adaboost'
    # combine_result(paths_result=paths_result,
    #                path_output=path_results)
    # result_disperse(path_results)
    # threshold = '0.8'
    # result_ts_to_pkt(path_ts_feature='data/data_video/vall/ts_feature_label_v11_v13_0&1_1&2',
    #                  path_result_ts='train/15_ts/RF/20220221161311/result_predict/result_28_None_2_1_auto',
    #                  path_result_pkt=path_results + '_' + threshold,
    #                  dir_output=dir_res + '/combine/result_predict_pkt/r20220221161311p28th' + threshold)


def get_path(path_params: list):
    if len(sys.argv) >= 4:
        objs = [int(sys.argv[1])]
        envs = path_params[1] if sys.argv[2] == 'n' else [int(sys.argv[2])]
        idxs = path_params[2] if sys.argv[3] == 'n' else [int(sys.argv[3])]
    else:
        objs = path_params[0]  # 对象编号: 1, 5, 6, 7, 8, 9, 10, 11, 12, 13
        envs = path_params[1]  # 环境编号: 0, 1, 2, 3
        idxs = path_params[2]  # 样本编号: 1, 2
    dir_cap_list = []
    dir_ap_list = []
    for obj in objs:
        for env in envs:
            for idx in idxs:
                root = 'data/data_video'
                dtype = 'v'
                name_to = dtype + str(obj)
                name_toe = name_to + '_' + str(env)
                name_toei = name_toe + '_' + str(idx)
                dir_cap = root + '/' + name_to + '/' + name_toe + '/' + name_toei
                if os.path.exists(dir_cap):
                    dir_ap = dir_cap + '/ap_log_' + name_toei
                    dir_cap_list.append(dir_cap)
                    dir_ap_list.append((dir_ap))
    return dir_ap_list, dir_cap_list, [objs, envs, idxs]


def get_name(numbers: list):
    if len(numbers[0]) > 1:
        name_objs = 'v' + str(min(numbers[0])) + '_v' + str(max(numbers[0]))
    else:
        name_objs = 'v' + str(numbers[0][0])
    if len(numbers[1]) > 1:
        name_envs = '&'.join(str(n) for n in numbers[1])
    else:
        name_envs = str(numbers[1][0])
    if len(numbers[2]) > 1:
        name_idxs = '&'.join(str(n) for n in numbers[2])
    else:
        name_idxs = str(numbers[2][0])
    return name_objs + '_' + name_envs + '_' + name_idxs


def data_processing(path_params: list, process_params: list, other_params: list):
    # sys.argv说明:
    # 0: 文件名
    # 1-3: 样本参数(1: 视频编号; 2: 环境编号; 3: 样本编号)
    # 4-x: 操作参数(4: 操作数量; 5-x: 操作编号)
    # (x+1)-y: 额外参数
    dir_ap_list, dir_cap_list, numbers = get_path(path_params)
    process_list = process_params[1:]
    other_list = other_params
    if len(sys.argv) > 4:
        process_list = list(map(int, sys.argv[5:(5+int(sys.argv[4]))]))
        if len(sys.argv) > 5 + int(sys.argv[4]):
            other_list = list(map(int, sys.argv[(5+int(sys.argv[4])):]))
    for process in process_list:
        # 提取分片信息
        if int(process) == 1:
            for dir_cap in dir_cap_list:
                extract_ts_stm_info(dir_cap)
        # 提取分片数据包信息
        if int(process) == 2:
            for dir_ap in dir_ap_list:
                extract_ts_pkt_info(dir_ap)
        # 合并分片样本数据
        if int(process) == 3:
            name = get_name(numbers)
            path_output = 'data/data_video/vall/ts_feature_label_' + name
            combine_ts_feature(dir_cap_list, path_output)
        # 提取数据包区间特征
        if int(process) == 4:
            for dir_ap in dir_ap_list:
                path_index_pkt = os.path.join(dir_ap, 'index_pkt_ts_label')
                path_output = os.path.join(dir_ap, 'index_ts_label')
                extract_index_pkt_region(path_index_pkt=path_index_pkt,
                                         path_output=path_output,
                                         region=REGION)
        # mRMR分析
        if int(process) == 5:
            index_ts_label = pd.DataFrame()
            for dir_ap in dir_ap_list:
                path_index_ts = os.path.join(dir_ap, 'index_ts_label')
                data = pd.read_csv(filepath_or_buffer=path_index_ts)
                index_ts_label = pd.concat([index_ts_label, data])
            name = get_name(numbers)
            logging.info('mRMR analysis.(' + name + ')')
            path_output = 'data/data_video/vall/mrmr/mrmr_' + name + '.txt'
            mrmr_analysis(data=index_ts_label, path_output=path_output)
        # 数据筛选
        if int(process) == 6:
            name = get_name(numbers)
            ts_on = True
            path_output = 'data/data_video/vall/index_ts_label' + str(int(ts_on)) + '_' + name
            data_selection(dir_ap_list=dir_ap_list, path_output=path_output, ts_on=ts_on)
        # 包特征新增：乱序
        if int(process) == 7:
            for dir_ap in dir_ap_list:
                extract_index_pkt_add_disorder(path_index_pkt=dir_ap + '/index_ts_label',
                                               path_output=dir_ap + '/index_ts_label',
                                               region=REGION)
        # 包特征新增：包间节点时间差
        if int(process) == 8:
            for dir_ap in dir_ap_list:
                extract_index_pkt_add_dt(path_index_pkt=dir_ap + '/index_ts_label',
                                         path_output=dir_ap + '/index_ts_label',
                                         region=REGION)
        # 分片特征新增
        if int(process) == 9:
            for dir_ap in dir_ap_list:
                extract_ts_pkt_add(path_index_pkt=dir_ap + '/index_ts_label',
                                   path_ts_feature=os.path.dirname(dir_ap) + '/ts_feature_label',
                                   path_output=os.path.dirname(dir_ap) + '/ts_feature_label')
        # 区间特征分离
        if int(process) == 10:
            for dir_ap in dir_ap_list:
                extract_region_feature(path_index_pkt=dir_ap + '/index_ts_label',
                                       region=REGION)
        # 样本区间特征合并
        if int(process) == 11:
            for rgn in other_list:
                name = get_name(numbers)
                path_output = 'data/data_video/vall/region_feature_' + str(rgn) + '_' + name
                combine_region_feature(dirs_ap=dir_ap_list,
                                       region=rgn,
                                       path_output=path_output)
    return


if __name__ == '__main__':
    objs = [11, 12, 13]  # 对象编号: 1, 5, 6, 7, 8, 9, 10, 11, 12, 13
    envs = [0, 1]  # 环境编号: 0, 1, 2, 3
    idxs = [1, 2]  # 样本编号: 1, 2
    pcss = [11]  # 处理编号: 1, 2, 3, 4, 5, 6
    oths = [5]  # 额外参数: region: 5, 10, 20, 50, 100, 200, 500, 1000, 2000
    path_params = [objs, envs, idxs]
    process_params = [len(pcss)] + pcss
    other_params = oths
    data_processing(path_params=path_params,
                    process_params=process_params,
                    other_params=other_params)

    # 测试
    # feature_selection()
    # extract_index_pkt_add_disorder(path_index_pkt='data/data_video/v1/v1_1/v1_1_1/ap_log_v1_1_1/test_index_ts_label',
    #                       path_output='data/data_video/v1/v1_1/v1_1_1/ap_log_v1_1_1/test_index_ts_label',
    #                       region=REGION)
    # extract_index_pkt_add_dt(path_index_pkt='data/data_video/v1/v1_1/v1_1_1/ap_log_v1_1_1/index_pkt_ts_label',
    #                          path_output='data/data_video/v1/v1_1/v1_1_1/ap_log_v1_1_1/test_index',
    #                          region=REGION)
    # extract_ts_pkt_add(path_index_pkt='data/data_video/v1/v1_1/v1_1_1/ap_log_v1_1_1/index_ts_label',
    #                    path_ts_feature='data/data_video/v1/v1_1/v1_1_1/ts_feature_label',
    #                    path_output='data/data_video/v1/v1_1/v1_1_1/test_ts_feature_label.csv')
    # extract_region_feature(path_index_pkt='data/data_video/v1/v1_1/v1_1_1/ap_log_v1_1_1/index_ts_label',
    #                        region=[5])

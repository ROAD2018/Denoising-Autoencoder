import os, re, glob, csv, datetime
from adtk.visualization import plot
from adtk.data import validate_series
from adtk.detector import InterQuartileRangeAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import netCDF4 as nc
from collections import Counter
from soil_moisture_analysis.mining.constants import *
from soil_moisture_analysis.merge_all import merge_data


def isNaN(num):
    return num != num or num == 0


def get_long_seq_tup(d1):
    for tup in list(zip(d1['seqs'], d1['mls_start'], d1['mls_end'])):
        if d1['long_seq'] == tup[0]:
            return tup


def index_for_gaps(d1, seq_start, seq_end):
    start_gap = None
    end_gap = None
    for i, tup in enumerate(list(zip(d1['gaps'], d1['starts'], d1['ends']))):
        if seq_start == tup[2]:
            start_gap = i
        if seq_end == tup[1]:
            end_gap = i
        if start_gap and end_gap:
            return start_gap, end_gap
    return start_gap, end_gap


def get_modified_start_date_for_seq(seq_gaps, start_gap):
    start_seq = None
    for i in reversed(range(start_gap + 1)):
        if seq_gaps[i][0] < 45:
            if i > 0:
                start_seq = seq_gaps[i - 1][2]
            else:
                start_seq = seq_gaps[i][1]
        else:
            return start_seq
    return start_seq


def get_modified_seq_end_date(seq_gaps, end_gap):
    end_seq = None
    for i in range(end_gap, len(seq_gaps)):
        if seq_gaps[i][0] < 45:
            if i < len(seq_gaps) - 1:
                end_seq = seq_gaps[i + 1][1]
            else:
                end_seq = seq_gaps[i][2]
        else:
            return end_seq
    return end_seq


def get_dates_filter(s1, dates):
    max_val = 0
    longest_gap = 0
    long_seq = 0
    max_long_seq = 0
    no_nulls = True
    gaps = []
    starts = []
    ends = []
    mls_start = []
    mls_end = []
    seqs = []
    for val, date in zip(s1, dates):
        if isNaN(val):
            no_nulls = False
            if max_val == 0:
                starts.append(date)
            max_val += 1
            if max_long_seq < long_seq:
                max_long_seq = long_seq
            if long_seq > 0:
                seqs.append(long_seq)
                mls_end.append(date)
            long_seq = 0
        else:
            if long_seq == 0:
                mls_start.append(date)
            long_seq += 1
            if longest_gap < max_val:
                longest_gap = max_val
            if max_val > 0:
                gaps.append(max_val)
                ends.append(date)
            max_val = 0
    if max_long_seq < long_seq:
        max_long_seq = long_seq
    if longest_gap < max_val:
        longest_gap = max_val
    if max_val > 0:
        ends.append(dates[-1])
        gaps.append(max_val)
    if no_nulls:
        max_long_seq = s1.shape[0]
    if long_seq > 0:
        mls_end.append(dates[-1])
        seqs.append(long_seq)
    d = dict()
    d['gaps'] = gaps
    d['starts'] = starts
    d['ends'] = ends
    d['mls_start'] = mls_start
    d['mls_end'] = mls_end
    d['seqs'] = seqs
    d['long_seq'] = max_long_seq
    d['longest_gap'] = longest_gap
    return d


def find_longest_seq_count(s1):
    max_val = 0
    longest_gap = 0
    long_seq = 0
    max_long_seq = 0
    num_gaps = 0
    no_nulls = True
    for val in s1:
        if isNaN(val):
            if max_val == 0:
                num_gaps += 1
            no_nulls = False
            max_val += 1
            if max_long_seq < long_seq:
                max_long_seq = long_seq
            long_seq = 0
        else:
            long_seq += 1
            if longest_gap < max_val:
                longest_gap = max_val
            max_val = 0
    if max_long_seq < long_seq:
        max_long_seq = long_seq
    if longest_gap < max_val:
        longest_gap = max_val
    if no_nulls:
        max_long_seq = s1.shape[0]
    return max_long_seq, longest_gap, num_gaps


def get_gaps_sequences_plots(filtered_df):
    ## retrieving the start and end dates for each node
    try:
        res_tup = []
        list_cols = list(filtered_df.columns)
        list_cols.remove('date')
        for col in list_cols:
            sample = filtered_df[col]
            sample = sample.dropna()
            if not sample.empty and sample.shape[0] > 10:
                sample.index = pd.to_datetime(sample.index)
                s1 = sample.resample('H').mean()
                d1 = get_dates_filter(s1, list(filtered_df.index))
                seq_tuple = list(zip(d1['seqs'], d1['mls_start'], d1['mls_end']))
                seq_gaps = list(zip(d1['gaps'], d1['starts'], d1['ends']))
                if seq_tuple:
                    _, seq_start, seq_end = get_long_seq_tup(d1)
                    start_gap, end_gap = index_for_gaps(d1, seq_start, seq_end)
                    if start_gap:
                        start_seq_modified = get_modified_start_date_for_seq(seq_gaps, start_gap)
                        if start_seq_modified:
                            seq_start = start_seq_modified
                    if end_gap:
                        end_seq_modified = get_modified_seq_end_date(seq_gaps, end_gap)
                        if end_seq_modified:
                            seq_end = end_seq_modified
                    res_tup.append((col, seq_start, seq_end))
                    plt.figure(figsize=(35, 10))
                    fig = plt.gcf()
                    # x = [str(i) for i in range(len(d1['gaps']))]
                    plt.bar(d1['starts'], d1['gaps'], align='center', width=0.3)
                    plt.title("Histogram of gap sequences of {}".format(col))
                    plt.xlabel('gap start date')
                    plt.ylabel('gap length')
                    fig.autofmt_xdate()
                    fig.savefig(gap_hist_path + col + ".png")
                    plt.close('all')
        resdf = pd.DataFrame(res_tup, columns=['node', 'start_date', 'end_date'])
        resdf.to_csv(gap_hist_path + "sequence_date.csv", index=None)
    except Exception as e:
        print(e)


def get_missing_stats_data(filtered_data):
    ## plotting the outliers for each node and saving the stats of gaps, filled sequences to a csv file
    s1 = None
    outlier_list = []
    try:
        list_dict = []
        list_cols = list(filtered_data.columns)
        list_cols.remove('date')
        for col in list_cols:
            d1 = {}
            sample = filtered_data[col]
            sample = sample.dropna()
            if not sample.empty and sample.shape[0] > 10:
                sample.index = pd.to_datetime(sample.index)
                s1 = sample.resample('H').mean()
                d1['site'] = col.split("n=")[0]
                node_id = col.split("n=")[1].split("_")[0]
                d1['depth'] = col.split("n=")[1].split("_d=")[1]

                s = validate_series(s1)
                iqr_ad = InterQuartileRangeAD(c=1.5)
                anomalies = iqr_ad.fit_detect(s)
                plot(s, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red',
                     anomaly_tag="marker")
                plt.figure(figsize=(20, 10))
                fig = plt.gcf()
                plt.xlabel('date')
                plt.ylabel('soil moisture')
                plt.title("SITE = " + d1['site'] + ",  NODE_ID = " + node_id + ",  DEPTH = " + d1['depth'])
                fig.savefig(hourly_visualisations + col + ".png")
                plt.close('all')
                d1['node_id'] = node_id
                d1['total_rows'] = s1.shape[0]
                d1['start'] = s1.index[0].date()
                d1['end'] = s1.index[-1].date()
                max_long_seq, longest_gap, num_gaps = find_longest_seq_count(s1)
                d1['nulls'] = s1.isnull().sum()
                d1['filled_percentage'] = 1 - d1['nulls'] / float(d1['total_rows'])
                d1['longest_gap'] = longest_gap
                d1['longest_seq'] = max_long_seq
                d1['num_gaps'] = num_gaps
                list_dict.append(d1)
        with open(hourly_node_wise_data_path, 'w', newline="") as f:
            writer = csv.DictWriter(f, ['site', 'node_id', 'depth', 'start', 'end', 'nulls', 'total_rows',
                                        'filled_percentage', 'longest_gap', 'longest_seq', 'num_gaps'])
            writer.writeheader()
            writer.writerows(list_dict)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    merge_data()
    df = pd.read_csv(merge_data_path, sep=";")
    filtered = df.sort_values(by=['date'])
    filtered.index = filtered.date
    filtered = filtered.replace(0, np.nan)
    get_missing_stats_data(filtered)
    get_gaps_sequences_plots(filtered)

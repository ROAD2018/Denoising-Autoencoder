import pandas as pd
import re
import numpy as np

def isNaN(num):
    return num != num

def get_dates_filter(s1,dates):
    max_val = 0
    longest_gap = 0
    long_seq = 0
    max_long_seq = 0
    no_nulls=True
    gaps = []
    starts = []
    ends = []
    mls_start = []
    mls_end = []
    seqs=[]
    for val,date in zip(s1,dates):
        if isNaN(val):
            no_nulls=False
            if max_val ==0:
                starts.append(date)
            max_val+=1
            if max_long_seq<long_seq:
                max_long_seq = long_seq
            if long_seq > 0:
                seqs.append(long_seq)
                mls_end.append(date)
            long_seq = 0
        else:
            if long_seq ==0:
                mls_start.append(date)
            long_seq+=1
            if longest_gap<max_val:
                longest_gap = max_val
            if max_val > 0:
                gaps.append(max_val)
                ends.append(date)
            max_val = 0
    if max_long_seq<long_seq:
        max_long_seq = long_seq
    if longest_gap<max_val:
        longest_gap = max_val
    if max_val>0:
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


def plot_hist(s1,col,col_dict):
    d1 = get_dates_filter(s1[col], list(s1.index))
    year = s1.index[0].year
    seq_tuple = list(zip(d1['seqs'], d1['mls_start'], d1['mls_end']))
    seq_gaps = list(zip(d1['gaps'], d1['starts'], d1['ends']))
    if seq_tuple:
        if not d1['gaps'] or (d1['gaps'] and max(d1['gaps'])==0):
            if s1.dropna().shape[0]>=360*24:
                max_val = 0
                if d1['gaps']:
                    max_val  = max(d1['gaps'])
                # print("{},{},{},{}".format(col,year,s1.dropna().shape[0],max_val))
                col_dict.setdefault(col,{})
                col_dict[col].setdefault('years_dict',[])
                col_dict[col].setdefault('years',[])
                year_dict ={}
                year_dict['year'] = year
                # year_dict['max_val'] =max_val
                year_dict['data_size'] = s1.dropna().shape[0]
                col_dict[col]['years_dict'].append(year_dict)
                col_dict[col]['years'].append(year)


def get_years_per_column(ameri_data):
    col_dict= {}
    for col in ameri_data.columns:
        sample = ameri_data[col]
        # sample[sample < 1] = np.nan
        sample = sample.dropna()
        if not sample.empty and sample.shape[0] > 10:
            sample.index = pd.to_datetime(sample.index)
            s1 = sample.resample('H').mean()
            x = pd.DataFrame(s1)
            x.index = pd.to_datetime(x.index)
            x['time'] = x.index
            x.groupby(pd.Grouper(key='time', freq='Y')).apply(lambda x: plot_hist(x, col,col_dict))
    return col_dict
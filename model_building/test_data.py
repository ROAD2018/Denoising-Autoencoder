"""
Different input data sets to validate the trained models
"""
import torch
import pandas as pd
from data_preprocessing.filter_ameri_files import add_noise,get_sequence
from model_building import batch_size,seq_len


def process_test(test):
    test = test.fillna(0)
    test_ip = get_sequence(test, batch_size, seq_len)
    test_ip = torch.tensor(test_ip)
    noisy_test = get_sequence(add_noise(torch.tensor(test)), batch_size, seq_len)
    test_seq = zip(test_ip, noisy_test)
    return test,test_seq


def kendall():
    df = pd.read_csv("/home/sonakshireddy/Documents/soil/soil/pred_exp/merge_all_result_hourly.csv", sep=";")
    df.index = pd.to_datetime(df['date'])
    df = df.sort_index()
    test = df[(df.index.year == 2017) & (df.index.month.isin(range(7, 9)))]['Kendall_AZ_n=1408_d=5']
    return process_test(test),'kendall'


def scd():
    scd_data_path = "/home/sonakshireddy/Documents/soil/merged_data_SCD.csv"
    scd = pd.read_csv(scd_data_path)
    scd.date = pd.to_datetime(scd.date)
    del scd['Unnamed: 0']
    scd.index = scd.date
    scd = scd[['SWC']]
    scd = scd.fillna(0)
    test = scd[scd.index.year.isin(range(2009, 2011))]['SWC']
    return process_test(test),'scd'


def fmf():
    fmf = pd.read_csv("/home/sonakshireddy/Documents/soil/ameri_fmf.csv")
    fmf.date = pd.to_datetime(fmf.date)
    # del fwf['Unnamed: 0']
    del fmf['TIMESTAMP_START']
    fmf.index = fmf.date
    test = fmf[fmf.index.year == 2010]['SWC_1_1_1']
    return process_test(test),'fmf'


def bar():
    bar = pd.read_csv("/home/sonakshireddy/Documents/soil/ameri_BAR.csv")
    bar.date = pd.to_datetime(bar.date)
    del bar['TIMESTAMP_START']
    bar.index = bar.date
    bar = bar[['SWC_1_1_1']]
    bar = bar.fillna(0)
    test = bar[bar.index.year.isin(range(2007, 2010))]['SWC_1_1_1']
    return process_test(test),'bar'


def tonzi():
    filtered = pd.read_csv("/home/sonakshireddy/Documents/soil/merged_data_tera.csv")
    filtered.date = pd.to_datetime(filtered.date)
    del filtered['Unnamed: 0']
    filtered.index = filtered.date
    test = filtered[filtered.index.year == 2015]['TonziRanch_CAn=408_d=5']
    return process_test(test),'tonzi'



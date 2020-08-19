import json,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
LARGE_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)
from data_preprocessing import dir_name
os.chdir(dir_name)
with open("/home/sonakshireddy/Downloads/zip_files/file_columns_years.json",'r') as f:
    files = json.load(f)
    files_arr = files['files']
    for file_tup in files_arr:
        df = pd.read_csv(file_tup[0],skiprows=2)
        df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
        df['TIMESTAMP_END'] = pd.to_datetime(df['TIMESTAMP_END'], format='%Y%m%d%H%M')
        df.index = df['TIMESTAMP_START']
        del df['TIMESTAMP_START']
        del df['TIMESTAMP_END']
        df = df[[file_tup[1]]]
        df = df[df[file_tup[1]].index.year.isin(file_tup[2])]
        df = df.replace(-9999.0, np.nan)
        df = df.dropna(axis=0, how='all')
        df = df[df >= 0]
        plt.figure(figsize=(20, 10))
        fig = plt.gcf()
        plt.plot(df.index,df[file_tup[1]])
        fig.savefig( file_tup[2]+ ".png")
import os,re,glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import netCDF4 as nc
from soil_moisture_analysis.mining.constants import *
from soil_moisture_analysis.mining.read_soil_moisture_data import avg_hourly_sm_data
from collections import Counter


def merge_data():
    pred_df = None
    for match_str in matches_list:
        count = 1
        for f in glob.glob(os.path.join(data_path+match_str)):
            in_nc = nc.Dataset(os.path.join(data_path, f))
            daily_df = avg_hourly_sm_data(in_nc)
            node_id = ""
            site = f.split("\\")[1][17:-8]
            m = re.match(".*n(\\d+).*", f)
            if m:
                node_id = m.group(1)
            node_id = "n="+node_id
            ##filtering columns to only depth = 5 data
            col_filter = [i for i in list(daily_df.columns) if not i == 'date' and i == 5]
            list_cols = [site+node_id+"_d=" + str(i) for i in list(daily_df.columns) if not i == 'date' and i == 5]
            if list_cols:
                col_filter.append('date')
                list_cols.append('date')
                daily_df = daily_df[col_filter]
                daily_df.columns = list_cols
                if pred_df is None:
                    pred_df = daily_df
                else:
                    pred_df = pred_df.merge(daily_df, on='date', how='outer')
    pred_df.to_csv(merge_data_path,index=None,sep=";")


if __name__ == '__main__':
    merge_data()

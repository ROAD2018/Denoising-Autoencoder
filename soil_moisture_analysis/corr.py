import os,re,glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import netCDF4 as nc
from mining import constants
from mining.read_soil_moisture_data import avg_daily_sm_data
from collections import Counter

matches_list = [ 'soil_moist_20min_Kendall_AZ_*','soil_moist_20min_Vaira_CA_*','soil_moist_20min_BLMLand1STonzi_CA_*','soil_moist_20min_LuckyHills_AZ_*',
                'soil_moist_20min_MatthaeiGardens_*','soil_moist_20min_NewHoganLakeN_CA_*',
                'soil_moist_20min_TerradOro_CA_*',
                'soil_moist_20min_TonziRanch_CA_*','soil_moist_20min_BLMLand2STonzi_CA_*','soil_moist_20min_BLMLand3NTonzi_CA_*']


for match_str in matches_list:
    for f in glob.glob(os.path.join(constants.data_path+match_str)):
        in_nc = nc.Dataset(os.path.join(constants.data_path, f))
        daily_df = avg_daily_sm_data(in_nc)
        print("_______________________________________")
        # print("FILE - "+f.split("\\")[1][17:-3])
        print(daily_df.corr(method= 'pearson'))
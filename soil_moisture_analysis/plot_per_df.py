import os,re,glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import netCDF4 as nc
from soil_moisture_analysis.mining import constants
from soil_moisture_analysis.mining.read_soil_moisture_data import avg_daily_sm_data
from collections import Counter

matches_list = [ 'soil_moist_20min_Kendall_AZ_*','soil_moist_20min_Vaira_CA_*','soil_moist_20min_BLMLand1STonzi_CA_*','soil_moist_20min_LuckyHills_AZ_*',
                'soil_moist_20min_MatthaeiGardens_*','soil_moist_20min_NewHoganLakeN_CA_*',
                'soil_moist_20min_TerradOro_CA_*',
                'soil_moist_20min_TonziRanch_CA_*','soil_moist_20min_BLMLand2STonzi_CA_*','soil_moist_20min_BLMLand3NTonzi_CA_*']


for match_str in matches_list:
    count = 1
    for f in glob.glob(os.path.join(constants.data_path+match_str)):
        in_nc = nc.Dataset(os.path.join(constants.data_path, f))
        daily_df = avg_daily_sm_data(in_nc)
        node_id = ""
        m = re.match(".*n(\\d+).*", f)
        if m:
            node_id = m.group(1)
        list_cols = ["depth=" + str(i) for i in list(daily_df.columns) if not i == 'date']

        list_cols.append('date')
        daily_df.columns = list_cols
        try:
            daily_df['date'] = pd.to_datetime(daily_df['date'], format='%Y%m%d').apply(lambda x: x.date())
            daily_df.index = daily_df['date']
            daily_df = daily_df.drop(columns=['date'])
            plt.figure()
            split_str = f.split("\\")[1][17:-8]
            site_str = f.split("\\")[1][17:-8]
            f_str = f.split("\\")[1][17:-3]
            ax = daily_df.plot(y=daily_df.columns, title="Site = " + site_str + ",   Node = " + node_id,
                               figsize=(16, 7))
            myFmt = DateFormatter("%Y-%m-%d")

            ax.xaxis.set_major_formatter(myFmt)
            ax.set(ylabel="soil moisture counts per day", xlabel='date')
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

            fig = ax.get_figure()
            fig.autofmt_xdate()
            fig.savefig(os.path.join('C:/Users/M Sonakshi/Documents/soil_imgs/countPlotsPerDF/', f_str + '.png'))
            plt.close('all')
        except Exception as e:
            print(e)

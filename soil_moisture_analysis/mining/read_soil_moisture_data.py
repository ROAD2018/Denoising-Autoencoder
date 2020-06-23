import pandas as pd
import numpy as np
import netCDF4 as nc
import soil_moisture_analysis.mining.constants as c


def avg_daily_sm_data(in_nc = None):
    if in_nc is None:
        in_nc = nc.Dataset(c.data_path+c.soil_moisture_path) # read file
    # print(in_nc) # print file information
    # y = in_nc.variables['lat'][:] # read latitude variable
    # x = in_nc.variables['lon'][:] # read longitude variable
    soil_moisture = in_nc.variables['soil_moisture'][:]
    depth = in_nc.variables['depth'][:] # read depth variable
    time = in_nc.variables['time'][:] # read time variable
    time_unit = in_nc.variables["time"].getncattr('units')
    time_cal = in_nc.variables["time"].getncattr('calendar')
    local_time = nc.num2date(time, units=time_unit, calendar=time_cal)
    sm_df = pd.DataFrame(soil_moisture, columns=depth, index=local_time.tolist())
    sm_df['time'] = sm_df.index
    sm_df_daily = sm_df.groupby(pd.Grouper(key='time',freq='1D')).aggregate(lambda x: x.count())
    sm_df_daily['date'] = sm_df_daily.index
    return sm_df_daily


def avg_hourly_sm_data(in_nc = None):
    if in_nc is None:
        in_nc = nc.Dataset(c.data_path+c.soil_moisture_path) # read file
    # print(in_nc) # print file information
    # y = in_nc.variables['lat'][:] # read latitude variable
    # x = in_nc.variables['lon'][:] # read longitude variable
    soil_moisture = in_nc.variables['soil_moisture'][:]
    depth = in_nc.variables['depth'][:] # read depth variable
    time = in_nc.variables['time'][:] # read time variable
    time_unit = in_nc.variables["time"].getncattr('units')
    time_cal = in_nc.variables["time"].getncattr('calendar')
    local_time = nc.num2date(time, units=time_unit, calendar=time_cal)
    sm_df = pd.DataFrame(soil_moisture, columns=depth, index=local_time.tolist())
    sm_df['time'] = sm_df.index
    sm_df_hourly = sm_df.groupby(pd.Grouper(key='time',freq='H')).aggregate(np.nanmean)
    sm_df_hourly['date'] = sm_df_hourly.index
    return sm_df_hourly

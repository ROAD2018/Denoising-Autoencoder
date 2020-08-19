"""
Using alhe ameriflux sites' data and creating sequences to be input to the autoencoder model
"""
import os, re,json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from data_preprocessing import dir_name
from data_preprocessing.validate_ameri_yearly import get_years_per_column

os.chdir(dir_name)

RANDOM_SEED = 557
def get_col_year_tuple(data):
    """
    :param data: dictionary of columns in an ameri dataframe satsifying some conditions, each column contains years and years_dict keys
    :return: tuple of column name and years list of the column with max num of years
    """
    max_years = 0
    tuple_col_year = None
    for col in data:
        years = data[col]['years']
        if max_years < len(years):
            tuple_col_year = (col, years)
    return tuple_col_year


def add_noise(x):
    """
    :param x: a list/series of data points for a year
    :return: a list/series infused with noise based on the probability given in the body of the function
    """
    np.random.seed(seed = RANDOM_SEED)
    A = np.random.random(x.size())>.5
    # print(sum(A))
    A = torch.tensor(A.astype(np.float32))
    return A*x


def get_sequence(data,batch_size,seq_len):
    """
    :param data: a list/series of data over a year
    :param batch_size: an integer depicting how many sequences in each batch
    :param seq_len: integer to determine the length of the sequence
    :return: a list of batch arrays where each batch array contains sequences of size size_len
    """
    input_sequence = []
    batch_array = []
    for i in range(0,len(data),seq_len):
        seq = data[i:i+seq_len]
        if np.nan not in seq:
            if len(batch_array)==batch_size:
                input_sequence.append(np.array(batch_array,dtype=np.float32))
                batch_array = []
                if len(seq) == seq_len:
                    batch_array.append(np.array(seq, dtype=np.float32).reshape(-1, 1))
            else:
                if len(seq) == seq_len:
                    batch_array.append(np.array(seq, dtype=np.float32).reshape(-1, 1))
    return input_sequence


if __name__ == '__main__':

    main_yearly_dict = {}
    input_sequence = []
    noisy_sequence = []
    total_data_avail = 0
    for file in os.listdir(dir_name):
        if file.endswith(".csv"):
            try:
                file_name = os.path.abspath(file)
                ameri_data = pd.read_csv(file_name, skiprows=2)
                ameri_data['TIMESTAMP_START'] = pd.to_datetime(ameri_data['TIMESTAMP_START'], format='%Y%m%d%H%M')
                ameri_data['TIMESTAMP_END'] = pd.to_datetime(ameri_data['TIMESTAMP_END'], format='%Y%m%d%H%M')
                ameri_data.index = ameri_data['TIMESTAMP_START']
                del ameri_data['TIMESTAMP_START']
                del ameri_data['TIMESTAMP_END']
                list_columns = []
                # list_columns.append('P')
                list_columns.extend(
                    [re.match("SWC.*", x).group() for x in ameri_data.columns if re.match("SWC.*", x) is not None])
                if len(list_columns) > 0:
                    ameri_data = ameri_data[list_columns]
                    ameri_data = ameri_data.replace(-9999.0, np.nan)
                    ameri_data = ameri_data.dropna(axis=0, how='all')
                    ameri_data = ameri_data[ameri_data >= 0]
                    ameri_data = ameri_data.resample('H').mean()
                    col_dict = get_years_per_column(ameri_data)
                    if col_dict:
                        tuple_col = get_col_year_tuple(col_dict)
                        ameri_data = ameri_data[[tuple_col[0]]]
                        for year in set(tuple_col[1]):
                            ameri_data = ameri_data[ameri_data.index.year == year]
                            plt.figure(figsize=(20, 10))
                            fig = plt.gcf()
                            plt.plot(ameri_data.index, ameri_data[tuple_col[0]])
                            fig.savefig(str(file.split(".")[0]) + "_"+str(year)+".png")
                            # ameri_data = ameri_data[ameri_data.index.year == year]
                            # total_data_avail += ameri_data.shape[0]
                            # input_sequence.extend(get_sequence(ameri_data[tuple_col[0]], batch_size=1, seq_len=24))
                            # noisy_sequence.extend(
                            #     get_sequence(add_noise(torch.tensor(ameri_data[tuple_col[0]])), batch_size=1, seq_len=24))

            except Exception as e:
                print(file,e)
    # input_sequence = np.array(input_sequence)
    # print(total_data_avail)
    # np.save('input_all_mixed2.npy',input_sequence)
    # np.save('noise_all_mixed2.npy',np.array(noisy_sequence))


    # with open("ameri_yearly_data_0.dict",'w') as f:
    #     json.dump(main_yearly_dict,f)



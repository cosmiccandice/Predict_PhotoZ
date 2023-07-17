import pandas as pd
import glob as glob
import numpy as np
import os

data_input_folder = '/projects/b1094/cstauffer/PhotoZ/sdss17_and_ps1dr2_to_unwise_xmatch.csv'

ps_mag =     ['ps_gpsfmag', 'ps_rpsfmag', 'ps_ipsfmag', 'ps_zpsfmag','ps_ypsfmag']
ps_kronmag = ['ps_gkronmag', 'ps_rkronmag', 'ps_ikronmag', 'ps_zkronmag', 'ps_ykronmag']
unwise_mag = ['unwise_w1_mag_ab','unwise_w2_mag_ab'] #'unwise_w1_mag_vega','unwise_w2_mag_vega'

sdss_columns = ['sdss_class','sdss_z','sdss_zErr',]

all_columns = ps_mag + ps_kronmag + unwise_mag + sdss_columns 

nrows = 300_000
total_rows = sum(1 for _ in open(data_input_folder)) - 1  # count the total number of rows in the file (minus header row)
skiprows = sorted(np.random.choice(range(1, total_rows+1), total_rows-nrows, replace=False))  # randomly choose rows to skip
filtered_xmatched = pd.read_csv(data_input_folder, skiprows=skiprows, nrows=nrows,usecols=all_columns)


#replace values of "None" with NaN and then drop rows with NaN
filtered_xmatched = filtered_xmatched.replace('None', np.nan)
filtered_xmatched = filtered_xmatched.dropna()


# Only keep redshifts above 0 and below 1
filtered_xmatched = filtered_xmatched[filtered_xmatched['sdss_z'] > 0]
filtered_xmatched = filtered_xmatched[filtered_xmatched['sdss_z'] < 1]
# Filter classes
filtered_xmatched = filtered_xmatched[filtered_xmatched['sdss_class'] == 'GALAXY']
# Filter stars
# filtered_xmatched = filtered_xmatched[filtered_xmatched['ps1dr2_p_star'] < 0.5]

# function to convert numeric strings to float, leave other strings as is
def convert_numeric(val):
    try:
        # try converting to float
        return float(val)
    except ValueError:
        # if it's not a valid float, return original value
        return val

# apply the function to each element of the dataframe
filtered_xmatched = filtered_xmatched.applymap(convert_numeric)

filtered_xmatched=filtered_xmatched[0:10]

mag_cols = ps_mag+ps_kronmag+unwise_mag 

for j in mag_cols: 
    for i in mag_cols: 
        if j != i:
            new_col_name = f'{j}-{i}'
            filtered_xmatched[new_col_name] = filtered_xmatched[j]-filtered_xmatched[i] 

#OR THIS SO I DON'T GET REDUNDANT VALUES
import itertools

for j, i in itertools.combinations(mag_cols, 2):
    new_col_name = f'{j}-{i}'
    filtered_xmatched[new_col_name] = filtered_xmatched[j] - filtered_xmatched[i]

#Save to CSV
#filtered_xmatched[all_columns].to_csv('FILTERED_sdss17_and_ps1dr2_to_unwise_xmatch.csv', index=False)
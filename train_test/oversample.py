import pandas as pd
import numpy as np

# load your dataset
df = pd.read_csv('FILTERED_sdss17_and_ps1dr2_to_unwise_xmatch.csv')

# find rows where 'sdss_z' > 0.7
mask_high = df['sdss_z'] > 0.7
df_to_duplicate_high = df[mask_high]

# randomly select 30% of these rows
rows_to_duplicate_high = df_to_duplicate_high.sample(frac=0.4, replace=True, random_state=1)

# find rows where 'sdss_z' < 0.1
mask_low = df['sdss_z'] < 0.1
df_to_duplicate_low = df[mask_low]

# randomly select 30% of these rows
rows_to_duplicate_low = df_to_duplicate_low.sample(frac=0.2, replace=True, random_state=1)

# append these rows back to the dataframe
df = df.append(rows_to_duplicate_high)
df = df.append(rows_to_duplicate_low)

# save the new dataframe to a new csv file
df.to_csv('FILTERED_sdss17_and_ps1dr2_to_unwise_xmatch_oversample.csv', index=False)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pickle
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


###  _____ CREATING MODEL  ______ ###

ps_mag =     ['ps_gpsfmag', 'ps_rpsfmag', 'ps_ipsfmag', 'ps_zpsfmag','ps_ypsfmag']
ps_kronmag = ['ps_gkronmag', 'ps_rkronmag', 'ps_ikronmag', 'ps_zkronmag', 'ps_ykronmag']
unwise_mag = ['unwise_w1_mag_ab','unwise_w1_mag_vega','unwise_w2_mag_ab','unwise_w2_mag_vega'] 
sdss_columns = ['sdss_class','sdss_z','sdss_zErr',]
all_columns = ps_mag + ps_kronmag + unwise_mag + sdss_columns 

#data_input_folder  = 'FILTERED_sdss17_and_ps1dr2_to_unwise_xmatch.csv'
data_input_folder='FILTERED_sdss17_and_ps1dr2_to_unwise_xmatch_oversample.csv'
data = pd.read_csv(data_input_folder,usecols=all_columns)


training_features = ['ps_gpsfmag', 'ps_rpsfmag', 'ps_ipsfmag', 'ps_zpsfmag', 'ps_ypsfmag',
                     'ps_gkronmag', 'ps_rkronmag', 'ps_ikronmag', 'ps_zkronmag', 'ps_ykronmag',
                     'unwise_w1_mag_ab','unwise_w1_mag_vega','unwise_w2_mag_ab','unwise_w2_mag_vega']


# Select the predictor that we're using to guide the model
predictor = 'sdss_z'

# Split our data set into a training and testing set in order to test our model performance
X_train, X_test, y_train, y_test = train_test_split(data[training_features], data[predictor], test_size=0.2, random_state=42)

# Define the random forest model
RF_reg = RandomForestRegressor(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)

# Fit the model with the training data
RF_reg.fit(X_train, y_train)

# Save model to file
with open('photoz_reg.pkl', 'wb') as file:
    pickle.dump(RF_reg, file)
print(RF_reg.feature_names_in_)


# Predict redshifts from the test data
test_predict = RF_reg.predict(X_test)
train_predict = RF_reg.predict(X_train)




### _____ PLOTS  ______ 

outfolder = 'plots/'

###TEST HIST 2
nbins = np.linspace(start=0, stop=1, num=20) 

fig, ax = plt.subplots()
h = plt.hist2d(y_test, test_predict, bins=nbins, vmin=0,vmax=3.5) 
cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label('log(N+1)')

# Take the log of the counts
counts_log = np.log10(h[0]+1)
counts_log[np.isneginf(counts_log)] = 0  # replace negative infinity with 0

# Update the colorbar with the log counts
h[3].set_array(counts_log.flatten())

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
plt.plot(lims, lims, linestyle='--', color='black', zorder=100)
plt.xlim(0,1)
plt.ylim(0,1)

# Add in 1 standard deviation lines
stds = []
means = []
for i in range(len(h[1]) - 1):
    stds.append(np.std(test_predict[(y_test >= h[1][i]) & (y_test < h[1][i+1])]))
    means.append(np.average(test_predict[(y_test >= h[1][i]) & (y_test < h[1][i+1])]))
means, stds = np.asarray(means), np.asarray(stds)
plt.plot(h[1][:-1], means + 1.*stds, linestyle='--', color='red', zorder=100)
plt.plot(h[1][:-1], means - 1.*stds, linestyle='--', color='red', zorder=100)
plt.plot(h[1][:-1], means, color='red', zorder=100)

plt.xlabel(r'$Z_{spec}$')
plt.ylabel(r'$Z_{phot}$')
plt.savefig(outfolder + 'hist2_reg.png')
plt.clf()

###TEST HIST 3
nbins = np.linspace(start=0, stop=1, num=20) 

z_diff = test_predict - y_test

fig, ax = plt.subplots()
h = plt.hist2d(z_diff,y_test, bins=nbins, vmin=0, vmax=3.5)
cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label('log(N+1)')

counts_log = np.log10(h[0]+1)
counts_log[np.isneginf(counts_log)] = 0

h[3].set_array(counts_log.flatten())

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
plt.plot(lims, [0,0], linestyle='--', color='black', zorder=100) #zero line
plt.xlim(0,1)
plt.ylim(np.min(z_diff), np.max(z_diff))

# Calculate the 10th and 90th percentiles
percentiles = []
for i in range(len(h[1]) - 1):
    values_in_bin = z_diff[(y_test >= h[1][i]) & (y_test < h[1][i+1])]
    percentiles.append(np.percentile(values_in_bin, [10, 90]))
percentiles = np.asarray(percentiles)

plt.plot(h[1][:-1], percentiles[:, 0], linestyle='--', color='red', zorder=100)
plt.plot(h[1][:-1], percentiles[:, 1], linestyle='--', color='red', zorder=100)

plt.xlabel(r'$Z_{spec}$')
plt.ylabel(r'$Z_{phot}-Z_{spec}$')
plt.savefig(outfolder + 'hist3_reg.png')
plt.clf()

### _____ CALCULATIONS ______ ### 

# Compute the bias
bias = (np.asarray(test_predict) - np.asarray(y_test))/(1. + np.asarray(y_test))
bias_std = np.std(bias)
filtered_bias = []

for item in bias:
    if np.abs(item) < 3. * bias_std:
        filtered_bias.append(item)
filtered_bias = np.asarray(filtered_bias)

# Define a function to compute the outlier fraction
def outlier_fraction(bias_list, threshold=3):
    bias_std = np.std(bias_list)
    outliers = [item for item in bias_list if np.abs(item) > threshold * bias_std]
    return len(outliers) / len(bias_list)

# Define bins to group y_test values
bins = np.linspace(np.min(y_test), np.max(y_test), 20)
bins = nbins 

# Compute the outlier fraction for each bin
outlier_fractions = []
for i in range(len(bins) - 1):
    bias_list = bias[(y_test >= bins[i]) & (y_test < bins[i+1])]
    outlier_fractions.append(outlier_fraction(bias_list))

# Plot the outlier fractions
plt.figure()
plt.plot(bins[:-1], outlier_fractions)
plt.scatter(bins[:-1], outlier_fractions)
plt.ylabel(r'$P_0$')
plt.xlabel(r'$Z_{spec}$')
plt.savefig(outfolder + 'outlier_fraction.png')
plt.clf()

##Residuals 
z_spec = y_test 
z_phot = test_predict 
residuals = z_phot - z_spec

# Define the number of bins
num_bins = 20

# Define the bins equally spaced on z_spec
bins = np.linspace(np.min(z_spec), np.max(z_spec), num_bins)

# Define the mid-points of the bins
bin_mids = (bins[:-1] + bins[1:]) / 2

# Initialize a list to hold the median residuals for each bin
median_residuals = []

# Calculate the median residual for each bin
for i in range(num_bins-1):
    # Select the residuals for points within this bin
    residuals_in_bin = residuals[(z_spec >= bins[i]) & (z_spec < bins[i+1])]
    # Compute the median of these residuals
    median_residuals.append(np.median(residuals_in_bin))

# Plot the median residuals against the mid-points of the bins
plt.figure(figsize=(10, 8))
plt.plot(bin_mids, median_residuals) 
plt.scatter(bin_mids, median_residuals) 
plt.axhline(0, color='red', linestyle='--')  # add a horizontal line at zero
plt.xlabel(r'$z_{spec}$',fontsize=20)
plt.ylabel(r'$z_{phot} - z_{spec}$',fontsize=20)
plt.savefig(outfolder + 'median_residuals_plot.pdf')
plt.clf()

#METRICS 
# Calculate ∆znorm
delta_znorm = (test_predict - y_test) / (1 + y_test)

print('∆znorm: {}'.format(np.average(filtered_bias)))
# Calculate σ(∆znorm)

sigma_delta_znorm = np.std(delta_znorm)
print('σ(∆znorm) = ', sigma_delta_znorm)

print('P0: {}%'.format(100.*(len(bias)-len(filtered_bias))/len(bias)))

print('<∆znorm>: {}'.format(np.average(bias)))

# Calculate the mean absolute deviation (MAD)
sigma_MAD = np.mean(np.abs(bias))
print('Sigma_MAD: {}'.format(sigma_MAD))

#FROM BECK et al 
# Calculate delta_znorm values greater than 0.15 (these are the outliers according to Beck et al. 2020)
# Calculate O
O_val = np.sum(np.abs(delta_znorm) > 0.15) / len(delta_znorm)
print('O = ', O_val)
outliers_delta_znorm = delta_znorm[np.abs(delta_znorm) > 0.15]
# Compute the average bias (∆znorm) for these outliers
average_bias_outliers = np.mean(outliers_delta_znorm)
# Compute the standard deviation σ(∆znorm) for these outliers
std_dev_outliers = np.std(outliers_delta_znorm)
# Now we remove these outliers to compute statistics for non-outliers
non_outliers_delta_znorm = delta_znorm[np.abs(delta_znorm) <= 0.15]
# Compute the average bias (∆znorm) for non-outliers
average_bias_non_outliers = np.mean(non_outliers_delta_znorm)
print("The average bias (∆znorm') for non-outliers is: {}".format(average_bias_non_outliers))
# Compute the standard deviation σ(∆znorm) for non-outliers
std_dev_non_outliers = np.std(non_outliers_delta_znorm)
print("The standard deviation σ(∆znorm') for non-outliers is: {}".format(std_dev_non_outliers))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.colors import LogNorm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


wise_columns = ['unwise_decl','unwise_ra','unwise_w1_decl12','unwise_w1_dflux','unwise_w1_dfluxlbs','unwise_w1_dspread_model','unwise_w1_dx',
                'unwise_w1_dy','unwise_w1_flux','unwise_w1_fluxlbs','unwise_w1_fracflux',
                'unwise_w1_fwhm','unwise_w1_nm','unwise_w1_ra12','unwise_w1_rchi2','unwise_w1_sky',
                'unwise_w1_spread_model','unwise_w1_x','unwise_w1_y',
                'unwise_w2_decl12','unwise_w2_dflux','unwise_w2_dfluxlbs','unwise_w2_dspread_model','unwise_w2_dx','unwise_w2_dy',
                'unwise_w2_flux','unwise_w2_fluxlbs','unwise_w2_fracflux','unwise_w2_fwhm','unwise_w2_nm',
                'unwise_w2_ra12','unwise_w2_rchi2','unwise_w2_sky','unwise_w2_x','unwise_w2_y'] 

all_columns = ['ps_gpsfmag', 'ps_rpsfmag', 'ps_ipsfmag', 'ps_zpsfmag', 'ps_ypsfmag','ps_gkronmag', 'ps_rkronmag', 'ps_ikronmag', 'ps_zkronmag', 'ps_ykronmag','sdss_z']+wise_columns

# data_input_folder = 'FILTERED_sdss17_to_ps1dr2_xmatch.csv'

data_input_folder  = 'FILTERED_sdss17_and_ps1dr2_to_unwise_xmatch.csv'
data = pd.read_csv(data_input_folder)[all_columns] 

print (len(data))

dict = {'sdss_z': 'z'}
data.rename(columns=dict,inplace=True)


# Define the redshift bins/classes
z_bins = np.linspace(start=0, stop=1, num=22)  # dividing redshift range into 21  equally sized bins

# Generate list of equally spaced string numbers between 0 and 1
nums = np.linspace(0, 1, num=21)
class_names = [f"{num:.2f}" for num in nums]

data['z'] = pd.cut(data['z'], bins=z_bins, labels=class_names)  # map redshift values to corresponding bin

# Choose training features
training_features = ['ps_gpsfmag', 'ps_rpsfmag', 'ps_ipsfmag', 'ps_zpsfmag', 'ps_ypsfmag', 
                     'ps_gkronmag', 'ps_rkronmag', 'ps_ikronmag', 'ps_zkronmag', 'ps_ykronmag']

# Select the predictor that we're using to guide the model
predictor = 'z'

# Renormalise our features and the predictor
# scaler = StandardScaler()
# data[training_features] = scaler.fit_transform(data[training_features])

# z_mean = scaler.mean_[-1]
# z_std = np.sqrt(scaler.var_[-1])

# Split the data set into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(data[training_features], data[predictor], test_size=0.3, random_state=42)

# Define the random forest model
RF_cls = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)

# Fit the model with the training data
RF_cls.fit(X_train, y_train)


# Compute the accuracy
accuracy_train = RF_cls.score(X_train, y_train)
accuracy_test = RF_cls.score(X_test, y_test)
print (accuracy_test)
print (accuracy_train)

test_predict = RF_cls.predict(X_test)
train_predict = RF_cls.predict(X_train)

#convert mapped number-strings back to numbers
test_predict= np.array([float(s) for s in test_predict]) 
train_predict = np.array([float(s) for s in train_predict]) 
y_train = y_train.astype(float)
y_test = y_test.astype(float)


outfolder = 'plots/'

# #classification report
# report = classification_report(y_test,test_predict) 
# lines = report.split('\n')
# data = [line.split() for line in lines[2:-3]]
# headers = lines[0].split()
# if len(data[0]) > len(headers):
#     headers.append('extra')
# df = pd.DataFrame(data, columns=headers)
# df.to_csv(outfolder + 'classification_report.csv', index=False)
# #print (report)

# # Get and reshape confusion matrix data
# #print (confusion_matrix(y_test, test_predict))  #print confusion matrix array
# matrix = confusion_matrix(y_test, test_predict)
# matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# # Confusion Matrix
# plt.figure(figsize=(16,7))
# sns.set(font_scale=1.4)
# sns.heatmap(matrix, annot=True, annot_kws={'size':10},
#             cmap=plt.cm.Greens, linewidths=0.2,fmt=".2f")

# # Add labels to the plot
# tick_marks = np.arange(len(class_names))
# tick_marks2 = tick_marks + 0.5
# plt.xticks(tick_marks, class_names, rotation=25)
# plt.yticks(tick_marks2, class_names, rotation=0)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.title('Confusion Matrix for Random Forest Model')
# plt.savefig(outfolder + 'confusion_matrix.png')
# plt.clf()


# # Plot training data spectroscopic and photometric redshift
# plt.scatter(y_train, train_predict_float)
# plt.plot(y_train, y_train)
# plt.xlabel('Zspec')
# plt.ylabel('Zphot')
# plt.savefig(outfolder + 'test_class.png')
# plt.clf()


#Create 2D histogram with colorbar
fig, ax = plt.subplots()
h = plt.hist2d(y_test, test_predict, bins=(z_bins,z_bins),vmin=0, vmax=3.5)
cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label('log(N+1)')

# Take the log of the counts
counts_log = np.log10(h[0]+1)
counts_log[np.isneginf(counts_log)] = 0  # replace negative infinity with 0

# Update the colorbar with the log counts
h[3].set_array(counts_log.flatten())

#set the plot limits 
lims = [    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
plt.plot(lims, lims, linestyle='--', color='black', zorder=100)
plt.xlim(0, 1)
plt.ylim(0, 1)

# Add in 1 standard deviation lines
stds = []
means = []
for i in range(len(h[1]) - 1):
    stds.append(np.std(test_predict[(y_test >= h[1][i]) & (y_test < h[1][i+1])]))
    means.append(np.average(test_predict[(y_test >= h[1][i]) & (y_test < h[1][i+1])]))
    
means, stds = np.asarray(means), np.asarray(stds)

#plot the plot :) 
plt.plot(h[1][:-1], means + 1.*stds, linestyle='--', color='red', zorder=100)
plt.plot(h[1][:-1], means - 1.*stds, linestyle='--', color='red', zorder=100)
plt.plot(h[1][:-1], means, color='red', zorder=100)

plt.xlabel('Zspec')
plt.ylabel('Zphot')
plt.savefig(outfolder + 'hist2_class.png')
plt.clf()

##HexPlot
# fig, ax = plt.subplots(figsize=(9,5))
# h = ax.hexbin(y_test, test_predict_float, bins='log', cmap='viridis', gridsize=19) 
# cb = plt.colorbar(h)
# cb.set_label('log(N+1)')

    
# ax.set_xlabel('Zspec')
# ax.set_ylabel('Zphot')

# plt.legend()
# plt.savefig(outfolder + 'hexbin_plot.png')
# plt.clf()

# ### Compute the bias
# bias = (np.asarray(test_predict_float) - np.asarray(y_test))/(1. + np.asarray(y_test))
# bias_std = np.std(bias)
# filtered_bias = []

# for item in bias:
#     if np.abs(item) < 3. * bias_std:
#         filtered_bias.append(item)
# filtered_bias = np.asarray(filtered_bias)
# print('The average bias is: {}'.format(np.average(bias)))
# print('The bias with outliars removed is: {}'.format(np.average(filtered_bias)))
# print('The outlier percentage is: {}%'.format(100.*(len(bias)-len(filtered_bias))/len(bias)))
import pandas as pd
import pickle

with open('photoz_reg.pkl', 'rb') as file:
    RF_reg = pickle.load(file)

training_features = ['ps_gpsfmag','ps_rpsfmag','ps_ipsfmag','ps_zpsfmag','ps_ypsfmag',
'ps_gkronmag','ps_rkronmag','ps_ikronmag','ps_zkronmag','ps_ykronmag',
 'unwise_w1_mag_ab','unwise_w1_mag_vega','unwise_w2_mag_ab',
 'unwise_w2_mag_vega']

data_input_folder= input("Please enter the name of the data input folder: ") or 'example_data.csv'

data = pd.read_csv(data_input_folder,usecols=training_features) 

sample_data = data.iloc[0].to_frame().transpose()
sample_data.columns = training_features

new_predictions = RF_reg.predict(sample_data)

print (data.iloc[0]) 
print ('Photo-Z Prediction:', new_predictions[0])


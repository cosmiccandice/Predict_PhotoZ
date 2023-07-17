
# Predict Photometric Redshift (Photo-Z) 

This tool uses are random forest regression to predict photometric redshift of galaxies using features from PanSTARRS and unWISE survey data. It is tested against spectroscopic redshift estimates from SDSS survey. The training and testing of this model/data can be seen in the `train_test` directory, while the cross-matching of SDSS, PanSTARRS, and unWISE sources is shown in the `xmatch` directory. 

The `photoz_reg.pkl`, `main.py`, and `example_data.csv` files contain the saved model, the script to run the model, and an example of data that can be run on this tool respectively. 


### How to use the tool:

1. Download `photoz_reg.pkl`, `main.py`, and `example_data.csv` into any local directory.

2. Locate/download the following features of an object: `ps_gpsfmag,	ps_rpsfmag, ps_ipsfmag, ps_zpsfmag, ps_ypsfmag, ps_gkronmag	ps_rkronmag, ps_ikronmag, ps_zkronmag, ps_ykronmag, unwise_w1_mag_ab, unwise_w1_mag_vega, unwise_w2_mag_ab, unwise_w2_mag_vega`

3. Create a .csv file that follows the template of `example_data.csv` with the aformentioned features OR simply revise example_data.csv to include new targets features.

4. Run `main.py` by running `python main.py` in the terminal

5. When prompted, enter the name of your file (e.g. `example_data.csv`) or press enter to run the tool with data from `example_data.csv`

6. Program will return the data of the features you entered and the predicted photometric redshift. 
   

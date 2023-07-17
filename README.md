
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
   
Results from running example_data.py should look like this:



    ✨ python main.py
    
    Please enter the name of the data input folder: example_data.py
    
    [Parallel(n_jobs=12)]: Using backend ThreadingBackend with 12 concurrent workers.
    
    [Parallel(n_jobs=12)]: Done  26 tasks  | elapsed:  0.0s
    
    [Parallel(n_jobs=12)]: Done 100 out of 100 | elapsed:  0.0s finished
    
    ps_gpsfmag  19.296301
    
    ps_rpsfmag  18.241699
    
    ps_ipsfmag  17.846901
    
    ps_zpsfmag  17.717300
    
    ps_ypsfmag  17.301001
    
    ps_gkronmag 18.182501
    
    ps_rkronmag 17.292900
    
    ps_ikronmag 16.792101
    
    ps_zkronmag 16.573601
    
    ps_ykronmag 16.399200
    
    unwise_w1_mag_ab  16.471410
    
    unwise_w1_mag_vega  13.772410
    
    unwise_w2_mag_ab  16.933155
    
    unwise_w2_mag_vega  13.594154
    
    Name: 0, dtype: float64
    
    Photo-Z Prediction: 0.09868804969999995

# Dissertation


# 1. Inputs

This directory is missing due to excessive files' size 

To view and download the files used visit [Kaggle Website](
    Calendar and Listings Datasets 2020: (https://www.kaggle.com/code/labdmitriy/inside-airbnb-london/input)
    Calendar and Listings Datasets 2021:(https://www.kaggle.com/code/levorato/eda-airbnb-london-uk/input)
    Calendar and Listings Datasets 2022:(https://www.kaggle.com/code/lifuhaha/vis-group-14/input)
)

# 2. Analysis

This directory contains scripts and notebooks for analyzing the data for Listings and Calendar Datasets:

## Calendar Analysis

This folder comprises:

- A separate folder (2020, 2021, 2022) for the analysis and changes applied to geenerate the average price by listing for each year from 2020 to 2022 
---- In each folder there is also a subfolder with a csv file comprising the calcualted average price by listing along with price segment, neighbourhood, most_frequent_segment 

- A folder (Comparison_2020-2022) that compares the average price changes and the coefficient of price variation from 2020 to 2022

- A folder (Functions) comprises three files with functions created to automate the transformations and observations for the three calendar datasets. 
---- The first (1) multiple_segments.py provides the functions necessary to handle listings with multiple price segments assgined. 
---- The second (2) myfuntions.py provides all the functions for data transformations (eg. remove $ signs etc.) and functions for exploratory data analysis
---- The third (3) seasonalfunctions.py provides functions to check how the price deviates from month to month with line plots etc.

## Listings Analysis

- A separate folder (2020, 2021, 2022) for the analysis and datasets for each year from 2020 to 2022 

---- For 2020 folder:
     The subfolder (listings_analysis_2020) comprises the listings_2020.ipynb which has the splits for all datasets (Dataset_1 to Dataset_7) describsed in the report, along with the models' results after training and testing 

     For 2021 and 2022 folders:
     The subfolders comprise the listing_2021 and listings_2022 respsectively. Each one has the variables transfroamtions on the same features as 2020 (used for models' training in 2020), along with the models' training.

---- In each folder there is also a subfolder with the a dataset for each segment and a second subfolder comprising the models training and testing in a jupyter notebook

---- For 2020 folder, there is extra:

     - In the analysis folder:
      (a) a second jupyter notebook that was used to split the datasets for each segment and includes explaratory analysis for listings' datasets (the exploratory part was no utilised for the report due to words limit and since the main point of this research was to generate a more accurate price to be used as the target in the models and compare with the original one and also determine if separate machine learning models can perform better with the same features as the general models)

     - In the datasets' folder:
      (a) two extra datasets comprising the features from all the data but one has the 'mean_price' and the other has the 'price'
      (b) one csv for the amenities and one for the host_verifications that were necessary for the comparison between 2020 and 2021 and 2022
      (c) two subfolders that comprise the train, validation and test sets used, one for those after the split of Dataset_1 and one for those after the split of Dataset_2
      ** the test,validation and test sets for the other datasets used were not uploaded due to size (over 100 mb). However, they are not necessary for the code to run and they would have been uploaded just to show the shape of datasets and the transformed variables that were fed to the models**
---- 

# 3. Outputs 

This directory is missing since it comprised only the plots used and presented in the analysis and due to files' size. ** the plots can be found in jupyter notebooks' outputs**

# 4. Models

This directory contains the models'weights after training the models in different datasets.

## Files
- `XGB_weights_dataset_1.json`: XGBoost model weights for dataset 1.
- `XGB_weights_dataset_2.json`: XGBoost model weights for dataset 2.
- `XGB_weights_dataset_3_10_comps.json`: XGBoost model weights for dataset 3 with 10 components.
- `XGB_weights_dataset_3_no_text.json`: XGBoost model weights for dataset 3 without text features.
- `XGB_weights_dataset_3.json`: XGBoost model weights for dataset 3.
- `XGB_weights_original.json`: Original Price as Target XGBoost model weights.

** the XG-Boost that was trained on the budget listings dataset is not included due to poor performance however the results are present in the jupyter notebook.
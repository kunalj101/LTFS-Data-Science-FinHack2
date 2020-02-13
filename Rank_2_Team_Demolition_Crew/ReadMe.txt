Packages required along with their version:

data.table_1.12.2
dplyr_0.8.3
xgboost_0.90.0.2
MLmetrics_1.1.1
rPython_0.0-6
reticulate_1.13
fpp_0.5
imputeTS_3.0
forecast_8.10
zoo_1.8-6


Steps to generate the final prediction:
1. Please put train, test, submission and holiday (external data, shared over mail, source:  https://www.officeholidays.com/countries/india) file in data folder and set it as working directory. 

2. Now please execute the codes in sequence as provided below: 
	1_seg_1_xgb
	2_1_tbats.ipynb
	2_2_seg_1_Tbats
	3_seg_2_xgb
	4_ensemble_all
	
Executing the 4th code will generate the final submission with name 'final_pred_seg_1_2.csv' in data folder. 

import pandas as pd
from ltfs_finhack import predict_finhack_data

input_train = pd.read_csv('../input/ltfs_train.csv')
input_test = pd.read_csv('../input/ltfs_test.csv')

try:
	festival_data = pd.read_csv('../input/festival_dates.csv')
except:
	festival_data = None

try:
	predicted_test_data = predict_finhack_data(input_train, input_test, festival_data)
	predicted_test_data.to_csv('../output/predicted_test_data.csv', index = False)
except Exception as e:
	print('Execution failed :(')
	print(e)





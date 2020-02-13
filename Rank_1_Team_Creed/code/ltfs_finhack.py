import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta

def predict_finhack_data(input_train, input_test, festive_dates):
	"""
	"""
	# time stamp conversion
	input_train['application_date'] = pd.to_datetime(input_train['application_date'], format = "%Y-%m-%d")
	input_test['application_date'] = pd.to_datetime(input_test['application_date'], format = "%Y-%m-%d")
	print('Aggregating input done on country level...')
	input_train = aggregate_input_data(input_train.copy(deep = True))
	print('   Done!')
	final_test = None
	for each_segment, each_segment_data in input_train.groupby(['seg']):
		print('Analyzing segment', each_segment, '...')
		each_segment_test = input_test[input_test['segment'] == each_segment]
		each_segment_data = each_segment_data.reset_index(drop = True)
		each_segment_test = each_segment_test.reset_index(drop = True)
		predicted_result = predict_on_test_data(each_segment, each_segment_data, each_segment_test, festive_dates)
		final_test = pd.concat([predicted_result, final_test])
		print('   Done!')

	# final_test = final_test.sort_values()
	return final_test
	# input_train = compute_festive_variation(each_segment_data, festive_dates)
# =====================================
def aggregate_input_data(input_train):
	"""
	"""
	new_df = []
	for i, j in input_train.groupby(['application_date','segment']):
		add = sum(j['case_count'].tolist())
		temp = {'date': i[0], 'seg': i[1], 'apps': add}
		new_df.append(temp)
		
	new_df = pd.DataFrame(new_df, columns = new_df[0].keys())
	return new_df
# =====================================
def predict_on_test_data(each_segment, each_segment_data, each_segment_test, festive_dates):
	"""
	"""
	# add temporal values to the data set
	# import pdb; pdb.set_trace()
	print('   Finding pattern...')
	each_segment_data.loc[:, 'weekday'] = each_segment_data.date.dt.weekday.tolist()
	each_segment_data.loc[:, 'year'] = each_segment_data.date.dt.year.tolist()
	each_segment_data.loc[:, 'day'] = each_segment_data.date.dt.day.tolist()
	each_segment_data.loc[:, 'month'] = each_segment_data.date.dt.month.tolist()

	each_segment_test.loc[:, 'weekday'] = each_segment_test.application_date.dt.weekday.tolist()
	each_segment_test.loc[:, 'year'] = each_segment_test.application_date.dt.year.tolist()
	each_segment_test.loc[:, 'day'] = each_segment_test.application_date.dt.day.tolist()
	each_segment_test.loc[:, 'month'] = each_segment_test.application_date.dt.month.tolist()

	# Special temporal values
	k = each_segment_data.loc[:, 'year'].astype(str)+"-"+each_segment_data.loc[:, 'month'].astype(str)
	kk = each_segment_test.loc[:, 'year'].astype(str)+"-"+each_segment_test.loc[:, 'month'].astype(str)
	l = pd.to_datetime(k, format = '%Y-%m')+pd.offsets.MonthEnd(1)
	ll = pd.to_datetime(kk, format = '%Y-%m')+pd.offsets.MonthEnd(1)
	# Getting last but one day of month
	l1 = l-timedelta(days=1)
	l1 = list(set(l1.to_list()))
	ll1 = ll-timedelta(days =1)
	ll1 = list(set(ll1.to_list()))
	each_segment_data.loc[:,'special'] = each_segment_data['day'].astype(str)
	each_segment_data.loc[:,'bankSpecific'] = ['others']*each_segment_data.shape[0]
	each_segment_test.loc[:,'special'] = each_segment_test['day'].astype(str)
	each_segment_test.loc[:,'bankSpecific'] =['others']*each_segment_test.shape[0]
	if each_segment == 2:
		each_segment_data.loc[each_segment_data['day'] <= 10, 'bankSpecific'] = 'others1'
		each_segment_data.loc[each_segment_data['day'] > 10, 'bankSpecific'] = 'others2'
		each_segment_test.loc[each_segment_test['day'] <= 10, 'bankSpecific'] = 'others1'
		each_segment_test.loc[each_segment_test['day'] > 10, 'bankSpecific'] = 'others2'

	each_segment_data.loc[each_segment_data['day'] == 1, 'special'] = 'startOfMonth'
	each_segment_data.loc[each_segment_data['day'] == 15, 'special'] = 'midOfMonth'
	each_segment_data.loc[each_segment_data['date'].isin(l), 'special'] = 'endOfMonth'
	each_segment_data.loc[each_segment_data['date'].isin(l1), 'special'] = 'endOfMonth1'

	each_segment_test.loc[each_segment_test['day'] == 1, 'special'] = 'startOfMonth'
	each_segment_test.loc[each_segment_test['day'] == 15, 'special'] = 'midOfMonth'
	each_segment_test.loc[each_segment_test['application_date'].isin(ll), 'special'] = 'endOfMonth'
	each_segment_test.loc[each_segment_test['application_date'].isin(ll1), 'special'] = 'endOfMonth1'

	pattern_matrix = None
	# predcition part
	data_x = each_segment_data.copy(deep = True)
	# Considering behavioural chagne
	data_x = data_x[data_x['year'] == 2019]
	data_x = data_x[data_x['month'] > 3]
	if each_segment == 2:
		# data have two clusters dividing the clusters()
		part_1 = data_x[data_x['bankSpecific'] == 'others1']
		part_2 = data_x[data_x['bankSpecific'] == 'others2']
		in_data = {'others2': part_2, 'others1': part_1}
	if each_segment == 1:
		part_1 = data_x.copy(deep = True)
		in_data = {'others': part_1}
	for part, each_part in in_data.items():
		each_part_data = each_part.copy(deep = True)
		# weekay
		unique_wdays = list(set(each_part_data.weekday.tolist()))
		wday_dict = []
		sub_in_data = each_part_data[each_part_data['special'] != 'endOfMonth']
		sub_in_data = sub_in_data[sub_in_data['special'] != 'endOfMonth1']
		if part == 'others':
			sub_in_data = sub_in_data[sub_in_data['special'] != 'startOfMonth']
			sub_in_data = sub_in_data[sub_in_data['special'] != 'midOfMonth']
		seg = str(each_segment)
		key = part
		# ===============================================================================
		for ii in unique_wdays:
		#	print('======')
			values = sub_in_data.loc[sub_in_data['weekday'] == ii, 'apps']
			if len(values) == 1:
				avg = values.values[0]
				rep_value = avg
				std = 0
			else:
				vals = values.tolist()
				avg = np.percentile(vals, 75)
				rep_value = linear_regression(vals)
				avgg = np.mean(vals)
				if pd.isnull(avg):
					avg = vals[0]
				std = values.std()
			variation = std/avgg
			m_m_variation = max(values.values) - min(values.values)
			temp = {'pattern_value': ii, 'avg':avg, 'std': std, 'variation': variation, 'min_max_diff': m_m_variation, 
					'pattern':'DOW', 'data': seg+'_'+key ,'rep_value': rep_value}
			wday_dict.append(temp)
		wday_dict = pd.DataFrame(wday_dict, columns = wday_dict[0].keys())
		# =================================================================================
		# special
		unique_sdays = list(set(each_part_data.special.tolist()))
		sub_in_data = each_part_data[each_part_data['weekday'] != 6]
		sday_dict = []
		for iii in unique_sdays:
			values = sub_in_data.loc[sub_in_data['special'] == iii, 'apps']
			if len(values) == 1:
				avg = values.values[0]
				rep_value = avg
				std = 0
			else:
				vals = values.tolist()
				if len(vals) > 2:
					vals = vals[-2:]
				rep_value = linear_regression(vals)
				#                 print(iii, rep_value, vals)
				avg = np.percentile(vals, 75)
				avgg = np.mean(vals)
				if pd.isnull(avg):
					avg = vals[0]
				std = values.std()
			variation = std/avgg
			m_m_variation = max(values.values) - min(values.values)
			temp = {'pattern_value': iii, 'avg':avg, 'std': std, 'variation': variation, 'min_max_diff': m_m_variation, 
						'pattern':'special', 'data': seg+'_'+key, 'rep_value': rep_value}
			sday_dict.append(temp)
		sday_dict = pd.DataFrame(sday_dict, columns = sday_dict[0].keys())
		temp_pattern_matrix = pd.concat([wday_dict, sday_dict])
		# import pdb; pdb.set_trace()
		pattern_matrix = pd.concat([temp_pattern_matrix, pattern_matrix])

	pattern_matrix = pattern_matrix.reset_index(drop = True)
	# =================================================
	# predict future
	each_segment_test['data'] = each_segment_test['segment'].astype(str)+'_'+each_segment_test['bankSpecific']
	test = each_segment_test.copy(deep = True)
	forecast = []
	dom_avg_variation = round(np.mean(pattern_matrix.loc[pattern_matrix['pattern'] == 'special','variation'].tolist()), 3)
	dow_avg_variation  = round(np.mean(pattern_matrix.loc[pattern_matrix['pattern'] == 'DOW','variation'].tolist()), 3)
	print('      DOM variation: ', dom_avg_variation)
	print('      DOW variation: ', dow_avg_variation)
	best_dimension = 'DOM'
	if dom_avg_variation > dow_avg_variation:
		best_dimension = 'DOW'
	print('      Best pattern',  best_dimension)
	print('         Done!')
	print('   Forecasting future values')
	for i, j in test.iterrows():
		data_type = j['data']
		sub_pattern_matrix = pattern_matrix[pattern_matrix['data'] == data_type]
		dow_sub_data = sub_pattern_matrix[sub_pattern_matrix['pattern'] == 'DOW']
		dow_sub_data = dow_sub_data.reset_index(drop = True)
		dom_sub_data = sub_pattern_matrix[sub_pattern_matrix['pattern'] == 'special']
		dom_sub_data = dom_sub_data.reset_index(drop = True)
		dow_variation = float(dow_sub_data.loc[dow_sub_data['pattern_value'] == j['weekday'], 'variation'])
		dom_variation = float(dom_sub_data.loc[dom_sub_data['pattern_value'] == str(j['special']), 'variation'])

		dow_min_max = float(dow_sub_data.loc[dow_sub_data['pattern_value'] == j['weekday'], 'min_max_diff'])
		dom_min_max = float(dom_sub_data.loc[dom_sub_data['pattern_value'] == str(j['special']), 'min_max_diff'])
		dow_std = float(dow_sub_data.loc[dow_sub_data['pattern_value'] == j['weekday'], 'std'])
		dom_std = float(dom_sub_data.loc[dom_sub_data['pattern_value'] == str(j['special']), 'std'])
		
		dow_rep = float(dow_sub_data.loc[dow_sub_data['pattern_value'] == j['weekday'], 'rep_value'])
		dom_rep = float(dom_sub_data.loc[dom_sub_data['pattern_value'] == str(j['special']), 'rep_value'])
		case_count = dom_rep
		if best_dimension == 'DOM':
			if j['weekday'] == 6:
				case_count = dow_rep
			if j['special'] in ['endOfMonth1', 'endOfMonth']:
				case_count = dom_rep
		if best_dimension == 'DOW':
			case_count = dow_rep

		case_count = round(case_count)
		dow_threshold = float(dow_sub_data.loc[dow_sub_data['pattern_value'] == j['weekday'], 'avg'])
		dom_threshold = float(dom_sub_data.loc[dom_sub_data['pattern_value'] == j['special'], 'avg'])
		tempp = {'id': j['id'], 'application_date': j['application_date'], 'segment': j['segment'],
				'case_count': case_count}
		forecast.append(tempp)

	predicted_test = pd.DataFrame(forecast, columns = forecast[0].keys())
	print('      Done!')
	try:
		predicted_test = adjust_festive_dates(each_segment_data, predicted_test, each_segment, festive_dates)
	except Exception as e:
		print(e)
		pass
	return predicted_test

# ====================================
def identify_best_k_for_a_measure(values, measure, CONST_K_RANGE = np.linspace(-2,2,41),
								  CONST_QUANTILE_RANGE = np.linspace(30, 80, 6)):
	"""
	Identify best k for each measure for mean
	Input: values, measure, CONST_K_RANGE
	Output: best k, best_value
	"""
	absolute_distance_one = np.array([])
	absolute_distances = np.array([])
	all_representative_values = np.array([])
	if (measure == "mean"):
		avg_val = np.mean(values)
		sd_val = np.std(values)
		all_representative_values = (avg_val + sd_val * CONST_K_RANGE)
	if (measure == "median"):
		median_val = np.percentile(values, 50)
		mad_val = np.percentile(np.abs(values - np.percentile(values, 50)), 50)
		all_representative_values = (median_val + mad_val * CONST_K_RANGE)
	if (measure == "quantile"):
		all_representative_values = np.percentile(values, CONST_QUANTILE_RANGE)
	for x in all_representative_values:
		absolute_distance_one = np.sum(np.absolute(values - x))
		absolute_distances = np.append(absolute_distances, absolute_distance_one)
	best_value = min(absolute_distances)
	best_measure_index = np.argmin(absolute_distances)
	if measure == "mean":
		best_k = CONST_K_RANGE[(best_measure_index)]
	if measure == "median":
		best_k = CONST_K_RANGE[(best_measure_index)]
	if measure == "quantile":
		best_k = CONST_QUANTILE_RANGE[(best_measure_index)]

	return list([best_k, best_value])

def compute_representative_value(values, CONST_ALL_MEASURES, CONST_K_RANGE = np.linspace(-2,2,41),
								 CONST_QUANTILE_RANGE = np.linspace(20, 90, 15)):
	"""
	# Compute representative value of each dimension value in given timeseries
	# Input: values
	# Output: representative value
	"""
	if len(values) == 1:
		return values.values[0]
	best_k = np.array([])
	best_measure = np.array([])
	best_value_mean = np.array([])
	best_value_mad = np.array([])
	best_value_quantile = np.array([])

	# compute representative value of each dimension value in given timeseries
	# Identify best measure and best k
	for one_measure in CONST_ALL_MEASURES: # mohit : test by putting only 2 measures in CONST
		# Identify best k for each measure for mean
		if one_measure == "mean":
			(best_k_mean, best_value_mean) = identify_best_k_for_a_measure(values,
																		   one_measure,
																		   CONST_K_RANGE,
																		   CONST_QUANTILE_RANGE)
		# Identify best k for each measure for median
		if one_measure == "median":
			(best_k_mad, best_value_mad) = identify_best_k_for_a_measure(values,
																		 one_measure,
																		 CONST_K_RANGE,
																		 CONST_QUANTILE_RANGE)
		if one_measure == "quantile":
		# Identify best k for each measure for quantile
			(best_k_quantile, best_value_quantile) = identify_best_k_for_a_measure(values,
																		   one_measure,
																		   CONST_K_RANGE,
																		   CONST_QUANTILE_RANGE)

	# Identify best measure and best_k
	all_measures_values = np.array([best_value_mean, best_value_mad, best_value_quantile])
	best_of_all_measure_index = np.argmin(all_measures_values)
	best_measure = CONST_ALL_MEASURES[best_of_all_measure_index]
	if best_measure == "mean":
		best_k = best_k_mean
	if best_measure == "median":
		best_k = best_k_mad
	if best_measure == "quantile":
		best_k = best_k_quantile

	# Compute representative values as per best measure and best k
	# Input: values, best_measure, best_k
	# Output: representative value
	if best_measure == "mean":
		representative_val = np.mean(values) + best_k * np.std(values)
	if best_measure == "median":
		median_val = np.percentile(values, 50)
		mad_val = np.percentile(np.abs(values - np.percentile(values, 50)), 50)
		representative_val = median_val + best_k * mad_val
	if best_measure == "quantile":
		representative_val = np.percentile(values, best_k)

	return representative_val
# =====================================================================
def linear_regression(n):
	"""
	"""
	# print('====', n)
	m = list(range(1,len(n)+1))
	#     print(m)
	import numpy as np
	from sklearn.linear_model import LinearRegression
	x = np.array(m).reshape((-1, 1))
	y = np.array(n)
	model = LinearRegression()
	try:
		model.fit(x, y)
	except Exception as e:
		print(e)
		raise(e)
	intercept = model.intercept_
	#     print(intercept)
	slope = model.coef_[0]
	# detrend values to get the representative value
	values = np.array(n)-(np.array(m)*slope)
	try:
		rep_value = compute_representative_value(values, ['mean', 'quantile', 'median'])
	except Exception as e:
		print(e)
		raise(e)
	# Add trend to the representative value
	rep_value = rep_value+(np.mean(m)*slope)
	return rep_value
# =====================================================================
def compute_festival_variation(arg_input_ts, arg_input_festival_list, input_temporal_pattern):
	CONST_MONTHS_UNDER_CONSIDERATION = 3
	# Add day of week and day of month:
	arg_input_ts['dow'] = [x.weekday() for x in arg_input_ts['timestamp']]
	arg_input_ts['dom'] = [x.day for x in arg_input_ts['timestamp']]
	arg_input_ts['festival'] = None
	arg_input_ts['variationPercentage'] = None
	festival_list_grouped = arg_input_festival_list.groupby(['festival'])
	DAYS_TO_LOOK = CONST_MONTHS_UNDER_CONSIDERATION*31
	# --------------------------------------------------
	for one_festival in festival_list_grouped:
		# Prepare a subset of each festival
		one_festival_name = one_festival[0]
		one_festival_subset = one_festival[1]

		one_festival_previous_values_subset = pd.DataFrame()
		one_festival_unique_dates = one_festival_subset['date'].unique()

		# Find data of previous months and next months CONST_MONTHS_UNDER_CONSIDERATION:
		for one_festival_one_date in one_festival_subset.groupby(['date']):
			one_festival_date = one_festival_one_date[0]

			# Find day of month of that particular month:
			one_date_value = one_festival_date.day
			one_dow_value = one_festival_date.weekday()

			# Find date range:
			one_date_months_back = one_festival_date - relativedelta(days=DAYS_TO_LOOK)

			# Find subset of values greater than and lesser than the lower and upper limit of dates:
			one_date_value_subset = arg_input_ts[(arg_input_ts['timestamp'] >= one_date_months_back) & 
											(arg_input_ts['timestamp']<= one_festival_date)]

			if one_date_value_subset.shape[0] == 0:
				continue

			# Prepare a subset of values corresponding to one date value:
			if input_temporal_pattern == "dom":
				one_date_value_subset = one_date_value_subset[one_date_value_subset['dom'] == one_date_value]      

			if input_temporal_pattern == "dow":
				one_date_value_subset = one_date_value_subset[one_date_value_subset['dow'] == one_dow_value]  

			if one_date_value_subset.shape[0]==0:
				continue

			# Mark festival date under consider:
			one_date_value_subset.loc[one_date_value_subset['timestamp'] == one_festival_date, 'festival'] = one_festival_name

			# Check if festival date has been marked or not
			# If one festival date is not present, continue
			check_festival_mark = one_date_value_subset[one_date_value_subset['festival'] == one_festival_name]
			if check_festival_mark.shape[0] == 0:
				continue

			# Compute difference between festival date values and values of same date of previous and next months
			one_festival_one_date_metric_value = one_date_value_subset.loc[
						one_date_value_subset['festival'] == one_festival_name, 'values'].values[0]


			# Find the difference between values of previous months and festival dates
			one_date_value_subset['absoluteVariation'] = one_date_value_subset['values'] - one_festival_one_date_metric_value
			one_date_value_subset['denominator'] = one_date_value_subset['values']

			# Remove current festival value
			one_festival_previous_values_subset = one_festival_previous_values_subset.append(one_date_value_subset)

		# Compute variation percentage:

		if one_festival_previous_values_subset.shape[0] == 0:
			continue

		one_festival_all_values_subset = one_festival_previous_values_subset[one_festival_previous_values_subset['festival'] == one_festival_name]
		# Create a seperate subset with current festival values removed
		one_festival_values_removed_subset = one_festival_previous_values_subset[one_festival_previous_values_subset['festival'] != one_festival_name]

		one_festival_sum_abs_variation =  np.sum(one_festival_values_removed_subset['absoluteVariation'])
		one_festival_sum_denom = np.sum(one_festival_values_removed_subset['denominator'])
		one_festival_percentageVariation = one_festival_sum_abs_variation/one_festival_sum_denom

		one_festival_input_data_index = one_festival_all_values_subset.index

		# Update one festival name and variation percentage in the list
		arg_input_ts.loc[one_festival_input_data_index,'variationPercentage'] = one_festival_percentageVariation
		arg_input_ts.loc[one_festival_input_data_index,'festival'] = one_festival_name

	# ----------------------------------------------
	columns_of_interest = ['timestamp', 'values', 'festival', 'variationPercentage']
	arg_input_ts = arg_input_ts[columns_of_interest]
	return arg_input_ts
# ==========================================================================
def adjust_festive_dates(each_segment_data, predicted_test, each_segment, festive_dates):
	"""
	"""
	input_data = each_segment_data.copy(deep = True)
	input_data = input_data[['date', 'apps']]
	input_data.columns = ['timestamp', 'values']
	festive_dates.loc[:, 'date'] = pd.to_datetime(festive_dates['date'], format = "%m/%d/%Y")
	arg_input_festival_list = festive_dates[festive_dates['future_flag'] == False]
	arg_input_festival_list = arg_input_festival_list[['festival', 'date']]
	# arg_input_festival_list['date'] = pd.to_datetime(arg_input_festival_list['date'], format = CONST_INPUT_DATE_FORMAT, errors = 'coerce')
	input_temporal_pattern = "dom"
	if each_segment == 1:
		input_temporal_pattern = "dow"

	# import pdb; pdb.set_trace()
	try:
		print('   Computing festivals significance...')
		return_data = compute_festival_variation(input_data, arg_input_festival_list, input_temporal_pattern)
		print('      Done!')
		# import pdb; pdb.set_trace()
	except Exception as e:
		print(e)
		raise(e)
	# print('---1')
	# return_data = return_data.dropna()
	print('   Adopting festivals significance...')
	return_data = return_data.reset_index(drop =True)
	return_data['year'] = return_data.timestamp.dt.year.tolist()
	return_data = return_data[return_data['year'] == 2018]
	return_data = return_data.reset_index(drop =True)
	return_data = return_data.reset_index(drop = True)
	return_data = return_data[['festival', 'variationPercentage', 'values']]
	festive_dates_future = festive_dates[festive_dates['future_flag'] == True]
	predicted_test['festive'] = ['']*predicted_test.shape[0]
	for i, j in festive_dates_future.iterrows():
		# import pdb; pdb.set_trace()
		kkk = j['date']
		kk = []
		kk.append(kkk)
		predicted_test.loc[predicted_test['application_date'].isin(kk),'festive'] = j['festival']

	for key, val in predicted_test.iterrows():
		
		if val['festive'] == '':
			continue

		# import pdb; pdb.set_trace()
		flag = festive_dates_future[festive_dates_future['festival'] == val['festive']]
		if flag.shape[0] == 0:
			continue

		flag = flag['date_change_flag'].values[0]
		if flag == False:
			new_value = return_data.loc[return_data['festival'] == val['festive'], 'values'].tolist()[0]
		else:
			current_value = val['case_count']
			diff = return_data.loc[return_data['festival'] == val['festive'], 'variationPercentage'].tolist()[0]
			new_value = round(current_value*(1-diff), 0)
			# print(val['festive'], 'is adjusted from', current_value,new_value)
			if pd.isnull(new_value):
				continue
		predicted_test.loc[predicted_test['id'] == val['id'], 'case_count'] = new_value

	predicted_test = predicted_test.drop(['festive'], axis = 1)
	print('      Done!')
	return predicted_test




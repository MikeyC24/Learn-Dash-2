import pandas as pd
import numpy as np
import datetime, sys, time, re, io, os
import math
import timeit
import requests
from fred import Fred
from dateutil import relativedelta
from datetime import timedelta 
import matplotlib.pyplot as plt 
#from sklearn.cluster import KMeans


class Dashboard:

	def __init__(self, key, **kwargs):
		self.fred = Fred(api_key = key, response_type='df')
		self.data_array = kwargs.get('data_array', None)
		self.test_dict = kwargs.get('test_dict', None)
		self.group_interval = 1

	def get_dfs_from_fred(self):
		"""
		method, from class vars, builds df from wither test data (pull from csv)
		or data generated from fed 

		# thisn will change in end product, for now working off of csv 
		"""
		dict_df = {}
		if self.data_array is not None:
			fred_api_calls = {
			'Real GDP':'GDPC1', 'Federal Funds Rate':'FEDFUNDS',
			'10 YR Treasury':'DGS10', 'S&P500':'SP500', 'Moody Baa Yield':'BAA',
			'Moody Aaa Yield':'AAA', '30 Year Mortgage':'MORTGAGE30US',
			'Debt Percent GDP':'GFDEGDQ188S'
			}
			for data in self.data_array:
				api_call = fred_api_calls[data]
				dict_df[data] = self.fred.series.observations(api_call).set_index('date')
			return dict_df
		if self.test_dict is not None:
			path = self.test_dict['path']
			for data in self.test_dict['data_array']:
				df_path = os.path.join(path, data)
				dict_df[data] = pd.read_csv(df_path, index_col='date')
			return dict_df

	def write_fred_data_to_csv_from_dict(self, dict, save_path,):
		print('getting data')
		drop_cols = ['realtime_end', 'realtime_start']
		for name, dict_vars in dict.items():
			# get dataframe
			data = self.fred.series.observations(dict_vars['FRED var'])
			data = data.drop(drop_cols, axis=1)
			data['Observation Start'] = dict_vars['Start Date']
			data['Observation End'] = dict_vars['End Date']
			data['Seasonaly Adjusted'] = dict_vars['Frequency']
			data['Type'] = dict_vars['Type']
			print(name)
			csv_path  = os.path.join(save_path, name)
			print(csv_path)
			data.to_csv(csv_path)

	def saved_csvs_to_dict(self, array, path):
		return_dict = {}
		for name in array:
			return_dict[name] = pd.read_csv(os.path.join(path, name)).set_index('date')
		return return_dict

	def get_shortest_time_frame_metric(self, dict, freq_unit):
		"""
		method is is sued to combined mutliple measures to one dataframe, 
		it picks the metrhic with shortest time inreval and returns
		the name of shortest index and each dataframe labeled by month
		"""
		if freq_unit == 'M':
			freq_unit = 'MS'
		for name,df in dict.items():
			#df = df.set_index('date')
			df.index = pd.to_datetime(df.index)
			#print(df.index[0],df.index[-1])
			time_series = pd.date_range(df.index[0],df.index[-1], freq=freq_unit)
			#print(time_series)
			df = df.reindex(time_series)
			#print(name, len(df))
			dict[name] = df
		# get long time series
		index_name = self.GetMinFlox(dict)[1]
		return index_name, dict

	def align_time_frames(self, dict, name, freq_unit, group_unit='mean'):
		"""
		aligns dicts by time frames based on data frame, typically the shortest one
		fed from prior method, also returns min, max ans average of period 
		"""
		agg_dict = {}
		index_use = dict[name]
		if freq_unit == 'M':
			freq_unit = 'MS'
		for name, df in dict.items():
			time_series = pd.date_range(index_use.index[0],index_use.index[-1], freq=freq_unit)
			#print('time_series',time_series)
			df = df.reindex(time_series)
			#print('df', df)
			array = list(df.columns)
			array.remove('value')
			df = df.drop(array,axis=1)
			df[name + ' Value'] = df['value']
			agg_dict[name + ' Value'] = group_unit
			"""
			df[name + ' Min'] = df['value']
			df[name + ' Max'] = df['value']
			df[name + ' Average'] = df['value']
			agg_dict[name + ' Min'] = 'min'
			agg_dict[name + ' Max'] = 'max'
			agg_dict[name + ' Average'] = 'mean'
			"""
			df = df.drop('value',axis=1)
			#print(df)
			dict[name] = df
		return dict, agg_dict

	def fill_mising(self, dict):
		"""
		fills missing data for all dfs in dict
		"""	
		for name, df in dict.items():
			df = df.fillna(method='pad')
			dict[name] = df
		return dict 

	def combine_dict_to_df(self, dict):
		"""
		comined dfs from dict
		"""
		return pd.concat(dict.values(), axis=1)

	# method groups dict by wanted time period,
	# also creates a col with name accross 
	def combine_dict(self, df, agg_dict, interval):
		"""
		takes in add dict which is tells method to group my mean, min or max
		interval which groups by wante dtime peruod interval 
		and returns df with these groupers
		also returns label with range in string, human form 
		"""
		self.group_interval = float(interval[:-1])
		df = df.groupby(pd.Grouper(freq=interval)).agg(agg_dict)
		df['Time Period'] = df.apply(self.calculate_period, axis=1)
		#df = df.set_index('Time Period')
		return df

	# helpers
	def GetMaxFlox(self,flows): 
		# return dict len and name for ll values in dict
		return max((len(v), k) for k,v in flows.items())

	def GetMinFlox(self,flows): 
		# return dict len and name for all values in dict
		return min((len(v), k) for k,v in flows.items())

	def calculate_period(self, row):
		# creates human raeadable col using strings 
		return row.name.strftime('%m-%Y') +'--'+ (row.name +datetime.timedelta(self.group_interval*365/12)).strftime('%m-%Y')

	def drop_cols_by_group(self, df, wanted_group):
		# takes in dataframe and array of wanted values
		# used to only, min, average, or max
		wanted_group = wanted_group.capitalize() 
		if wanted_group == 'all':
			return df
		else:
			drop_cols = []
			for col in df.columns:
				if wanted_group not in col:
					drop_cols.append(col)
			df = df.drop(drop_cols,axis=1)
			return df

	# method to grab dataframe together by set time interval
	def get_full_group_df(self,dict, time_interval='6M', group_unit='mean'):
		"""
		combines many method to return combined df, based on shorest time frame of all vals
		in internval given by user with a time period col fo human reading 
		"""
		freq_unit = time_interval[-1]
		index_name, dict = self.get_shortest_time_frame_metric(dict, freq_unit)
		dict, agg_dict = self.align_time_frames(dict, index_name, freq_unit, group_unit=group_unit)
		dict = self.fill_mising(dict)
		# loses datetime concept for groups after here 
		df = self.combine_dict_to_df(dict)
		df = self.combine_dict(df, agg_dict, time_interval)
		df = df.iloc[1:,:]
		#df = self. drop_cols_by_group(df, 'Average')
		return df

	def run_program(self):
		start =os.path.dirname(os.path.realpath(sys.argv[0])) 
		dict = self.get_dfs_from_fred()
		print(dict.keys())
		#for k,v in dict.items():
		#	save_path = os.path.join(start, 'fed_graphs', k)
		#	v.to_csv(save_path)
		#self.get_full_group_df(dict)

	# reorg dfs to drop all cols but value and pad fill nas, keep return date
	def isolate_df_data(self, dict_df):
		"""
		# drops all cols from Fred given data execpt for value with name in df and keeps in dict
		"""
		return_dict_df = {}
		for k,v in dict_df.items():
			new_df = pd.DataFrame(index=v.index)
			new_df[k + ' Value'] = v['value']
			return_dict_df[k] = new_df
		return return_dict_df

	def plot_wanted_cols(self, df, cols_wanted_array):
		"""	
		plots all wanted cols on same plot, takes in prgoram var fo each data
		such as Federal Fund Rates and keeps all cols with them, done thsi way, 
		even though slower, bc col may have different name sucha s wanted val plus average or mean
		"""
		cols_to_plot = {}
		for col_wanted in cols_wanted_array:
			for col in df.columns:
				if col_wanted in col:
					cols_to_plot[col] = df[col]
		for k,v in cols_to_plot.items():
			plt.plot(df.index, v, label=k)
		plt.title("Simple Plot")
		plt.legend()
		plt.show()

	def get_yield_spreads(self,df, cols_against, base, graph='yes'):
		"""
		can only take in data with yields/rates, ptakes in a combined df,
		a base col to comapre against all and an array of cols wanted
		array takes in program var names and finds cols with them in it 
		then returns dict with spreads and key is titel
		and plots if user wants 
		"""
		cols_to_plot = {}
		for col_wanted in cols_against:
			for col in df.columns:
				if col_wanted in col:
					cols_to_plot[col] = df[col]
		for col in df.columns:
			if base in col:
				cols_to_plot[base] = df[col]
		base_array = cols_to_plot[base]
		spread_dicts = {}
		for name in cols_to_plot:
			if name != base:
				title = name + 'spread_v_' + base 
				base_line = cols_to_plot[base]
				spreads  = cols_to_plot[name]
				spread_dicts[title] = [a - b for a, b in zip(spreads,base_line)]
				plt.plot(df.index, spread_dicts[title], label=title)
		if graph == 'yes':
			plt.legend()
			plt.show()
		return spread_dicts

	def get_yield_spreads_new(self,df, cols_against, base, graph='yes'):
		"""
		can only take in data with yields/rates, ptakes in a combined df,
		a base col to comapre against all and an array of cols wanted
		array takes in program var names and finds cols with them in it 
		then returns dict with spreads and key is titel
		and plots if user wants 
		"""
		cols_to_plot = {}
		for col in cols_against:
			cols_to_plot[col] = df[col+' Value']
		cols_to_plot[base] = df[base + ' Value']
		base_array = cols_to_plot[base]
		spread_dicts = {}
		for name in cols_to_plot:
			if name != base:
				title = name + '_spread_v_' + base 
				base_line = cols_to_plot[base]
				spreads  = cols_to_plot[name]
				spread_dicts[title] = [a - b for a, b in zip(spreads,base_line)]
				plt.plot(df.index, spread_dicts[title], label=title)
		if graph == 'yes':
			plt.legend()
			plt.show()
		return spread_dicts

	# needed next 
	"""
	method to return if col is pos or neg for 1 or 2
	method to compare to prior periods aka, higher than last period,
	last 5 periods, should be an array with each cal number of periods ahead or back wanted (neg for ahead)
	use the Nber states to lay out recession/booms
	with months start and end in better form and severaity
	# then have method to go through cateforated by group and assign 1-4 
	for in rec, in boom, chnage to boom, change to rec for time period 
	to do this will need to take index as date time and add user wanted interval
	then build array thats add to is time period in each of 4 
	and give that back as var can have a 5 - 8 for more than 2 booms and for what it went it 
	as and what it exited, this could be braod macro category to look at
	maybe use this to fund probabilty of recession/boon and use this for stress testing 
	
	could also add other cats for unqie time period such as stagflation, war, etc....
	crissis, tech boom, us savings and loans crisis 

	could also provide stats such as in x percent of reces/boons this event happened
	and could have user input, ie, rate higher than another rate, inflation measure etc

	method to calculate if variable was leading or lagging during time period, ie user inputed 
	var changed before of after change, such as spread or inflation v deflation

	for some vars it could go by rate of change 

	was a new high reached 

	"""

	

	def calculate_pos_neg_cat(self, row):
		return 1 if row > 0 else 0 
			
# get vars to save to csv, input will always be from dict to stat/filter classes to mimic live calls and cross over for each
# will get csv for histoic to not constantly ping site 

			

# for now just doing spreads



fred_econ_data = {
	'Federal Funds Rate':{
	'Start Date':'1954-07-01',
	'End Date':'2018-11-01',
	'Frequency':'Monthly',
	'Type':'Interest Rate',
	'Seasonaly Adjusted':'No',
	'FRED var':'FEDFUNDS',
	'Summary':'NA'
	},
	'10 YR Treasury':{
	'Start Date':'1962-01-02',
	'End Date':'2018-12-06',
	'Frequency':'Daily',
	'Type':'Interest Rate',
	'Seasonaly Adjusted':'No',
	'FRED var':'DGS10',
	'Summary':'NA'
	},
	'Moody Aaa Yield':{
	'Start Date':'1919-01-01',
	'End Date':'2018-11-01',
	'Frequency':'Monthly',
	'Type':'Interest Rate',
	'Seasonaly Adjusted':'Yes',
	'FRED var':'AAA',
	'Summary':'NA'
	},
	'Moody Baa Yield':{
	'Start Date':'1919-01-01',
	'End Date':'2018-11-01',
	'Frequency':'Monthly',
	'Type':'Interest Rate',
	'Seasonaly Adjusted':'Yes',
	'FRED var':'BAA',
	'Summary':'NA'
	},
	'30 Year Mortgage':{
	'Start Date':'1971-04-02',
	'End Date':'2018-12-06',
	'Frequency':'Weekly',
	'Type':'Interest Rate',
	'Seasonaly Adjusted':'No',
	'FRED var':'MORTGAGE30US',
	'Summary':'NA'
	}
}
inflation_fred_vars ={
	'Flexible Price Consumer Price Index less F&E':{
	'Start Date':'1967-01-01',
	'End Date':'2018-10-01',
	'Frequency':'Monthly',
	'Type':'Percent',
	'Seasonaly Adjusted':'Yes',
	'FRED var':'COREFLEXCPIM158SFRBATL',
	'Summary':'Products in CPI that change quickly less food and energy. This incorporates less of expected inflation '
	},
	'Prices for Personal Consumption less F&E':{
	'Start Date':'1959-02-01',
	'End Date':'2018-10-01',
	'Frequency':'Monthly',
	'Type':'Percent Change',
	'Seasonaly Adjusted':'Yes',
	'FRED var':'DPCCRGM1M225SBEA',
	'Summary':'THis is chained price index'
	},
	'Inflation Expectation':{
	'Start Date':'1978-01-01',
	'End Date':'2018-10-01',
	'Frequency':'Monthly',
	'Type':'Percent',
	'Seasonaly Adjusted':'No',
	'FRED var':'MICH',
	'Summary':'Median expected price change next 12 months, Surveys of Consumers. '
	},
	'Inflation, consumer prices':{
	'Start Date':'1960-01-01',
	'End Date':'2017-01-01',
	'Frequency':'Annual',
	'Type':'Percent',
	'Seasonaly Adjusted':'Yes',
	'FRED var':'FPCPITOTLZGUSA',
	'Summary':'Inflation as measured by the consumer price index reflects the annual percentage change in the cost to the average consumer of acquiring a basket of goods and services that may be fixed or changed at specified intervals, such as yearly. The Laspeyres formula is generally used. '
	},
	'Median Consumer Price Index':{
	'Start Date':'1983-01-01',
	'End Date':'2018-10-01',
	'Frequency':'Monthly',
	'Type':'Percent',
	'Seasonaly Adjusted':'Yes',
	'FRED var':'MEDCPIM158SFRBCLE',
	'Summary':'Median Consumer Price Index (CPI) is a measure of core inflation calculated the Federal Reserve Bank of Cleveland and the Ohio State University. Median CPI was created as a different way to get a “Core CPI” measure, or a better measure of underlying inflation trends. '
	},
}

gdp_vars ={
	'Real Gross Domestic Product Change':{
	'Start Date':'1947-04-01',
	'End Date':'2018-07-01',
	'Frequency':'Quarterly',
	'Type':'Percent Change',
	'Seasonaly Adjusted':'Yes',
	'FRED var':'A191RL1Q225SBEA',
	'Summary':'inflation adjusted'
	},
	'Total Public Debt as Percent of Gross Domestic Product':{
	'Start Date':'1966-01-01',
	'End Date':'2018-07-01',
	'Frequency':'Quarterly',
	'Type':'Percent',
	'Seasonaly Adjusted':'Yes',
	'FRED var':'GFDEGDQ188S',
	'Summary':''
	},
	' Federal Surplus or Deficit as Percent of Gross Domestic Product':{
	'Start Date':'1929-01-01',
	'End Date':'2017-01-01',
	'Frequency':'Annual',
	'Type':'Percent',
	'Seasonaly Adjusted':'No',
	'FRED var':'FYFSGDA188S',
	'Summary':''
	},
	'Federal Outlays: Interest as Percent of Gross Domestic Product':{
	'Start Date':'1940-01-01',
	'End Date':'2017-01-01',
	'Frequency':'Quarterly',
	'Type':'Percent Change',
	'Seasonaly Adjusted':'No',
	'FRED var':'FYOIGDA188S',
	'Summary':''
	},
	'Civilian Unemployment Rate':{
	'Start Date':'1948-01-01',
	'End Date':'2018-11-01',
	'Frequency':'Monthly',
	'Type':'Percent',
	'Seasonaly Adjusted':'Yes',
	'FRED var':'UNRATE',
	'Summary':''
	},
	'Gross Domestic Product (without inflation)':{
	'Start Date':'1947-04-01',
	'End Date':'2018-07-01',
	'Frequency':'Quarterly',
	'Type':'Percent Change',
	'Seasonaly Adjusted':'yes',
	'FRED var':'A191RP1Q027SBEA',
	'Summary':'not inflation addjusted'
	},
	'Gross National Product ':{
	'Start Date':'1947-04-01',
	'End Date':'2018-07-01',
	'Frequency':'Quarterly',
	'Type':'Percent Change',
	'Seasonaly Adjusted':'yes',
	'FRED var':'A001RP1Q027SBEA',
	'Summary':''
	},
}

# temp methid to get graph for testing in routes

def return_combine_df_for_graph():
	"""
	rly this is just for ir rates 
	"""

	key = '18c95216b1230de68164158aeb02e2c2'
	# bade login with key
	base = Dashboard(key)
	# get csv with write vars
	start =os.path.dirname(os.path.realpath(sys.argv[0])) 
	path  = os.path.join(start, 'Fred Graphs')
	# this path was used for flask,anothe day to fix this one 
	#path = '/home/mike/Documents/coding_all/ModelApp/app/Fred Graphs'
	#base.write_fred_data_to_csv_from_dict(fred_econ_data, path)

	# convert csv to dict to use
	df_dict = base.saved_csvs_to_dict(fred_econ_data.keys(), path)

	# skipped step, drop down, whyich can be typed into bc at some point will have list of all vars
	# and display graph indivusal with relevent data (type, seaonailty) displayed liek fed

	# next combine wanted vars to single df
	combined_df = base.get_full_group_df(df_dict, time_interval='6M', group_unit='mean')
	# get spreads for IR rates
	cols_against = ['10 YR Treasury','Moody Aaa Yield','Moody Baa Yield','30 Year Mortgage', ]
	base_col = 'Federal Funds Rate'
	spread_dict = base.get_yield_spreads_new(combined_df, cols_against, base_col, graph='no')
	spread_dict['date'] = combined_df.index
	combined_spread_df = pd.DataFrame.from_dict(spread_dict)
	combined_spread_df = combined_spread_df.set_index('date')
	combined_spread_df.index = pd.to_datetime(combined_spread_df.index)
	return combined_df, combined_spread_df


def write_new_data_to_csv(dict):
	#print('ere')
	key = '18c95216b1230de68164158aeb02e2c2'
	# bade login with key
	base = Dashboard(key)
	# get csv with write vars
	start =os.path.dirname(os.path.realpath(sys.argv[0])) 
	path  = os.path.join(start, 'Fred Graphs')
	print(path)
	#/home/mike/Documents/coding_all/Learn-Dash/Dispatcher/reports/Fred Graphs
	base.write_fred_data_to_csv_from_dict(dict, path)

inflation_vars= ['Inflation, consumer prices', 'Prices for Personal Consumption less F&E',
'Flexible Price Consumer Price Index less F&E', '10 YR Treasury', 'Federal Funds Rate']
start =os.path.dirname(os.path.realpath(sys.argv[0])) 
path  = os.path.join(start, 'Fred Graphs')

def get_basic_combined_df_for_dash(array,path):
	key = '18c95216b1230de68164158aeb02e2c2'
	# bade login with key
	base = Dashboard(key)
	df_dict = base.saved_csvs_to_dict(array, path)
	combined_df = base.get_full_group_df(df_dict, time_interval='6M', group_unit='mean')
	return combined_df

def get_spreads(df, base_col, cols_against):
	key = '18c95216b1230de68164158aeb02e2c2'
	# bade login with key
	base = Dashboard(key)
	spread_dict = base.get_yield_spreads_new(df, cols_against, base_col, graph='no')
	spread_dict['date'] = df.index
	combined_spread_df = pd.DataFrame.from_dict(spread_dict)
	combined_spread_df = combined_spread_df.set_index('date')
	return combined_spread_df

def get_dfs_for_display_inflation():
	inflation_vars= ['Inflation, consumer prices',
	'Flexible Price Consumer Price Index less F&E', '10 YR Treasury', 'Federal Funds Rate']
	start =os.path.dirname(os.path.realpath(sys.argv[0])) 
	path  = os.path.join(start, 'Fred Graphs')
	inflation_df = get_basic_combined_df_for_dash(inflation_vars,path)
	inflation_spreads = get_spreads(inflation_df, 'Inflation, consumer prices',inflation_vars[1:] )
	return inflation_df, inflation_spreads


percent_change_vars = ['Prices for Personal Consumption less F&E', 'Real Gross Domestic Product Change']

def get_gdp_graphs():
	key = '18c95216b1230de68164158aeb02e2c2'
	# bade login with key
	base = Dashboard(key)
	gdp_percent_change_vars = ['Real Gross Domestic Product Change', 'Federal Outlays: Interest as Percent of Gross Domestic Product','Civilian Unemployment Rate'  ,'Gross Domestic Product (without inflation)']
	gdp_percent_vars = ['Total Public Debt as Percent of Gross Domestic Product', ' Federal Surplus or Deficit as Percent of Gross Domestic Product','Civilian Unemployment Rate', 'Gross National Product']
	start =os.path.dirname(os.path.realpath(sys.argv[0])) 
	path  = os.path.join(start, 'Fred Graphs')
	df_dict_change = base.saved_csvs_to_dict(gdp_percent_change_vars, path)
	df_dict_percent = base.saved_csvs_to_dict(gdp_percent_vars, path)
	df_gdp_change = base.get_full_group_df(df_dict_change, time_interval='6M', group_unit='mean')
	df_gdp_percent = base.get_full_group_df(df_dict_percent, time_interval='6M', group_unit='mean')
	return df_gdp_percent, df_gdp_change

#print(inflation_spreads)
"""
sys.exit()	

#print(run)



vars=[
'Real GDP','Federal Funds Rate',
		'10 YR Treasury', 'Moody Baa Yield',
		'Moody Aaa Yield', '30 Year Mortgage',
		'Debt Percent GDP', 
		]
# 'S&P500'
start =os.path.dirname(os.path.realpath(sys.argv[0])) 
path = path = os.path.join(start, 'fed_graphs')

test_dict ={
	'path':path,
	'data_array':vars
}

key = '18c95216b1230de68164158aeb02e2c2'
#base = Dashboard(key, data_array=vars)
# base set up and dfs from Fred
base = Dashboard(key, test_dict=test_dict)
df_dict_from_fred = base.get_dfs_from_fred()
# combining dfs
combined_df = base.get_full_group_df(df_dict_from_fred, time_interval='1M')
#print(combined_df)
#combined_df.plot()
# get spreads amongst columns
spreads_dict_vals = base.get_yield_spreads(combined_df, ['10 YR Treasury', 'Moody Baa Yield',
		'Moody Aaa Yield','30 Year Mortgage'], 'Federal Funds Rate', graph='no')
for name, values in spreads_dict_vals.items():
	print(name)
	print(len(values))
	pos_array = []
	neg_array = []
	for x in values:
		pos_array.append(x) if x >0 else neg_array.append(x)
	print('pos', (len(pos_array)))
	print('neg', (len(neg_array)))

def pos_neg_cat(item):
	return 1 if item > 0 else 0 
df = pd.DataFrame.from_dict(spreads_dict_vals)
df.index = combined_df.index
df = df.dropna()
for col in df.columns:
	df[col + 'cat'] = df[col].apply(pos_neg_cat)
df.to_csv('check')

sys.exit()

# plot wanted cols from df
base.plot_wanted_cols(combined_df, ['Federal Funds Rate',
		'10 YR Treasury', 'Moody Baa Yield',
		'Moody Aaa Yield','30 Year Mortgage'])
sys.exit()

for col in combined_df:
	combined_df.index= pd.to_datetime(combined_df.index)
	graph = pd.Series(combined_df[col], index=combined_df.index)
	graph.plot()
	plt.show()
sys.exit()



sys.exit()
df_dict = base.isolate_df_data(df_dict_from_fred)
# combined df



for name, df in df_dict.items():
	print(name)
	from pandas.plotting import autocorrelation_plot
	autocorrelation_plot(df)
	#df.plot()
	plt.show()
"""
"""

dict has dfs with all dates and data is filled, 
next step is to either combine dfs 
figuring out how to analyze and graph


"""
"""
sys.exit()
#df  = base.run_program()
#df  = base.get_full_group_df(dict)
drop_cols = ['value', 'realtime_end', 'realtime_end']
start =os.path.dirname(os.path.realpath(sys.argv[0])) 
for var in vars:
	path = os.path.join(start, 'fed_graphs', var)
	df = pd.read_csv(path, index_col='date')
	new_df = pd.DataFrame(index=df.index)
	new_df[var + ' value'] = df['value']
	# pad pulls the last data and is very commin in time series 
	new_df = new_df.fillna(method='pad')
	print(new_df)

"""



"""
dict ={
'tr10':pd.read_csv('10 YR Treasury',index_col='date'),
'gdp':pd.read_csv('Real GDP',index_col='date'),
'sp':pd.read_csv('S&P500',index_col='date'),
'fedrate':pd.read_csv('Federal Funds Rate',index_col='date')
}

sys.exit()
"""


"""
idk 
from sklearn.cluster import KMeans
df_new = pd.DataFrame.from_dict(spreads_dict_vals)
n_clusters = 2
for col in df_new.columns:
    kmeans = KMeans(n_clusters=n_clusters)
    #X = list(df_new[col]).reshape(-1, 1)
    X = np.reshape(list(df_new[col]), (-1, 1))
    kmeans.fit(X)
    print("{}: {}".format(col, kmeans.predict(X)))



"""


# good example for clustering 
# https://www.kaggle.com/dhanyajothimani/basic-visualization-and-clustering-in-python


"""
good fred measures 
Median Consumer Price Index (MEDCPIM158SFRBCLE) - 1983-01-01, 2018-10-01
Inflation, consumer prices for the United States (FPCPITOTLZGUSA) 1960 - 2017
University of Michigan: Inflation Expectation (MICH) 1978 - 2018
Prices for Personal Consumption Expenditures: Chained Price Index: PCE excluding food and energy (DPCCRGM1M225SBEA) 1959 - 2018
Flexible Price Consumer Price Index less Food and Energy (COREFLEXCPIM157SFRBATL) 1967 - 2018 
https://fred.stlouisfed.org/series/USSLIND 1982 - 2018 
Industrial Production: Manufacturing (NAICS) (IPMAN) 1972 - 2018
Industrial Production Index (INDPRO) 1919 - 2018
Civilian Unemployment Rate (UNRATE) 1948 - 2018 
4-Week Moving Average of Initial Claims (IC4WSA) 1967 - 2018 
Real Gross Domestic Product (A191RL1Q225SBEA) (percent chnage) 1947 - 2018 
https://fred.stlouisfed.org/series/GFDEGDQ188S  Federal Debt: Total Public Debt as Percent of Gross Domestic Product (GFDEGDQ188S) 1966-01-01 - 2018-07-01 quartely
https://fred.stlouisfed.org/series/FYFSGDA188S Federal Surplus or Deficit [-] as Percent of Gross Domestic Product (FYFSGDA188S) 1929-01-01 - 2017-01-01 annual 
https://fred.stlouisfed.org/series/FYOIGDA188S Federal Outlays: Interest as Percent of Gross Domestic Product (FYOIGDA188S) 1940-01-01 - 2017-01-01 annual 
"""


"""

1) productivity growth, 2) the “long wave” cycle and 3) the business/market cycle. 
"""


"""
what has contributed to gdp over time
https://fred.stlouisfed.org/categories/106?t=percent&ob=pv&od=desc

"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 00:33:20 2019

@author: Wonbin Kang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ri = pd.read_csv('police.csv')
ri.columns
ri.head()

## Check the NaNs
print(ri.isnull().sum())

# .drop() entire columns
ri.state.value_counts()
ri.drop(['state', 'county_name'], axis = 'columns', inplace = True) # sum() by default is by columns 

# .dropna() entire rows from DF
ri.dropna(subset = ['driver_gender'], inplace = True)
print(ri.isnull().sum()) # Note that dropping the driver_gender NaN rows cleaned up the other rows. Note that search_type also has a very large proportion of NaNs but this is due to there being no search being conducted at time of the stop, i.e. when search_conducted is False. 

## Check the datatypes of each column of DF
ri.dtypes # is_arrested should be bool; stop_date and stop_time should be combined and converted to datetime objects.

# ri.is_arrested
ri.is_arrested.head() # Entries are True and False
ri.is_arrested.dtypes # dtype('O')
ri['is_arrested'] = ri.is_arrested.astype('bool') # Note that when creating or writing over a column, you must use bracket notation and dot notation can't be used. 

# ri.stop_date and ri.stop_time
# method 1
ri['stop_datetime'] = pd.to_datetime(ri['stop_date'] + ' ' + ri['stop_time'], format = '%Y-%m-%d %H:%M')
ri.head()
ri.dtypes

# method 2: DC method
ri.drop('stop_datetime', axis = 1, inplace = True)
ri['stop_datetime'] = ri.stop_date.str.cat(ri.stop_time, sep = ' ')
ri['stop_datetime'] = pd.to_datetime(ri.stop_datetime)
ri.head()
ri.dtypes

# Set the new datetime column as index:
ri.set_index('stop_datetime', inplace = True)
ri.index # 2005-01-04 to 2015-12-31
ri.dtypes

# Set driver_gender and driver_race as categories
ri[['driver_gender', 'driver_race']] = ri[['driver_gender', 'driver_race']].astype('category', inplace = True)

## Proportion of the types of violations stopped for men and women
female = ri[ri.driver_gender == "F"]
male = ri[ri.driver_gender == "M"]
female.violation.value_counts(normalize = True)
male.violation.value_counts(normalize = True)

## Do females get away with warnings more than citations than males when stopped for speeding? 
female_speeding = ri[(ri.driver_gender == "F") & (ri.violation == "Speeding")]
male_speeding = ri[(ri.driver_gender == "M") & (ri.violation == "Speeding")]
print(female_speeding.stop_outcome.value_counts(normalize = True))
print(male_speeding.stop_outcome.value_counts(normalize = True))
# Note that Citation is very similar between M and F

## search_type had a lot of NaNs because there is no entry if a search was not conducted during a stop. closer look:
ri.search_type.value_counts(dropna = False) # To see the NaN counts

# How many incidence of 'Protective Frisk'? 
frisk = ri.search_type.str.contains('Protective Frisk', na = False) # In creating the bool, code NaNs as False

## Are males frisked more than females? Note to subset the DF to only those obs where a search was conducted.
ri['frisk'] = frisk
searched = ri[ri.search_conducted]
searched.groupby('driver_gender').frisk.mean()

## Using the datetime object as an index and using the dt accessor. Note that a datetime index has access to the .dt accessor, but uses .index accessor instead
# arrests by time of day:
ri.is_arrested.mean() # 3.557% of stops lead to an arrest
hourly_arrest_rate = ri.groupby(ri.index.hour).is_arrested.mean() # Note that .dt.{} is intuitive and uses natural words (week, hour, day, weekday, to extract)
hourly_arrest_rate.plot()
plt.xlabel('Hour')
plt.ylabel('Arrest Rate')
plt.title('Arrest Rate by Time of Day')
plt.show()

## Resampling, concatenation (again), and subplots
ri.drugs_related_stop.resample("A").mean().plot()
plt.show()

annual_drugs_stops = ri.drugs_related_stop.resample("A").mean()
annual_search_rate = ri.search_conducted.resample("A").mean()
annual = pd.concat([annual_drugs_stops, annual_search_rate], axis = 1)
annual.plot(subplots = True)
plt.show()

## pd.crosstab(): frequency table for two discrete variables; and barplots
all_zones = pd.crosstab(index = ri.district, columns = ri.violation); print(all_zones)
k_zones = all_zones.loc['Zone K1':'Zone K3', : ]
k_zones.plot(kind = 'bar')
plt.show()

k_zones.plot(kind = 'bar', stacked = True)
plt.show()

## Converting dtypes without astype() using map and dictionaries. 
ri.stop_duration.dtypes # object
ri.stop_duration.value_counts() # 0-15 Min, 16-30 Min, 30+ Min

# Convert to integers using mapping dictionary and .map()
mapping_dict = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
ri['stop_minutes'] = ri.stop_duration.map(mapping_dict)
ri['stop_minutes'].unique() # type integer and 3 unique values

## bar plot options
ri.violation_raw.dtypes
ri.violation_raw.value_counts() # a more detailed version of violation
stop_length = ri.groupby('violation_raw').stop_minutes.mean()

stop_length.sort_values().plot(kind = 'barh')
plt.show()

## A bit confusing, but let's create a new 'stop_length' column in ri DF using the categories from stop_duration which is an ORDERED category column:
# Create a new mapping_dict to use in the .map() call:
mapping_dict = {'0-15 Min':'short', '16-30 Min':'medium', '30+ Min':'long'}

# map() the dict to stop_duration column (object column) and create new ordered category column:
# The following code has been deprecated:
ri['stop_length'] = ri.stop_duration.map(mapping_dict).astype('category', ordered = True, categories = ['short', 'medium', 'long'])
ri.drop('stop_length', axis = 1, inplace = True)

# From the FutureWarning and pandas documentation, use the following:
ri['stop_length'] = ri.stop_duration.map(mapping_dict).astype(pd.CategoricalDtype(categories = ['short', 'medium', 'long'], ordered = True))
ri.stop_length.dtypes # CategoricalDtype(categories=['short', 'medium', 'long'], ordered=True)
ri.stop_length.head()

## Introduce the weather data to see whether weather has an effect on police stops
weather = pd.read_csv('weather.csv')
weather.columns
weather.dtypes # DATE to datetime and set as index???
weather.info() # Note that TAVG has NaNs
weather.TAVG.isna().sum() #2800 NaNs. Possibly use .interpolate() or .fillna()? Other method? 

# Temps summary statistics and EDA
weather.loc[:, 'TAVG':'TMAX'].describe()
weather.loc[:, 'TAVG':'TMAX'].plot(kind = 'box')
plt.show() # no outliers
weather['TDIFF'] = weather.TMAX - weather.TMIN
print(weather.TDIFF.describe())

weather.TDIFF.plot(kind = 'hist', bins = 20)
plt.show()


## The weather DataFrame contains 20 columns that start with 'WT', each of which represents a bad weather condition. For example:
# WT05 indicates "Hail"
# WT11 indicates "High or damaging winds"
# WT17 indicates "Freezing rain"
# For every row in the dataset, each WT column contains either a 1 (meaning the condition was present that day) or NaN (meaning the condition was not present).
weather['bad_conditions'] = weather.loc[:, 'WT01':'WT22'].sum(axis = 1)
weather.bad_conditions.value_counts(dropna = False) # Values range from 0 to 9 (int)
weather.bad_conditions.isnull().sum() # There are no NaN values, but DC wants to set possible NaNs to 0
weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')

## Create a bad weather rating system:
mapping_dict = {0:'good', 1:'bad', 2:'bad', 3:'bad', 4:'bad', 5:'worse', 6:'worse', 7:'worse', 8:'worse', 9:'worse'}
weather['rating'] = weather.bad_conditions.map(mapping_dict).astype(pd.CategoricalDtype(categories = ['good', 'bad', 'worse'], ordered = True))
weather.rating.dtypes
weather.rating.head() 
weather.rating.value_counts().sort_index() # value_counts() sorts by largest number to smallest. sort_index() sorts the index by order (if ordered category) or alphabetically. Here it is ordered. 

## Note that ri is datetime indexed. If merged with weather (no datetime index), the index and therefore, the information stored in the combined stop_date and stop_time column is lost. Therefore, .reset_index() to return to rangeIndex. 
ri.reset_index(inplace = True)
ri_weather = pd.merge(left = ri, 
                      right = weather[['DATE', 'rating']], 
                      how = 'left', 
                      left_on = 'stop_date', 
                      right_on = 'DATE') # Note that we're joining using two string columns of dates which luckily matched. This will seldom be the case and these should be changed to datetime objects. 
ri_weather.set_index('stop_datetime',inplace = True)

## Do police officers arrest drivers more often when the weather is bad?
# Calculate arrest rate:
ri_weather.is_arrested.mean()

# Calculate arrest rate for each weather rating:
ri_weather.groupby('rating').is_arrested.mean()

# Calculate arrest rate for every permutation of violation and weather rating:
arrest_rate = ri_weather.groupby(['violation', 'rating']).is_arrested.mean()

# Access data from the multi index series arrest_rate. Note that this isn't a multi index DF and so tuples for the row multi index is not necessary, but use them anyway for consistency. 
print(arrest_rate[('Registration/plates', 'worse')])
print(arrest_rate['Speeding'])
print(arrest_rate[(slice(None), 'worse')])

# Since we're dealing with a multi index SERIES, we can easily convert this into 2D DF using the .unstack() method. Alternatively, we can go from the original ri_weather DF to the unstacked 2D DF.
print(arrest_rate) # To see what we're dealing with

# unstacked to 2D DF
print(arrest_rate.unstack()) 

# pivot_table
print(pd.pivot_table(data = ri_weather, 
                     index = 'violation', 
                     columns = 'rating', 
                     values = 'is_arrested', 
                     aggfunc = 'mean'))















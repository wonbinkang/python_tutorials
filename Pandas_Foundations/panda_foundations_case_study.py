# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:13:51 2019

@author: Wonbin Kang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Read in the NOAA file as df. Note that there is no header.
df = pd.read_csv("NOAA_QCLCD_2011_hourly_13904.txt", header = None)
df.head()

# comma separated string of column names:
column_labs = 'Wban,date,Time,StationType,sky_condition,sky_conditionFlag,visibility,visibilityFlag,wx_and_obst_to_vision,wx_and_obst_to_visionFlag,dry_bulb_faren,dry_bulb_farenFlag,dry_bulb_cel,dry_bulb_celFlag,wet_bulb_faren,wet_bulb_farenFlag,wet_bulb_cel,wet_bulb_celFlag,dew_point_faren,dew_point_farenFlag,dew_point_cel,dew_point_celFlag,relative_humidity,relative_humidityFlag,wind_speed,wind_speedFlag,wind_direction,wind_directionFlag,value_for_wind_character,value_for_wind_characterFlag,station_pressure,station_pressureFlag,pressure_tendency,pressure_tendencyFlag,presschange,presschangeFlag,sea_level_pressure,sea_level_pressureFlag,record_type,hourly_precip,hourly_precipFlag,altimeter,altimeterFlag,junk'
column_labs_list = list(column_labs.split(","))
df.columns = column_labs_list

# Remove columns in list_to_drop (copy pasted from DC)
list_to_drop = ['sky_conditionFlag',
 'visibilityFlag',
 'wx_and_obst_to_vision',
 'wx_and_obst_to_visionFlag',
 'dry_bulb_farenFlag',
 'dry_bulb_celFlag',
 'wet_bulb_farenFlag',
 'wet_bulb_celFlag',
 'dew_point_farenFlag',
 'dew_point_celFlag',
 'relative_humidityFlag',
 'wind_speedFlag',
 'wind_directionFlag',
 'value_for_wind_character',
 'value_for_wind_characterFlag',
 'station_pressureFlag',
 'pressure_tendencyFlag',
 'pressure_tendency',
 'presschange',
 'presschangeFlag',
 'sea_level_pressureFlag',
 'hourly_precip',
 'hourly_precipFlag',
 'altimeter',
 'record_type',
 'altimeterFlag',
 'junk']

df = df.drop(list_to_drop, axis = "columns")
df.head()
df.info()

## Create a datetime object using "date" and "Time", which are currently integers, and set to index. Moreover, the Time column needs to be padded with zeros in front to create a 4 digit string. 
# Convert "date" column to string.
df.date = df.date.astype("str")

# Pad with zeros to create a four digit string for all integers in the Time column.
# The following padding method returns a string whether you pass a string or integer:
lst = [1, 21, 338, 1613]
for i in lst:
    print("{0:0>4}".format(i))
# For more information on "{}".format(), see https://www.geeksforgeeks.org/python-format-function/. Note that {0:0>4}.format(i) means that i is placed in the first placeholder, indexed by the first 0, there are 4 spaces filled with 0s and i is right justified (>). 

for i in lst:
    print("{0:1<5}".format(i))
# {0:1<5}.format(i) means that i is placed in the first placeholder, indexed by the first 0, there are 5 spaces filled with 1s and i is left justified (<). 

df.Time = df.Time.apply(lambda i: "{0:0>4}".format(i)) 
df.info() # df.Time is an object column. The lambda spit out a string. 
df.Time.head()

datetime_string = df.date + df.Time
datetime_string.head()

datetime = pd.to_datetime(datetime_string, format = "%Y%m%d%H%M")
df.set_index(datetime, inplace = True)
df.info()
df.head()

# Note that the DF also has many columns that should be floats but are objects. Change these using a for loop. 
# Consider:
print(df.loc["2011-06-20 08:00":"2011-06-20 09:00", "dry_bulb_faren"])
# Note that missing values are encoded as "M", hence the entire col is object. 

# Create a list of cols to turn into numerics:
col_lst = ["visibility", "dry_bulb_faren", "dry_bulb_cel", "wet_bulb_faren", "wet_bulb_cel", "dew_point_faren", "dew_point_cel", "relative_humidity", "wind_speed", "wind_direction", "station_pressure", "sea_level_pressure"]
for col in col_lst:
    df[col] = pd.to_numeric(df[col], errors = "coerce")
    
df.info()
print(df.loc["2011-06-20 08:00":"2011-06-20 09:00", "dry_bulb_faren"])
# "M"s are now all NaN.

## Indexing DF using loc
# The following results are the same
print(df.dry_bulb_faren["2011-Aug"].median())
print(df.loc["2011-Aug", "dry_bulb_faren"].median())

# indexing:
print(df.loc["2011-Aug":"2011-Sept", "dry_bulb_faren"].median())

## Now we're going to use two datasets and compare:
df1 = df
df1.info()
df2 = pd.read_csv("weather_data_austin_2010.csv", header = 0, index_col = "Date", parse_dates = True)
df2.info()

# Our column of interest are df1.dry_bulb_faren and df2.Temperature. We will use numpy arrays to difference the two arrays:
# Resample df1 to daily means for all cols
daily_mean_2011 = df1.resample("D").mean()

# Select dry_bulb_faren and convert to array (using values):
daily_temp_2011 = daily_mean_2011.dry_bulb_faren.values
type(daily_temp_2011) #ndarray

# Resample df2 to daily means for all cols
daily_mean_2010 = df2.resample("D").mean()
type(daily_mean_2010)

# Select Temperature and convert to array (using reset_index):
daily_temp_2010 = daily_mean_2010.reset_index().Temperature
# reset_index will convert the datetime index to a integer range.
type(daily_temp_2010) # pandas Series
daily_temp_2010.index # RangeIndex( ... )

difference = daily_temp_2011 - daily_temp_2010
print(difference.mean())

# On average, how much hotter is it when the sun is shining? Use df1
df1.sky_condition.value_counts() # There are a lot of unique strings. 

# Clear skies index
is_sky_clear = df1.sky_condition == "CLR"
is_sky_clear.sum() # 2349 days out of 10337. 

# Filter the entire dataframe by whether sky_condition is CLR and down sample to daily aggregating by max
sunny = df1.loc[is_sky_clear]
sunny.info()
sunny_daily_max = sunny.resample("D").max()

# Overcast skies index
is_sky_overcast = df1.sky_condition.str.contains("OVC")
is_sky_overcast.sum() # 2758 days out of 10337

# Filter the entire DF by OVC and down sample to daily aggregate by max
overcast = df1.loc[is_sky_overcast]
overcast_daily_max = overcast.resample("D").max()

print(sunny_daily_max.mean() - overcast_daily_max.mean())
# Note dry_bulb_faren difference is 6.504304

## Is there a correlation between temperature and visibility? 
weekly_mean = df1.loc[:, ["visibility", "dry_bulb_faren"]].resample("W").mean()
print(weekly_mean.corr())
weekly_mean.plot(subplots = True)
plt.show()

## Use a box plot to visualize the fraction of days that are sunny.

is_sky_clear = df1["sky_condition"] == "CLR"
sunny_hours_per_day = is_sky_clear.resample("D").sum()
total_hours_per_day = is_sky_clear.resample("D").count()
sunny_fraction = sunny_hours_per_day / total_hours_per_day
sunny_fraction.plot(kind = "box")
plt.show()

## Explore the maximum temperature and dew point of each month. The columns of interest are 'dew_point_faren' and 'dry_bulb_faren'. 
monthly_max = df1.loc[:, ["dew_point_faren", "dry_bulb_faren"]].resample("M").max()
monthly_max.plot(kind = "hist", bins = 8, alpha = .5, subplots = True)
plt.show()

## Compare the maximum temperature in August 2011 against that of the August 2010 climate normals. More specifically, you will use a CDF plot to determine the probability of the 2011 daily maximum temperature in August being above the 2010 climate normal value. 

# Two methods to access max August 2010 temperature
august_max = df2.Temperature["2010-08"].max()
print(august_max)
august_max = df2.loc["2010-08", "Temperature"].max()
print(august_max)

# Resample Aug 2011 temps by day & aggregate to the max value
august_2011 = df1.loc["2011-08", "dry_bulb_faren"].resample("D").max()

# Filter for days in august_2011 where the value exceeds august_max: august_2011_high
august_2011_high = august_2011[august_2011 > august_max]

# Construct a CDF of august_2011_high
august_2011_high.plot(kind = "hist", normed = True, cumulative = True, bins = 25)
plt.show()

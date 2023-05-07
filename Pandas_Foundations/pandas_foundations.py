# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:04:26 2019

@author: Wonbin Kang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a DF using empty DF and populate with lists. 
exmp_df = pd.DataFrame()
exmp_df["total_pop"] = [3.034971e+09, 3.684823e+09, 4.436590e+09, 5.282716e+09, 6.115974e+09, 6.924283e+09]
exmp_df.index = [1960, 1970, 1980, 1990, 2000, 2010]
exmp_df.index.name = "Decade"
print(exmp_df)

np_vals = exmp_df.values
np_vals_log10 = np.log10(np_vals)
exmp_df_log10 = np.log10(exmp_df)

[print(x, "is of type", type(eval(x))) for x in ["np_vals", "np_vals_log10", "exmp_df", "exmp_df_log10"]]
# This shows that pandas use numpy functions and consist of numpy arrays. 

# Using dictionaries to create DF: recall that {key: [values]} dictionary leads to column names equal to key. 
# 1) zip lists of keys/colnames and values; 2) create list from zip object; 3) create dict from list of zipped objects; 4) pass to pd.DataFrame
list_keys = ["Country", "Total"]
list_values = [["United States", "Soviet Union", "United Kingdom"], [1118, 473, 273]]
lst_tuples = list(zip(list_keys, list_values))
print(lst_tuples)
data_dict = dict(lst_tuples)
data = pd.DataFrame(data_dict)
print(data)

data.columns = ["country", "total_medals"]
data.index = data["country"]
data = data.drop(labels = "country", axis = 1) # Drop the duplicate col. 
print(data)

# Broadcasting:
cities = ['Manheim', 'Preston park', 'Biglerville', 'Indiana', 'Curwensville', 'Crown', 'Harveys lake', 'Mineral springs', 'Cassville', 'Hannastown', 'Saltsburg', 'Tunkhannock', 'Pittsburgh', 'Lemasters', 'Great bend']
data_dict = {"state": "PA", "city": cities} # "PA" is only on string but will broadcast to all rows.
data = pd.DataFrame(data_dict)
print(data)

# Messy stock data using messy_stock_data.tsv. Open file and notice: 1) prologue; 2) tab delimited; 3) commented:
stock_df_messy = pd.read_csv("messy_stock_data.tsv")
print(stock_df_messy) # WRONG

stock_df_clean = pd.read_csv("messy_stock_data.tsv", delimiter = " ", skiprows = 3, comment = "#", )                             
print(stock_df_clean)
# stock_df_clean = pd.read_csv("messy_stock_data.tsv", delimiter = " ", header = 3, comment = "#", ) would also work. 

# Save the clean DF as a csv and xlsx to working directory:
stock_df_clean.to_csv("clean_stock_data.csv", index = False)
stock_df_clean.to_excel("clean_stock_data.xlsx", index = False)

## EDA with plots:
# USE the pandas DF method, or df.plot(): The pandas .plot() method makes calls to matplotlib to construct the plots. 
weather_df = pd.read_csv("weather_data_austin_2010.csv", index_col = "Date", parse_dates = True)
# Now weather_df is a time indexed DF

weath_df_sub = weather_df.iloc[0:744, 0:3]

# Simple plot:
weath_df_sub["Temperature"].plot(color = "blue")
plt.xlabel("Hours since January 1, 2010")
plt.ylabel("Temperature (F)")
plt.title("Temperature in Austin")

# Create a multi-column plot. Problem may arise with y axis scales.
# Simple version:
weath_df_sub.plot()
plt.show() # Notice the legend. However, Pressure seems to be constant.

# subplot version:
weath_df_sub.plot(subplots = True)
plt.show()

# plot Temperature and DewPoint together:
weath_df_sub[["Temperature", "DewPoint"]].plot()
plt.show()
# Note that with multiple columns, you need to have double brackets

weath_df_sub[["Temperature", "DewPoint"]].plot(color = ["blue", "red"], style = [".", ".-"])
plt.show()
# A very ugly plot, but use a list to determine color and style of plots

## Using the transposed version of the stock_df_clean, plot IBM and AAPL stock prices on y and month on x
stock_trnps = stock_df_clean.transpose() # colnames are integer indexed, extra row of "names", and row index is month
stock_trnps.columns = list(stock_trnps.iloc[0, :]) # set colnames to first row of transposed DF
stock_trnps = stock_trnps.drop("name", axis = 0) # drop the extra row of "names"
stock_trnps["Month"] = stock_trnps.index

stock_trnps.plot(x = "Month", y = ["APPLE", "GOOGLE"]) # Note that inside (), you don't need double brackets. 
plt.show()

# Scatterplot:
auto_df = pd.read_csv("auto-mpg.csv")
auto_df.columns

auto_df.plot(kind = "scatter", x = "hp", y = "mpg", alpha = .5) # Scatterplots have a "s" key argument for size of each dot. We can set this to a normalized value of the "weight" of the cars in DF.
plt.title('Fuel efficiency vs Horse-power')
plt.xlabel('Horse-power')
plt.ylabel('Fuel efficiency (mpg)')
plt.show()

# Boxplots with subplots: 2 versions
#1)
auto_df[["weight", "mpg"]].plot(kind = "box", subplots = True) # Don't forget: subplotS = . Argument is plural
plt.show()

#2)
auto_df.plot(kind = "box", y = ["weight", "mpg"], subplots = True)
plt.show()
# Notice the single and double brackets. 

# Histograms and CDFs
tip_df = pd.read_csv("tips.csv")
tip_df.head()

# Use "fraction" column to create pdf and cdf plots:
fig, axes = plt.subplots(nrows = 2, ncols = 1) # Create subplotS (again plural!!!) outside plot() because we're using the same column. The "fig" definition is that this is one figure. See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html 
tip_df.fraction.plot(kind = "hist", bins = 30, normed = True, range = (0, 0.3), ax = axes[0])
tip_df.fraction.plot(kind = "hist", bins = 30, normed = True, cumulative = True, range = (0, 0.3), ax = axes[1])
plt.show()
# Unlike DC, I have to call plt.show() at the very end. 

## Summary statistics and plots:
bach_deg = pd.read_csv("percent-bachelors-degrees-women-usa.csv")
bach_deg.head()
bach_deg.columns
bach_deg.index = bach_deg.Year
bach_deg = bach_deg.drop("Year", axis = 1)

# Calculate the mean percentage for each year (axis = 1): calculating ACROSS columns, but as rows
mean = bach_deg.mean(axis = 1)
mean.head()

# Since I fixed Year as index, we only need to call the plot
mean.plot(color = "blue")
plt.show()

# Mean, median and outliers using titanic fare data:
titanic = pd.read_csv("titanic.csv")
titanic.columns
titanic.info()

# Change the sex, survived as categories:
titanic[["survived", "sex"]] = titanic[["survived", "sex"]].astype("category")
titanic.info()

# Let's look at the "fare" and "age" columns:
titanic.fare.describe()
titanic.fare.plot(kind = "box")
plt.show()

# quantiles using titanic and boxplot side by side
print(titanic.fare.quantile([.05, .95]))
titanic[["age", "fare"]].plot(kind = "box")
plt.show() 
# This isn't a good plot because the yaxis are not the same for these two columns.

## Redo the above using gapminder life expct data:
gap = pd.read_csv("gapminder_life_expectancy_at_birth.csv")
gap.columns
gap.index #range(0, 260)

# Number of countries reporting in 2015
print(gap["2015"].count())

# 5 and 95 quantiles for all years in DF
print(gap.quantile([.05, .95]))

# Boxplot:
gap[["1800", "1850", "1900", "1950", "2000"]].plot(kind = "box")
plt.show()

## Use auto_df to work on Boolean indexing and filtering:
global_mean = auto_df.mean(axis =  0)
global_std = auto_df.std(axis = 0)

# Compare global mean and std with US cars:
auto_df.origin.unique()
auto_df.origin.value_counts()

bool_us = auto_df.origin == "US"; type(bool_us)
us = auto_df[bool_us]
us.info()
us_mean = us.mean(axis = 0)
us_std = us.std(axis = 0)

print(global_mean - us_mean); print(global_std - us_std)

# Use titanic to plot different sub populations of the dataset: View the distribution of "fare" grouped by "pclass"
titanic.info()
titanic.pclass.value_counts() # classes are 1, 2, 3

fig, axes = plt.subplots(nrows = 3, ncols = 1)
titanic[titanic.pclass == 1].plot(ax = axes[0], kind = "box", y = "fare")
titanic[titanic.pclass == 2].plot(ax = axes[1], kind = "box", y = "fare")
titanic[titanic.pclass == 3].plot(ax = axes[2], kind = "box", y = "fare")
plt.show()

## Time Series: datetime objects
# Creating a datetime indexed panda using weather_df
date_list = list(weather_df.index[0:743]) # Month's worth of data (2010 Jan)
temperature_list = list(weather_df.iloc[0:743, 0])

# Create a datetime list
my_datetimes = pd.to_datetime(date_list, format = "%Y-%m-%d %H:%M") # Note that format is the desired output format.
# The above code is unnecessary when I changed the code parsing datetime in the initial read_csv() call. 

# Create time Series
my_timeseries = pd.Series(temperature_list, index = my_datetimes)
my_timeseries.head(20)
my_timeseries.tail()

# Indexing and slicing using my_timeseries
ts1 = my_timeseries.loc["2010-01-25 17:00":"2010-01-25 18:00"]
ts2 = my_timeseries.loc["2010-01-26"]
ts3 = my_timeseries.loc["2010-01-25":"2010-01-26"]

# Reindexing .reindex(): Reindexing is useful in preparation for adding or otherwise combining two time series data sets. To reindex the data, we provide a new index and ask pandas to try and match the old data to the new index. If data is unavailable for one of the new index dates or times, you must tell pandas how to fill it in. Otherwise, pandas will fill with NaN by default.

lst_time_string = ["2016-07-01", "2016-07-02", "2016-07-03", "2016-07-04", "2016-07-05", "2016-07-06", "2016-07-07", "2016-07-08", "2016-07-09", "2016-07-10", "2016-07-11", "2016-07-12", "2016-07-13", "2016-07-14", "2016-07-15", "2016-07-16", "2016-07-17"]
lst_value_string = list(range(0,17))
ts4 = pd.Series(lst_value_string, index = lst_time_string)

lst_time_string = ["2016-07-01", "2016-07-04", "2016-07-05", "2016-07-06", "2016-07-07", "2016-07-08", "2016-07-11", "2016-07-12", "2016-07-13", "2016-07-14", "2016-07-15"]
lst_value_string = list(range(0,11))
ts5 = pd.Series(lst_value_string, index = lst_time_string)

# Let's say that we want to reindex the smaller ts5 using the index of the longer ts4.
ts6 = ts5.reindex(ts4.index)
print(ts6) # Note the creation of NaN's

ts7 = ts5.reindex(ts4.index, method = "ffill") # or "bfil"
print(ts7)

# Check out the following sums of time series
ts4 + ts5 # NaNs appear
ts4 + ts6 # NaNs appear
ts4 + ts7 # No NaNs

## Resampling (down and up-sampling: finer or coarser time stamps)
# For resampling, we use method chaining (%>% in R).
# use the weather_df which we'll index:
weath_dsample_6H = weather_df.loc[:, "Temperature"].resample("6H").mean()
weath_dsample_6H.head()

weath_dsample_1D = weather_df.loc[:, "Temperature"].resample("D").count()
weath_dsample_1D.head()

# Separating and resampling
august = weather_df.Temperature["2010-08"]
august_dhighs = august.resample("D").max()
feb = weather_df.Temperature["2010-02"]
feb_dlows = feb.resample("D").min()

# Note that with partial string indexing, we don't have to be exact:
feb2 = weather_df.Temperature["2010 Feb"]
feb2_dlows = feb2.resample("D").min()
(feb_dlows == feb2_dlows).all() # True

# Resampling and rolling means (a method to smooth out data)
# rolling means (or moving averages) require method chaining: e.g. .rolling().mean()
unsmoothed = weather_df.Temperature["2010-08-01":"2010-08-15"]; unsmoothed.head()
smoothed = unsmoothed.rolling(window = 24).mean()
dct = {"smoothed":smoothed, "unsmoothed":unsmoothed}
august = pd.DataFrame(dct); august.head() # Notice that the window = 24 and so first 24 entries of smooth are NaN

august.plot()
plt.plot()

## .rolling() in combination with .resample()
august = weather_df.Temperature["2010 August"] # Flexible partial string indexing
august.head()
august.tail()
august_dhighs = august.resample("D").max()
august_dhighs_smoothed = august.resample("D").max().rolling(window = 7).mean()
print(august_dhighs_smoothed)

# Plot unsmoothed and smoothed August daily highs
august = pd.DataFrame({"unsmoothed":august_dhighs, "smoothed":august_dhighs_smoothed})
august.plot()
plt.show()

## String filtering and summary statistics using method chaining: think stringr package in R
sw = pd.read_csv("austin_airport_departure_data_2015_july.csv", skiprows = 15, parse_dates = True, index_col = "Date (MM/DD/YYYY)") # There is a bunch of preamble
# Notice the white spaces in the column names:
sw.columns = sw.columns.str.strip() # list of string patterns can be included to remove. If no argument, then preceding and trailing white spaces are deleted. 
sw.head()
sw.info() # Note that there is an extra Carrier Code and this throws off the codes below.

# Create a logical vector and check that we don't have missing values
dallas = sw["Destination Airport"].str.contains("DAL") 
dallas.isnull().sum() # There is 1 NaN value.
dallas = dallas.dropna()
dallas.isnull().sum() # No more NaN value.
dallas.head()

# Total number of flights to Dallas in our dataframe:
dallas.sum()

# Total number of flights to Dallas each day:
daily_dep_DAL = dallas.resample("D").sum(); print(daily_dep_DAL)
# summary stats on daily departures:
daily_dep_DAL.describe() # avg 9.3225 flights; max 11 flights, min 3 flights

# Can we filter for DAL flights? 
# sw_DAL = sw[dallas.values] # We can't do this because of wrong length
sw.tail() # Because of "SOURCE: Bureau of Transportation ..." is included as a row. 

# Delete the final row and the column "Unnamed: 17":
sw = sw[:-1]
sw = sw.drop("Unnamed: 17", axis = 1)
sw.info()

# Filter dataset for flights to DAL
sw_DAL = sw[dallas]
sw_DAL.info()

## Missing Values and Interpolation:
# Reindex the smaller Series (ts5) using the index of the longer series (ts4) and interpolate the missing values:
ts5_interpolate1 = ts5.reindex(ts4.index, method = "ffill"); print(ts5_interpolate1)
ts5_interpolate2 = ts5.reindex(ts4.index).interpolate(how = "linear"); print(ts5_interpolate2)
# Which method I use will be determined by the data and the assumptions that I make. 

## Time zones and conversion
# We have to revert to a non datetime indexed version of sw dataframe:
sw_non_ind = pd.read_csv("austin_airport_departure_data_2015_july.csv", skiprows = 15)
sw_non_ind = sw_non_ind[:-1]
sw_non_ind = sw_non_ind.drop(["Unnamed: 17"], axis = 1)
sw_non_ind.columns = sw_non_ind.columns.str.strip()
sw_non_ind.info()

# Filter for flights to LAX
la_index = sw_non_ind["Destination Airport"].str.contains("LAX")
sw_non_ind_LA = sw_non_ind[la_index]

# Combine to columns in sw_non_ind_LA to create a date & time string which is converted to a datetime object.
times_no_tz = pd.to_datetime(sw_non_ind_LA["Date (MM/DD/YYYY)"] + " " + sw_non_ind_LA["Wheels-off Time"])
times_no_tz.head()
times_central = times_no_tz.dt.tz_localize("US/Central")
times_central.head()
times_pacific = times_central.dt.tz_convert("US/Pacific")
times_pacific.head()

## Plotting using a new weath_df_sub dataset which hasn't been indexed by data and comparing to one that has been indexed by datetime:
weather_df = pd.read_csv("weather_data_austin_2010.csv")
weath_df_sub = weather_df.iloc[0:744, [0, 3]]
weath_df_sub.head()
weath_df_sub.tail()
weath_df_sub.info() # Note that "Date" is object. 

weath_df_sub.plot()
plt.show() # Notice that the x axis are the number indexes.

weath_df_sub.index = pd.to_datetime(weath_df_sub.Date, format = "%Y-%m-%d %H:%M")
weath_df_sub = weath_df_sub.drop("Date", axis = 1)
weath_df_sub.head()
weath_df_sub.tail()

# Another quicker code is to use the .set_index() call to which you pass the column and set inplace = "True"
weather_df = pd.read_csv("weather_data_austin_2010.csv")
weath_df_sub = weather_df.iloc[0:744, [0, 3]]
weath_df_sub.Date = pd.to_datetime(weath_df_sub.Date)
weath_df_sub.set_index("Date", inplace = True) # No need to delete the Date column used as an index. 
weath_df_sub.head()
weath_df_sub.tail()

# Create weather_df with Date as index
weather_df.Date = pd.to_datetime(weather_df.Date)
weather_df.set_index("Date", inplace = True)
weather_df.info()

# Plot summer data
weather_df.Temperature["2010-Jun":"2010-Aug"].plot()
plt.show()
plt.clf()

# Plot the one week data
weather_df.Temperature["2010-06-10":"2010-06-17"].plot()
plt.show()
plt.clf()
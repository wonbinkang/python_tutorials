# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 00:18:05 2019

@author: Wonbin Kang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Using list comprehension to create a dataframe list:
# dataframes = [pd.read_csv(filename) for filename in filenames]

## Reindexing DF using another DF's index
names_1981 = pd.read_csv("names1981.csv", header = None, names = ["name", "gender", "count"], index_col = ["name", "gender"])
names_1881 = pd.read_csv("names1881.csv", header = None, names = ["name", "gender", "count"], index_col = ["name", "gender"])
# Note that both DFs have a multiindex. 

# Reindex 1981 with 1881: note that 1881 has much fewer names:
common_names = names_1981.reindex(names_1881.index)
no_longer_names = common_names[common_names.isna()]
no_longer_names.head(20)
common_names = common_names.dropna()

## Arithmetic operators (like +, -, *, and /) broadcast scalar values to conforming DataFrames when combining scalars & DataFrames in arithmetic expressions. Broadcasting also works with pandas Series and NumPy arrays.
# Note that you gain more flexibility if you use the pandas arithmetic operators:
sp500 = pd.read_csv("sp500.csv", parse_dates = True, index_col = "Date")
exchange = pd.read_csv("exchange.csv", parse_dates = True, index_col = "Date")

# Using the exchange rates, convert Open and Close values to GBP:
sp500_to_gbp = sp500[["Open", "Close"]].multiply(exchange["GBP/USD"], axis = "rows")
print(sp500_to_gbp.head())

# Or the .pct_change() method
gdp_us = pd.read_csv("gdp_usa.csv", parse_dates = True, index_col = "DATE")

# Use data post 2008:
post2008 = gdp_us.loc["2008-01": , :]
pst2008_yr = post2008.resample("A").last()
pst2008_yr["annual_%_growth"] = pst2008_yr.pct_change() * 100; print(pst2008_yr)

## Appending and concatenating data: recall that the indices are retained post append/concatenate until reindexed.
# Note also that .append() and pd.concatenate() achieve the same result, while the latter is more flexible. 
# .append() with reindexing
names_1881 = pd.read_csv("names1881.csv", names = ["name", "sex", "count"])
names_1981 = pd.read_csv("names1981.csv", names = ["name", "sex", "count"])
# Both DFs have range indices. 

combined_names = names_1881.append(names_1981, ignore_index = True) # new index from 0 to total rows (n) of both name DFs. 
print(combined_names[combined_names.name == "Morgan"])

## Concatenating columns (axis = "columns"): Key is understanding where the NaN values will be placed in new DF. 
# Using a for loop to read in numerous files:
# First create an empty list which will be populated by DFs:
medals = []

# Second create a list of strings of the names of the files:
filename_lst = ["bronze", "silver", "gold"]
for str_fragment in filename_lst:
    filename = "C://Users//wonbi//Documents//DataCamp_Python//Python_data_science_basics//Merging_DataFrames_with_pandas//Summer Olympic medals//%s_top5.csv" % str_fragment #%s is the placeholder for str_fragment
    columns = ["Country", str_fragment]
    medal_df = pd.read_csv(filename, header = 0, index_col = "Country", names = columns)
    medals.append(medal_df)

len(medals) # list of length 3

medals_df = pd.concat(medals, axis = "columns")
print(medals_df)

## The results of concatenation with indices or columns that are specific to each DF are straightforward: joins with NaN where appropriate. What happens when the DFs share indices and column names? We need to use multiindexing and multilevel column names using the "key" argument of pd.concat()
# Edit the for loop reading in the various medals:
medals = []
filename_lst = ["bronze", "silver", "gold"]
for str_fragment in filename_lst:
    filename = "C://Users//wonbi//Documents//DataCamp_Python//Python_data_science_basics//Merging_DataFrames_with_pandas//Summer Olympic medals//%s_top5.csv" % str_fragment #%s is the placeholder for str_fragment
    medal_df = pd.read_csv(filename, header = 0, index_col = "Country")
    medals.append(medal_df)
# Note that each DF does not have a column name:

# Concat by "rows" and NOT "columns"
medals_df = pd.concat(medals, axis = "rows", keys = filename_lst) # Careful of the sequence. 
print(medals_df) # Notice the multiindex

# Print the number of Bronze medals won by Germany
print(medals_df.loc[("bronze", "Germany")])

# Print data about silver medals
print(medals_df.loc["silver"])

# Print all the data on medals won by the United Kingdom
print(medals_df.loc[(slice(None), "United Kingdom"), : ]) # DC uses another method pd.IndexSlice()

## Concatenate horizontally to get multiindexed columns:
# Create a list of DFs
sales = []
for str_frag in ["Hardware", "Service", "Software"]:
    filename = "C://Users//wonbi//Documents//DataCamp_Python//Python_data_science_basics//Merging_DataFrames_with_pandas//Sales//feb-sales-%s.csv" % str_frag
    sales_subdf = pd.read_csv(filename, index_col = "Date", parse_dates = True, header = 0)
    sales.append(sales_subdf)
print(sales)

# Concatenate the list of DFs:
feb_sales = pd.concat(sales, axis = "columns", keys = ["Hardware", "Service", "Software"])
feb_sales.info()
# Extract slice called slice_2_8 from feb_sales (using .loc[] & idx) that comprises rows between Feb. 2, 2015 to Feb. 8, 2015 from columns under 'Company'.
slice_2_8 = feb_sales.loc["2015-02-02":"2015-02-08", (slice(None), "Company")] # I didn't use the pd.IndexSlice() function.
slice_2_8

## Concatenating from dictionary:
jan = pd.read_csv("C://Users//wonbi//Documents//DataCamp_Python//Python_data_science_basics//Merging_DataFrames_with_pandas//Sales//sales-jan-2015.csv", header = 0)
feb = pd.read_csv("C://Users//wonbi//Documents//DataCamp_Python//Python_data_science_basics//Merging_DataFrames_with_pandas//Sales//sales-feb-2015.csv", header = 0)
mar = pd.read_csv("C://Users//wonbi//Documents//DataCamp_Python//Python_data_science_basics//Merging_DataFrames_with_pandas//Sales//sales-mar-2015.csv", header = 0)

# Create a list of names and DF tuples:
lst = [("January", jan), ("February", feb), ("March", mar)]

# From the list of tuples, create a dictionary using a for loop:
dct = {}
for name, data in lst:
    dct[name] = data.groupby("Company").sum() # Note that the sum() only worked on the Units column.
dct["February"]

# Concatenate the dictionary by row:
sales = pd.concat(dct, axis = "rows")
sales.sort_index(inplace = True) # Notice the multiindex

# Print all the sale for Mediacore:
print(sales.loc[(slice(None), "Mediacore"), : ])

## Compare the historical 10-year GDP (Gross Domestic Product) growth in the US and in China. The data for the US starts in 1947 and is recorded quarterly; by contrast, the data for China starts in 1961 and is recorded annually.
# We have to make sure that the two datasets are of the same time series using resampling
gdp_china = pd.read_csv("gdp_china.csv", parse_dates = True, index_col = "Year") 

# China is annual actual gdp data, but let's reformat datetime index to last day of the year and use percent changes over 10 years and drop all NaNs. 
china_new = gdp_china.resample("A").last().pct_change(10).dropna()
china_new.head()
china_new.columns = ["China GDP % Change"]

# Do the same for US and concatenate the two DFs horizontally.
us_new = gdp_us.resample("A").last().pct_change(10).dropna()
us_new.head()
us_new.columns = ["US GDP % Change"]
gdp = pd.concat([china_new, us_new], axis = "columns", join = "inner")

# Print the decade means of 10 years worth of data:
print(gdp.resample("10A").last())

## Merge: stack or combine DFs, not along the index, but along a column or columns. merge() is an extension of concat(). If the "key", or merging on column", is not given then all rows that are common to both DFs are used. 

# Note that using the "left_on" and "right_on" argument is not always the most efficient. Providing multiple columns "on" which to merge may return a better result. 

# pd.merge() is a function and the most flexible of all the ways to merge DFs. .join() is a method. See DC slides for overview of which method is the best for merging DFs (Occam's razor)
revenue = pd.DataFrame({"city": ["Austin", "Denver", "Springfield", "Mendocino"], 
                        "branch_id": [10, 20, 30, 47],
                        "state": ["TX", "CO", "IL", "CA"],
                        "revenue": [100, 83, 4, 200]})
    
manager = pd.DataFrame({"branch": ["Austin", "Denver", "Mendocino", "Springfield"], 
                        "branch_id": [10, 20, 47, 31],
                        "state": ["TX", "CO", "CA", "MO"],
                        "manager": ["Charles", "Joel", "Brett", "Sally"]})
    
sales = pd.DataFrame({"city": ["Mendocino", "Denver", "Austin", "Springfield", "Springfield"], 
                      "state": ["CA", "CO", "TX", "MO", "IL"],
                      "units": [1, 4, 2, 5, 1]})

# Practice merging data:
r_s_right = pd.merge(revenue, sales, how = "right", on = ["city", "state"])
s_m_left = pd.merge(sales, manager, how = "left", left_on = ["city", "state"], right_on = ["branch", "state"])

# Print the results and check the differences:
print(pd.merge(s_m_left, r_s_right)) # inner, with all columns used as keys: Because of branch_id NaNs, all Springfield rows are deleted from the inner join
print(pd.merge(s_m_left, r_s_right, how = "outer")) # Again because of the branch_id NaNs, there are duplicates of both Springfield rows. 
print(pd.merge(s_m_left, r_s_right, how = "outer", on = ["city", "state"]))

## Ordered merges using pd.merge_ordered(): note that default is "outer" join, whereas pd.merge() is "inner".
hardware = pd.read_csv("C://Users//wonbi//Documents//DataCamp_Python//Python_data_science_basics//Merging_DataFrames_with_pandas//Sales//feb-sales-Hardware.csv", parse_dates = ["Date"])
software = pd.read_csv("C://Users//wonbi//Documents//DataCamp_Python//Python_data_science_basics//Merging_DataFrames_with_pandas//Sales//feb-sales-Software.csv", parse_dates = ["Date"])

print(pd.merge(hardware, software)) # Empty DF because merging on all columns with same name and the "Date" column has no entries that are the same. 

print(pd.merge(hardware, software, how = "outer").sort_values("Date")) # Note merging on all columns and using sort_values() to get a sorted merged DF.
print(pd.merge_ordered(hardware, software)) # Same result except the range indices are well ordered. 

print(pd.merge_ordered(hardware, software, on = ["Date", "Company"], suffixes = ["_hardware", "_software"]))

# Practice more ordered merges:
austin = pd.DataFrame({"date": ["2016-01-01", "2016-02-08", "2016-01-17"], 
                       "ratings": ["Cloudy", "Cloudy", "Sunny"]})
houston = pd.DataFrame({"date": ["2016-01-04", "2016-01-01", "2016-03-01"], 
                       "ratings": ["Rainy", "Cloudy", "Sunny"]})

print(pd.merge_ordered(austin, houston)) # By default, an outer merge on date and ratings. Lose location information
print(pd.merge_ordered(austin, houston, on = "date")) # Ordered by date
print(pd.merge_ordered(austin, houston, on = "date", fill_method = "ffill"))

## Merging using merge_asof() which aligns datetime indices:
auto = pd.read_csv("automobiles.csv", header = 0, parse_dates = ["yr"]); auto.head() # Yearly car models
oil = pd.read_csv("oil_price.csv", header = 0, parse_dates = ["Date"]); oil.head() # Monthly price data
# Note that "yr" in auto and "Date" in oil are datetimes. This is necessary for the use of merge_asof()

# We use merge_asof() where the the two on columns will be compared and only the rows for which the right_on column is less than the left_on column is added to the right_on key. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge_asof.html
merged = pd.merge_asof(left = auto, right = oil, left_on = "yr", right_on = "Date")
print(merged.resample("A", on = "Date")[["mpg", "Price"]].mean()) # resample annually on Date column (since we don't have a date index, we have to give a column); pick two columns and aggregate them using the mean().

# Correlation between mpg and Price:
print(merged.resample("A", on = "Date")[["mpg", "Price"]].mean().corr())


# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 00:58:14 2019

@author: Wonbin Kang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

election = pd.read_csv("pennsylvania2012_turnout.csv", index_col = "county")

## Indexing (only index methods I'm unfamiliar with will be included)
p_counties = election["Perry":"Potter"]
p_counties_reverse = election["Potter":"Perry":-1]; print(p_counties_reverse)

# .loc indexing works:
p_counties = election.loc["Perry":"Potter"]; print(p_counties)
p_counties_reverse = election.loc["Potter":"Perry":-1]; print(p_counties_reverse)

# any() and all(): refer to "non zeros"; can be coupled with isnull() and notnull()
titanic = pd.read_csv('C:\\Users\\wonbi\\Documents\\DataCamp_Python\\Python_data_science_basics\\Pandas_Foundations\\titanic.csv')
titanic_sub = titanic[["age", "cabin"]]
titanic_sub.info() # Multitude of null entries.

# Practice using the all() and any(); isnull() and notnull() calls:
log_non_null = titanic_sub["age"].notnull() # logical where if data is not null, then True
log_is_null = titanic_sub["age"].isnull() # converse where if data is null, then True
(log_non_null + log_is_null).all() # returns True. 
print(log_is_null.sum()) # 263 null entries in column

# Drop rows with NaN:
print(titanic_sub.dropna(how = "any").shape) # Will drop rows with any NoN in the row. Went from 1309 rows to 272 rows
print(titanic_sub.dropna(how = "all").shape) # Will drop rows where ALL values are NaN. 1069

# Back to titanic and now drop COLUMNS with NaNs: Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh = 1000, axis = "columns").info()) # axis = 0: across by column (or "row"); 1: across by row (or "column") 
# "cabin", "boat", and "home.dest" cols deleted. 

## Transforming DataFrame elements
# To transform the entire DF or subsets thereof of numerical data, one should use methods that do not require for loops
#1) pandas DF functions to change all elements at once or arithmetic operators to do element by element transforms;
#2) numpy array functions to change entire columns at a time;
# When performance is paramount, you should avoid using .apply() and .map() because those constructs perform Python for-loops over the data stored in a pandas Series or DataFrame. By using vectorized functions (functions that take vectors as inputs) instead, you can loop over the data at the same speed as compiled code (C, Fortran, etc.)! NumPy, SciPy and pandas come with a variety of vectorized functions (called Universal Functions or UFuncs in NumPy).

# See the following example:
from scipy.stats import zscore #Instead of using .apply(), the zscore UFunc will take a pandas Series as input and return a NumPy array. 
turnout_zscore = zscore(election.turnout)
print(type(turnout_zscore)) # Note that this is a numpy array
election["turnout_zscore"] = turnout_zscore
election.head(10)

#3) custom functions and lambda function coupled with the use df.apply()

# To transform string elements: use the .str attribute which is an accessor to many vectorized string transformations.

# For functions use .apply() which has a default of axis = 0 (or "rows"); The .map() method is used to transform values according to a Python dictionary look-up. 
redvblue = {"Obama":"blue", "Romney":"red"}
election_copy = election.copy()
election_copy["color"] = election.winner.map(redvblue)
print(election_copy.head())

## Indexes
sales = pd.read_csv("sales.csv", index_col = "month"); print(sales)

# Indexes are immutable, and so can be changede ONLY WHOLESALE
sales.index = [index.upper() for index in sales.index] # Note that since the index is already a string, we don't need str.
print(sales)

# Index and column names:
sales.index.name = "MONTHS"
sales.columns.name = "PRODUCTS"
print(sales)

# Build an index from scratch:
# First create the sales DF in DC:
sales.reset_index()
del sales["MONTHS"]
print(sales)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales.index = months
sales.index.name = "months"
print(sales)

## MultiIndex: tuple index
# reset the index
sales.reset_index(inplace = True)
del sales.columns.name
del sales["months"]

# Add new columns
sales["state"] = ["CA", "CA", "NY", "NY", "TX", "TX"]
sales["month"] = [1, 2, 1, 2, 1, 2]
print(sales)

# Set and sort a multiindex:
sales.set_index(["state", "month"], inplace = True)
sales.sort_index(inplace = True)
print(sales)

# Compare the two lines of code below:
print(sales.loc[["CA", "TX"]])
print(sales.loc["CA":"TX"])
# Note that these were the outermost index and so () to denote tuple was not needed.

# Use the multiindex:
# index names:
sales.index
sales.index.name # None
sales.index.names # FrozenList with the nameS

# extract NY, month1
print(sales.loc[("NY", 1), :]) # Note that we use both levels of tuples and so () needed.

# CA and TX, month2
print(sales.loc[(["CA", "TX"], 2), :])

# all states, month2: if you slice both the outer and the inner, then you need to use the slice() function.
print(sales.loc[(slice(None), 2), :])

# Return to the expanded unindexed DF and use "state" which is not a unique identifier as index
sales.reset_index(inplace = True)
sales.set_index(["state"], inplace = True)
sales.loc["NY"] # We used a single row index expecting a single row but got two instead.

## Rearranging and Reshaping DF
users = pd.read_csv("users.csv")
del users["Unnamed: 0"]

pd.pivot(data = users, index = "weekday", columns = "city", values = "visitors")
# Compare with values = None
pd.pivot(data = users, index = "weekday", columns = "city")
# You get hierarchical column labels (visitors or signups, city) tuple

# Note that with pivot() the resulting index, colname tuple must be unique, or else a ValueError caused by duplicate entries will arise. In this case, we can use pivot_table() with its aggfunc argument set to an aggregation function.
# Play around with pivot_table():
#1)
print(users.pivot_table(index = "weekday", columns = "city")) # hierarchical column labels

#2)
print(users.pivot_table(index = "weekday")) # aggfunc is "mean" by default. Note that "city" doesn't appear in output. Entries are means of the two values indexed by weekday and signups/visitors (because two cities)
print(users.pivot_table(index = "weekday", aggfunc = "count")) # Counts the number of columns in each cell created by indexing on "weekday". 
print(users.pivot_table(index = "weekday", aggfunc = len)) # Does the same as aggfunc = "count".

#3) 
signups_and_visitors = users.pivot_table(index = "weekday", aggfunc = sum, margins = True) # total visitors on Sun or total signups on Mon. margins = True gives totals in the margins
print(signups_and_visitors)
## Multiindexing and hierarchical columns are linked using stack() and unstack() methods
users_midx = users.set_index(["city", "weekday"])
users_midx.sort_index(inplace = True)
byweekday = users_midx.unstack(level = "weekday"); print(byweekday) # Since weekday is inner index, we could use level = 1
print(byweekday.stack(level = 1)) # Original users_midx is returned

bycity = users_midx.unstack(level = 0); print(bycity)
print(bycity.stack(level = "city")) # Note that this is not the original dataset users_midx.

# Use .swaplevel to return to originial users_midx:
newusers = bycity.stack(level = "city")
newusers = newusers.swaplevel(0,1)
newusers.sort_index(inplace = True)
print(newusers.equals(users_midx)) # True: Same as users_midx
# Note the .equals() method to compare DFs. .equals() is a DF attribute. 

## Melting
users_long = pd.melt(frame = users, id_vars = ["weekday", "city"], value_vars = ["visitors", "signups"], var_name = "type", value_name = "number")

# Obtaining key-value pairs with melt(): Sometimes, all you need is some key-value pairs, and the context does not matter. If said context is in the index, you can easily obtain the key-value pairs. 
kv_pairs = pd.melt(frame = users_midx, col_level = 0)
print(kv_pairs)
# Note that removing the col_level = 0 argument does not change the output. 

## Groupby: split; apply; combine
## 1) groupby and categories
# Count the number of columns in survived for each group grouped by plcass.
print(titanic.groupby("pclass")["survived"].count()) # Note that this isn't the number of people who survived in each class.
print(titanic.groupby(["embarked", "pclass"])["survived"].count())

# Note that if the row indexes of two DFs are the same, you can use a column in one DF to groupby() the other DF.
life_exp = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/production/course_1650/datasets/life_expectancy.csv", index_col = "Country")
life_exp.head()
regions = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/production/course_1650/datasets/regions.csv", index_col = "Country")
regions.head()

print(life_exp.groupby(regions["region"])["2010"].mean()) # Sub Saharan Africa had lowest life expectancy


## 2) groupby and aggregation 
# .agg(): The .agg() method can be used with a tuple or list of aggregations as input. When applying multiple aggregations on multiple columns, the aggregated DataFrame has a multi-level column index.
aggregated = titanic.groupby("pclass")["age", "fare"].agg(["max", "median"])
print(aggregated.loc[:, ("age", "max")])
print(aggregated.loc[:, ("fare", "median")])

# DC states: If you have a DataFrame with a multi-level row index, the individual levels can be used to perform the groupby. 

# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()

# Read the CSV file into a DataFrame with no index
gapminder = pd.read_csv("gapminder_tidy.csv")
aggregated1 = gapminder.groupby(["Year", "region"]).agg({"population":"sum", "child_mortality":"mean", "gdp":spread})


# Read the CSV file into a DataFrame and sort the multiindex: gapminder
gapminder = pd.read_csv("gapminder_tidy.csv", index_col = ['Year','region','Country'])
gapminder.sort_index(inplace = True)
gapminder.head() # multiindex

aggregated = gapminder.groupby(level = ["Year", "region"]).agg({"population":"sum", "child_mortality":"mean", "gdp":spread})
print(aggregated.tail(6))

aggregated1.equals(aggregated) # True. This shows that you can groupby on the row indexes as well as column entries. 

# Grouping on a function of the index
sales1 = pd.read_csv("sales-feb-2015.csv", index_col = "Date", parse_dates = True)
sales1.head()

# If we want to groupby on the index, we can simply use groupby(level = ). However, what if we want to manipulate the index and groupby on the output? 
print(sales1.groupby(sales1.index.strftime("%A"))["Units"].sum())

# .strftime()
from datetime import datetime
now = datetime.now()
Y = now.strftime("%Y")
print("year: ", Y)

print(now.strftime("%a"))
print(now.strftime("%m"))

## 3) groupby and transformation: apply .transform() method after grouping to apply a function to groups of data independently.
from scipy.stats import zscore
gapminder = pd.read_csv("gapminder_tidy.csv")
gapminder_2010 = gapminder[gapminder["Year"] == 2010]
gapminder_2010.info()

# Detect outliers in terms of "life" and "fertility" within a region by transforming each element to a regional zscore.
standardized = gapminder_2010.groupby("region")[["life", "fertility"]].transform(zscore)
outlier = (standardized["life"] < -3) | (standardized["fertility"] > 3)
outlier.sum() # 4 outliers

# Use the Boolean to index gapminder_2010
print(gapminder_2010[outlier])

# Imputation by group
# Fill in missing 'age' values for passengers on the Titanic with the median age from their 'gender' and 'pclass'. 
titanic.info() # "age" has over 250 missing values
print(titanic.age.tail(10)) # 3 in the final 10 age rows.

# Define a function that takes a vector of data and fills NAs with the median of the vector:
def impute_med(series):
    return series.fillna(series.median())

titanic.age = titanic.groupby(["sex", "pclass"])["age"].transform(impute_med)
print(titanic.age.tail(10))

## 4) groupby and agg and transform .apply()
# .apply() the following function:
def disparity(group):
    # Compute the spread of group['gdp']: s
    s = group['gdp'].max() - group['gdp'].min()
    
    # Compute the z-score of group['gdp']: z
    z = (group['gdp'] - group['gdp'].mean())/group['gdp'].std()
    
    # Return a DataFrame
    return pd.DataFrame({'zscore_gdp':z , 'regional spread(gdp)':s})

# The result of this split-apply-combine workflow is a new dataframe (transformed and aggregated):
gapminder_2010_indx = gapminder_2010.copy()
gapminder_2010_indx.set_index(["Country"], inplace = True)
regional_disparity = gapminder_2010_indx.groupby(["region"]).apply(disparity)
regional_disparity.head()
print(regional_disparity.loc[["United States", "United Kingdom", "China"], :])

## 5) groupby and filtering with grouping, aggregation and transformation
# Using the auto dataset, review the ins and outs of groupby()
auto = pd.read_csv("C://Users//wonbi//Documents//DataCamp_Python//Python_data_science_basics//Pandas_Foundations//auto-mpg.csv")
auto.info()
auto.groupby("yr")["mpg"].mean() # average of mpg by year. 

# What if we want to compare the average mpg of chevy cars and the rest within each year? 
grouping = auto.groupby("yr")
type(grouping) # pandas.core.groupby.generic.DataFrameGroupBy
type(grouping.groups) # dict THIS IS KEY!
print(grouping.groups.keys()) # The keys are the group_names.

# 1) groupby object: iteration
for group_name, group in grouping:
    avg = group["mpg"].mean()
    print(group_name, avg)

# 2) groupby object: iteration and filtering
for group_name, group in grouping:
    avg = group.loc[group["name"].str.contains("chevrolet"), "mpg"].mean()
    print(group_name, avg)

# 3) groupby object: dictionary comprehension and filtering
{group_name : group.loc[group["name"].str.contains("chevrolet"), "mpg"].mean() for group_name, group in grouping}

# 4) Boolean groupby
boo_chevy = auto["name"].str.contains("chevrolet")
auto.groupby(["yr", boo_chevy])["mpg"].mean()

## Groupby and filter with .apply()
# Group by "sex" and then apply the user defined function which calculates the survival rates for those in C deck:
def c_deck_survival(group):
    c_passengers = group["cabin"].str.startswith("C").fillna(False) # NA values are filled with False. 
    return group.loc[c_passengers, "survived"].mean() # "survived" encoded 0, 1 and so use mean to calculate proportion survived. 

print(titanic.groupby("sex").apply(c_deck_survival))

## Groupby and filter with .filter() 
sales1.groupby("Company")["Units"].sum() # Gives you the number of Units purchased by each company. 

# Select companies that bought more than 35 units:
sales1.groupby("Company").filter(lambda group: group["Units"].sum() > 35)
# Only rows for purchases by companies that bought mroe than 35 units (Mediacore and Streeplex). 

## Filtering and grouping with .map()
# In this exercise your job is to investigate survival rates of passengers on the Titanic by 'age' and 'pclass'. In particular, the goal is to find out what fraction of children under 10 survived in each 'pclass'. 
under10 = (titanic["age"] < 10).map({True:"under 10", False:"over 10"})
titanic.groupby(under10)["survived"].mean() # A little over 60 percent of the under 10 survived. 
titanic.groupby([under10, "pclass"])["survived"].mean() # Only 44% of the under 10 in third class survived. 



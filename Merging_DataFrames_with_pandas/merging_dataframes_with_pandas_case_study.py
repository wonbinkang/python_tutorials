# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:53:27 2019

@author: Wonbin Kang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "C://Users//wonbi//Documents//DataCamp_Python//Python_data_science_basics//Merging_DataFrames_with_pandas//Summer Olympic medals//"

## Load EDITIONS
editions = pd.read_csv(file_path + "Summer Olympic medalists 1896 to 2008 - EDITIONS.tsv", header = 0, sep = '\t')
editions = editions[["Edition", "Grand Total", "City", "Country"]]

## Load IOC Country Codes
ioc = pd.read_csv(file_path + 'Summer Olympic medalists 1896 to 2008 - IOC COUNTRY CODES.csv', header = 0)
ioc = ioc[['Country', 'NOC']]

## Load medalist data:
medals = pd.read_csv(file_path + 'Summer Olympic medalists 1896 to 2008 - ALL MEDALISTS.tsv', sep = '\t', skiprows = 4)
medals = medals[['Edition', 'Athlete', 'NOC', 'Medal']]
medals.head()
medals.tail()

# DC assumes that medals entries are separated into 26 different csv files: Create a dictionary and pd.concat() to create medals DF. Each csv file is: summer_1996.csv format. Use the editions from edition DF. Assume that all csv files are in working directory's Summer Olympic medals folder.
# medals_dict = {}
# for year in editions['Edition']:
    # filepath = file_path + "summer_{:d}.csv"
    # medals_dict[year] = pd.read_csv(filepath, header = 0)
    # medals_dict[year] = medals_dict[year][['Athlete', 'NOC', 'Medal']]
    # medals_dict[year]['Edition'] = year # Broadcast the "year" to each row of the dictionary's constituent DF
# medals = pd.concat(medals_dict, axis = "rows", ignore_index = True)

## Medal count for each country by edition:
medals_ed_cntry = pd.pivot_table(data = medals, index = 'Edition', 
                                 columns = 'NOC', 
                                 values = 'Athlete', # We could've used 'Athletes'
                                 aggfunc = 'count')
print(medals_ed_cntry.head())
print(medals_ed_cntry.tail())

## Computing fraction of medals per Olympics using pandas arithmetic methods
editions.set_index('Edition', inplace = True) # We set the index to 'Edition' to create medals_ed_cntry.
total_medals = editions["Grand Total"]
total_medals.head()
fractions = medals_ed_cntry.divide(total_medals, axis = 'rows')
fractions.head()
fractions.tail()

## Compute percentage change in fraction of medals won. 
mean_fractions = fractions.expanding().mean() # Calculates the mean() using that row's entry and all entries above for a column. Similar to rolling window over a datetime index. 
mean_fractions.head()

fractions_change = mean_fractions.pct_change() * 100

# Reset the index of fractions_change
fractions_change.reset_index(inplace = True)
fractions_change.head()
fractions_change.tail()

## Create the host DF by left joining with IOC
# First, remove the index from editions
editions.reset_index(inplace = True)
hosts = pd.merge(left = editions, right = ioc, on = 'Country', how = 'left') # If you use an inner join, you'll notice that the number of rows decreases, which means that there are some host countries that may not have a NOC code. 
hosts = hosts[['Edition', 'NOC']]
hosts.set_index('Edition', inplace = True)

# Deal with the NaN hosts
print(hosts[hosts.NOC.isnull()]) # 1972, 1980 and 1988 editions are NaN
hosts.loc[1972, 'NOC'] = 'FRG'
hosts.loc[1980, 'NOC'] = 'URS'
hosts.loc[1988, 'NOC'] = 'KOR'

# Reset the index of hosts
hosts.reset_index(inplace = True)

## Gather the fractions_change DF to tidy format:
reshaped = pd.melt(frame = fractions_change, 
                   id_vars = 'Edition', # Note that this doesn't mean that 'Edition' is an index!
                   var_name = 'NOC', 
                   value_name = 'Change')
# Extract rows from reshaped where 'NOC' == 'CHN'
chn = reshaped[reshaped['NOC'] == 'CHN']; print(chn.tail())

## INNER JOIN reshaped and hosts without the on argument. Therefore, only host country editions with coupled Change will be left after the merge:
merged = pd.merge(left = reshaped, right = hosts) # Default "inner" and all possible columns.

# Set the index to Edition and sort. Call this new DF influence
merged.set_index('Edition', inplace = True)
influence = merged.sort_index()

## Use influence to plot:
ax = influence.Change.plot(kind = "bar")
# Customizations
ax.set_ylabel("% Change of Host Country Medal Count")
ax.set_title("Is there a Host Country Advantage?")
ax.set_xticklabels(editions['City'])
plt.show()

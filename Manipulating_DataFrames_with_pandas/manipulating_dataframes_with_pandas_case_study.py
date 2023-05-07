# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:23:12 2019

@author: Wonbin Kang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
medals = pd.read_csv("all_medalists.csv", header = 0)
medals.columns
medals.head()
medals.info()
medals.Gender.value_counts()
medals.Event_gender.value_counts()
medals.Medal.value_counts()
medals[["Gender", "Event_gender", "Medal"]] = medals[["Gender", "Event_gender", "Medal"]].astype("category", inplace = True)

# To create ORDERED categoricals use the following code:
medals.Medal = pd.Categorical(values = medals.Medal, ordered = True, categories = ["Bronze", "Silver", "Gold"])

# Total number of medals won by the United States in each edition:
usa_medals_by_edition = medals[medals["NOC"] == "USA"].groupby("Edition")["Medal"].count(); print(usa_medals_by_edition)

# Using the medals dataset, print the top 15 countries in terms of medal counts:
print(medals["NOC"].value_counts().head(15))

# Create a table with NOC as index, Athletes as value and Medals as the head of columns:
counted = medals.pivot_table(index = "NOC", columns = "Medal", aggfunc = "count")
counted = medals.pivot_table(index = "NOC", columns = "Medal", values = "Athlete", aggfunc = "count")
# If value is not specified, you create a pivot table with hierarchical column labels for each of the possible values. 

counted["Totals"] = counted.sum(axis = 1) # axis = 0 / "index": across the rows; axis = 1 / "columns" is across the columns.
counted = counted.sort_values("Totals", ascending = False)
counted.head(20)

## What's the difference between "Event_gender" and "Gender" columns. Look at the unique permutations to glean some insight:
ev_gen_unique = medals[["Event_gender", "Gender"]].drop_duplicates(); print(ev_gen_unique)
# Notice Event_gender W and Gender Men and Event_gender X and both Gender Men and Women. 

# groupby both Event_gender and Gender and check the number of rows:
print(medals.groupby(["Event_gender", "Gender"]).count()) # Notice that there is only one row with Event_gender W and Gender Men. This is probably an error.

# Isolate the suspect row:
print(medals[(medals.Event_gender == "W") & (medals.Gender == "Men")])

## Which countries won medals in the most distinct sports? .nunique()
medals.groupby("NOC")["Sport"].nunique().sort_values(ascending = False)

## Cold War Olympics
# Create a Cold War and US, USSR Boolean to index the dataframe:
cold_war = (medals.Edition >= 1952) & (medals.Edition <= 1988)
us_ussr = (medals.NOC == "USA") | (medals.NOC == "URS") # Note that this Boolean could be created as:
us_ussr = medals.NOC.isin(["USA", "URS"])
cold_war_medals = medals[cold_war & us_ussr]
cold_war_medals.groupby("NOC")["Sport"].nunique().sort_values(ascending = False)

## USA vs USSR medal count during Cold War
# Create a pivot table with the edition as index, countries on the columns and values equal to the number of medals won by that country at that particular edition:
medals_by_cntry = pd.pivot_table(data = medals, index = "Edition", columns = "NOC", values = "Athlete", aggfunc = "count")
medals_by_cntry.info()
cold_war_us_ussr = medals_by_cntry.loc[1952:1988, ["USA", "URS"]]
most_medals = cold_war_us_ussr.idxmax(axis = 1) # .idxmax(); .idxmin(): across columns or rows which index/column label has the max or min value
most_medals.value_counts() # USSR won 8 times and USA 2 times. 

## Visualization:
## USA medal counts by edition:
usa = medals[medals["NOC"] == "USA"]
usa_medals_by_yr = usa.groupby(["Edition", "Medal"])["Athlete"].count().unstack() # Moves the inner row index to columns. 
usa_medals_by_yr.plot()
plt.show()
# The above created a line plot. Now create an area plot:
usa_medals_by_yr.plot(kind = "area")
plt.show()

## Visualizing USA medal counts by edition: area plot with ordered medals (currently the medals are in alphabetical order)
# I changed the code above so that Medal is now an ordered categorical. Now rerun the area plot above starting from usa = medals[medals.NOC == "USA"] 


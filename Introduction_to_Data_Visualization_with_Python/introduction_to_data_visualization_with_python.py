# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:14:59 2019

@author: Wonbin Kang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Getting started with a simple plot using numpy arrays as inputs:
bach = pd.read_csv('percent-bachelors-degrees-women-usa.csv')
bach.columns
bach.info()

# Single plot using pandas DF
bach.plot(x = 'Year', y = ['Physical Sciences', 'Computer Science'], color = ['red', 'blue'])
plt.show()

# DC using array:
year = np.array(bach.Year)
physical_sciences = np.array(bach['Physical Sciences'])
computer_science = np.array(bach['Computer Science'])

plt.plot(year, physical_sciences, color = 'red') # For arrays, DO NOT USE x = [], y = []. 
plt.plot(year, computer_science, color = 'blue')
plt.show()

## Create two plots side by side in row (not recommended method):
# Create plot axes for the first line plot
plt.axes([0.05, 0.05, 0.425, 0.9]) # [xlo, ylo, width, height] xlo = start of x axis, ylo = start of y axis on normalized scale between 0 and 1. This may take several attempts to get right, so not recommended for use. 

plt.plot(year, physical_sciences, color='blue')
plt.axes([0.525, 0.05, 0.425, 0.9])
plt.plot(year, computer_science, color = 'red')
plt.show()

## plt.subplot() better alternative. 
plt.subplot(1, 2, 1) # 1 x 2 dimensional plot's 1st subplot
plt.plot(year, physical_sciences, color = 'blue')
plt.title('Physical Sciences')

plt.subplot(1, 2, 2)
plt.plot(year, computer_science, color = 'red')
plt.title('Computer Science')

plt.tight_layout()
plt.show()

## Changing the x and y axis limits using previous plot:
plt.plot(year, physical_sciences, color = 'red') # For arrays, DO NOT USE x = [], y = []. 
plt.plot(year, computer_science, color = 'blue')
plt.title('Degrees awarded to women (1990-2010)\nComputer Science (red)\nPhysical Sciences (blue)')
plt.xlim([1990, 2010]) # You can use () tuples as well as input
plt.ylim([0, 50]) 
# plt.axis([1990, 2010, 0, 50]): Same result. NOTE plt.axis() and NOT plt.axes()
plt.show()

## Legends, annotations and plot styles
# Legends:
plt.plot(year, physical_sciences, color = 'red', label = 'Physical Sciences') # For arrays, DO NOT USE x = [], y = []. 
plt.plot(year, computer_science, color = 'blue', label = 'Computer Science')
plt.legend(loc = 'lower center')
plt.title('Degrees awarded to women')
plt.xlabel('Year')
plt.ylabel('% of Bachelor Degrees Awarded to Women')
plt.show()

# Annotate: xy (coordinate to annotate), xytext (coordinate of text, usually used with arrows), arrowprop (dictionary)
# Using the figure from directly above, add a arrow to the maximum computer science bachelor degree percent. 
cs_max = computer_science.max()
year_cs_max = year[computer_science.argmax()] # Provides the index number of computer science max which we can pass to the year array. 
plt.plot(year, physical_sciences, color = 'red', label = 'Physical Sciences') # For arrays, DO NOT USE x = [], y = []. 
plt.plot(year, computer_science, color = 'blue', label = 'Computer Science')

plt.legend(loc = 'lower center')
plt.annotate(s = 'maximum', xy = [year_cs_max, cs_max], xytext = [year_cs_max + 5, cs_max + 5], arrowprops = {'facecolor':'black'}) # Don't forget that it's arrowpropS (PLURAL)
plt.title('Degrees awarded to women')
plt.xlabel('Year')
plt.ylabel('% of Bachelor Degrees Awarded to Women')
plt.show()

# Style sheets (themes in ggplot)
print(plt.style.available)
plt.style.use('ggplot')

# 2 x 1 subplot with annotation for both max (no need for label)
plt.subplot(2, 1, 1)
plt.plot(year, physical_sciences, color = 'red')
plt.title('Percent of Physical Sciences Bachelors Degrees Awarded to Women')
plt.annotate(s = 'maximum', 
             xy = [year[physical_sciences.argmax()], physical_sciences.max()],
             xytext = [year[physical_sciences.argmax()], physical_sciences.max() - 5],
             arrowprops = {'facecolor':'black'})

plt.subplot(2, 1, 2)
plt.plot(year, computer_science, color = 'blue')
plt.title('Percent of Computer Science Bachelors Degrees Awarded to Women')
plt.annotate(s = 'maximum', 
             xy = [year[computer_science.argmax()], computer_science.max()],
             xytext = [year[computer_science.argmax()], computer_science.max() - 5],
             arrowprops = {'facecolor':'black'})

plt.tight_layout()
plt.show()

## Plotting 2D arrays (raster images) or bivariate functions:
u = np.linspace(-2, 2, 41)
v = np.linspace(-1, 1, 21)
x,y = np.meshgrid(u, v) # expandgrid in R
z = np.sin(3 * np.sqrt(x**2 + y**2))
print('z:\n', z)

plt.pcolor(z)
plt.show()

# Aesthetic concerns with plot above:
#1) labels are of the indexes of the x and y arrays.
plt.pcolor(x, y, z, cmap = 'autumn') # Note that default cmap = 'jet'
#2) No legend for colors:
plt.colorbar()
#3) white space margins:
plt.axis('tight')
plt.show()

# Orientation of pcolor() and numpy array
A1 = np.array([[1, 2, 1], [0, 0, 1], [-1, 1, 1]])
A2 = np.array([[1, 0, -1], [2, 0, 1], [1, 1, 1]])
A3 = np.array([[1, 1, 1], [2, 0, 1], [1, 0, -1]])
plt.pcolor(A1, cmap = 'Blues')
plt.colorbar()
plt.show() # Notice that [1, 2, 1] is colored from the BOTTOM LEFT TO BOTTOM RIGHT, while [-1, 1, 1] colors TOP LEFT TO TOP RIGHT. 
plt.clf()

plt.pcolor(A2, cmap = 'Reds')
plt.colorbar()
plt.show()
plt.clf()

plt.pcolor(A3, cmap = 'Blues')
plt.colorbar()
plt.show()
plt.clf()

## Plotting bivariate functions using plt.contour() and plt.contourf()
plt.subplot(2, 1, 1)
plt.contour(x, y, z, 30, cmap = 'ocean')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.contourf(x, y, z, 30, cmap = 'twilight')
plt.colorbar()
plt.show()

## Plotting 2d histograms and hexagrams
auto = pd.read_csv('auto_mpg.csv')
auto.info()
mpg = np.array(auto.mpg)
hp = np.array(auto.hp)

# 2D histogram
plt.hist2d(x = hp, y = mpg, bins = [20, 20], range = [[40, 235], [8, 48]])
plt.colorbar()
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()

# hexbin()
plt.hexbin(x = hp, y = mpg, gridsize = [15, 12], extent = [40, 235, 8, 48]) # Note that extent = is not nested as hist2d. 
plt.colorbar()
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')
plt.show()

## Working with images
# Loading and examining images:
img = plt.imread('astronaut_during_EVA_in_1984.jpg')
print(img.shape) # Notice the 3072 x 3072 x 3 dimensions. The final array is for colors (RGB). 

plt.imshow(img)
plt.axis('off')
plt.show()

# Create pcolor plot ('gray') from img
img_intensity = img.sum(axis = 2) # sum along the color channel axis
print(img_intensity.shape) # no 3rd array

plt.imshow(img_intensity, cmap = 'gray')
plt.colorbar() # To show the intensity of each pixel
plt.axis('off')
plt.show()

# Extent and aspect:
# When using plt.imshow() to display an array, the default behavior is to keep pixels SQUARE so that the height to width ratio of the output matches the ratio determined by the shape of the array (m x n). In addition, by default, the x- and y-axes are labeled by the number of samples in each direction.

# The ratio of the displayed width to height is known as the image aspect and the range used to label the x- and y-axes is known as the image extent. The default aspect value of 'auto' keeps the pixels square and the extents are automatically computed from the shape of the array if not specified otherwise.

# See below to see aspect and extent in action:
plt.subplot(2, 2, 1)
plt.title('extent = (-1, 1, -1, 1), \naspect = 0.5') # NOTE extent is (width tuple and height tuple) which is reverse of row column sequence. 
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.imshow(img, extent = (-1, 1, -1, 1), aspect = 0.5)

plt.subplot(2, 2, 2)
plt.title('extent = (-1, 1, -1, 1), \naspect = 1')
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.imshow(img, extent = (-1, 1, -1, 1), aspect = 1)

plt.subplot(2, 2, 3)
plt.title('extent = (-1, 1, -1, 1), \naspect = 2')
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.imshow(img, extent = (-1, 1, -1, 1), aspect = 2)

plt.subplot(2, 2, 4)
plt.title('extent = (-2, 2, -1, 1), \naspect = 2') # The aspect increases the height of image, but increasing the extent compensates and so image is the same as in plot 2 above. 
plt.xticks([-2, -1, 0, 1, 2])
plt.yticks([-1, 0, 1])
plt.imshow(img, extent = (-2, 2, -1, 1), aspect = 2)

plt.tight_layout()
plt.show()

# Rescaling pixel intensities
image = plt.imread('Unequalized_Hawkes_Bay_NZ.jpg')
image.shape # (683, 1024, 3)
image = image.mean(axis = 2)
image.shape # (683, 1024)

pmin, pmax = image.min(), image.max()
print('The smallest and largest pixel intensities are %d and %d, respectively.' % (pmin, pmax))

image_rescaled = 256 * (image - pmin)/(pmax - pmin) # image is an array and so element by element arithmetic is possible. 
print('The rescaled smallest and largest pixel intensities are %.1f and %.1f.' %(image_rescaled.min(), image_rescaled.max()))

plt.subplot(2, 1, 1)
plt.title('original image')
plt.axis('off')
plt.imshow(image)

plt.subplot(2, 1, 2)
plt.title('rescaled image')
plt.axis('off')
plt.imshow(image_rescaled)

plt.show() # Not clear that there is much of a difference.

## seaborn (ggplot within the tidyverse): works best with pandas
import seaborn as sns

## Simple regression:
sns.lmplot(x = 'weight', y = 'hp', data = auto)
plt.show()

## Simple residual plot:
sns.residplot(x = 'hp', y = 'mpg', data = auto, color = 'green') # Green dots.
plt.show()

## Higher order regressions (overfit??)
plt.scatter(x = auto.weight, y = auto.mpg, label = 'data', color = 'red', marker = 'o')
sns.regplot(x = 'weight', y = 'mpg', data = auto, order = 1, color = 'blue', label = 'order 1', scatter = None)
sns.regplot(x = 'weight', y = 'mpg', data = auto, label = 'order 2', color = 'green', scatter = None, order = 2)
plt.legend(loc = 'upper right')
plt.show()

## Grouped by categorical factor variable regression plot:
# By hue on one plot
sns.lmplot(x = 'weight', y = 'hp', data = auto, hue = 'origin', palette = 'Set1') # hue = is the color aesthetic in ggplot
plt.show()

# By row or column subplots
sns.lmplot(x = 'weight', y = 'hp', data = auto, row = 'origin') # col = to have plots left to right
plt.show()

## Visualizing Univariate data (aside from boxplots)
##Strip plot
plt.subplot(2, 1, 1)
sns.stripplot(x = 'cyl', y = 'hp', data = auto, jitter = False)
plt.subplot(2, 1, 2)
sns.stripplot(x = 'cyl', y = 'hp', data = auto, jitter = True, size = 3) # size = 5 is default
plt.tight_layout()
plt.show()

## Swarm plot: similar to jitter stripplot but doesn't allow for overlap in counts
plt.subplot(2, 1, 1)
sns.swarmplot(x = 'cyl', y = 'hp', data = auto)

plt.subplot(2, 1, 2)
sns.swarmplot(x = 'hp', y = 'cyl', hue = 'origin', data = auto, orient = 'h')

plt.show() # Note that grouping by hue = and changing orient = are available to strip plots as well. 

## Violin plot (for large set of a variable)
plt.subplot(2, 1, 1)
sns.violinplot(x = 'cyl', y = 'hp', data = auto)

## Overlaying strip plot on violin plot
plt.subplot(2, 1, 2)
sns.violinplot(x = 'cyl', y = 'hp', data = auto, inner = None, color = 'lightgray')
sns.stripplot(x = 'cyl', y = 'hp', data = auto, jitter = True, size = 1.5)

plt.show()

## Visualizing multivariate data:
## Joint plots
sns.jointplot(x = 'hp', y = 'mpg', data = auto)
plt.show()

# joint plot with hexbin used to show joint distribution
sns.jointplot(x = 'hp', y = 'mpg', data = auto, kind = 'hex')
plt.show()
# kind='scatter' uses a scatter plot of the data points (default)
# kind='reg' uses a regression plot (default is order 1)
# kind='resid' uses a residual plot
# kind='kde' uses a kernel density estimate of the joint distribution, which smooths out distribution.
# kind='hex' uses a hexbin plot of the joint distribution

## Pair plots
auto.info() # Note that there are 8 non categorical variables
sns.pairplot(data = auto)
plt.show() # Shows an 8 x 8 grid of scatterplots with histograms on the diagonal.

# Plot regression lines on the off diagonals and group by the 'origin' column. For purposes of this exercise, use a sub DF of auto:
auto_sub = auto[['mpg', 'hp', 'weight', 'origin']] # Should be a 3x3 grid.
sns.pairplot(data = auto_sub, hue = 'origin', kind = 'reg') 
plt.show() # Instead of histograms, we get densities on the diagonals. 

## Heat map
# Plotting relationships between many variables using a pair plot can quickly get visually overwhelming, as we saw when we used all of the columns from auto. 
auto_corr = auto.corr() # Note that again, categorical variables were removed
sns.heatmap(data = auto_corr)
plt.show()

## Time series using pandas and datetime objects
stocks = pd.read_csv('stocks.csv', index_col = 'Date', parse_dates = True)
stocks.info()

aapl = stocks['AAPL']; type(aapl) # Series
ibm = stocks['IBM']
csco = stocks['CSCO']
msft = stocks['MSFT']

## Construct a plot showing four time series stocks on the same axes.
plt.plot(aapl, color = 'blue', label = 'AAPL')
plt.plot(ibm, color = 'green', label = 'IBM')
plt.plot(csco, color = 'red', label = 'CSCO')
plt.plot(msft, color = 'magenta', label = 'MSFT')
plt.legend(loc = 'upper left')
plt.xticks(rotation = 60)
plt.show()

## Slicing and plotting time series (1)
plt.subplot(2, 1, 1)
plt.plot(aapl, color = 'blue')
plt.title('AAPL: 2000 to 2013')
plt.xticks(rotation = 45)

aapl_sub = aapl['2007':'2008']

plt.subplot(2, 1, 2)
plt.plot(aapl_sub, color = 'magenta')
plt.title('AAPL: 2007 to 2008')
plt.xticks(rotation = 45)

plt.tight_layout()
plt.show()

## Slicing and plotting time series (2)
aapl_sub = aapl['2007-11':'2008-04']; type(aapl_sub)

plt.subplot(2, 1, 1)
plt.plot(aapl_sub, color = 'red')
plt.xticks(rotation = 45)
plt.title('AAPL Nov. 2007 to Apr. 2008')

aapl_sub = aapl['2008-01']
plt.subplot(2, 1, 2)
plt.plot(aapl_sub, color = 'green')
plt.xticks(rotation = 45)
plt.title('AAPL Jan. 2008')

plt.tight_layout()
plt.show()

## Use plt.axes() with the 4-tuple to argument [xlo, ylo, width, height] to create an inset plot:
# First plot the entire aapl Series:
plt.style.use('classic')# Note that the ggplot style does not lend itself to easy viewing given the background. 
plt.plot(aapl, color = 'blue')
plt.title('AAPL: 2000 to 2013')
plt.xticks(rotation = 45)

# Second plot the inset plot using plt.axes()
aapl_sub = aapl['2007-11':'2008-04']

plt.axes([0.25, 0.5, 0.35, 0.35])
plt.plot(aapl_sub, color = 'red')
plt.xticks(rotation = 45)
plt.title('Nov. 2007 to Apr. 2008')
plt.show()

## Rolling averages
mean_30 = aapl.rolling(30).mean()
mean_75 = aapl.rolling(75).mean()
mean_125 = aapl.rolling(125).mean()
mean_250 = aapl.rolling(250).mean()

# Rolling average overlayed with AAPL time series on distinct subplots
plt.subplot(2, 2, 1)
plt.plot(mean_30, color = 'green')
plt.plot(aapl, '-.k') #'-.k' is black dashdot line. 
plt.title('30d averages')
plt.xticks(rotation = 60)

plt.subplot(2, 2, 2)
plt.plot(mean_75, color = 'red')
plt.plot(aapl, '-.k')
plt.title('75d averages')
plt.xticks(rotation = 60)

plt.subplot(2, 2, 3)
plt.plot(mean_125, color = 'magenta')
plt.plot(aapl, '-.k')
plt.title('125d averages')
plt.xticks(rotation = 60)

plt.subplot(2, 2, 4)
plt.plot(mean_250, color = 'cyan')
plt.plot(aapl, '-.k')
plt.title('250d averages')
plt.xticks(rotation = 60)

plt.tight_layout()
plt.show()

# Rolling std() plotted on common axes
std_30 = aapl.rolling(30).std()
std_75 = aapl.rolling(75).std()
std_125 = aapl.rolling(125).std()
std_250 = aapl.rolling(250).std()

plt.plot(std_30, color = 'red', label = '30d')
plt.plot(std_75, color = 'cyan', label = '75d')
plt.plot(std_125, color = 'green', label = '125d')
plt.plot(std_250, color = 'magenta', label = '250d')
plt.legend(loc = 'upper left')
plt.title('Moving standard deviations')
plt.show()

## Histogram equalization to sharpen images
# Recall that an image is a two-dimensional array of numerical intensities. An image histogram, then, is computed by counting the occurences of distinct pixel intensities over all the pixels in the image.
# Use 'image' loaded above and create a subplot with the grayscaled image stacked on its image histogram:
plt.subplot(2, 1, 1)
plt.imshow(image, cmap = 'gray')
plt.title('Original Image')
plt.axis('off')

# Create a flattened image vector from the 2D array image: pixels
pixels = image.flatten()

# Plot the normalized histogram
plt.subplot(2, 1, 2)
plt.title('Original Image Normalized Histogram')
plt.xlim([0, 255])
plt.hist(pixels, bins = 64, range = [0, 256], normed = True, color = 'red', alpha = 0.4) # Note we use range 256, because of the bins (256/64 = 4).

plt.show()
# Note that the pixel intensities are bunched up between the 130 to 210 range. If we were able to spread these pixel intensities out, then we'd get a sharper contrasted picture. 

## Now create a 2 x 1 subplot where the lower image is the histogram and cdf transposed with their own axes on the left and right of the plot. 
plt.subplot(2, 1, 1)
plt.imshow(image, cmap = 'gray')
plt.axis('off')
plt.title('Original Image (grayscale)')

plt.subplot(2, 1, 2)
plt.hist(pixels, bins = 64, range = [0, 256], color = 'red', alpha = 0.4, density = True)
plt.grid(False) # Not sure what this is for. Note that 'off' (as DC notes) is deprecated. Use boolean. 

plt.twinx()

plt.hist(pixels, bins = 64, range = [0, 256], color = 'blue', alpha = 0.4, density = True, cumulative = True)
plt.grid(False) 
plt.title('Original Image PDF and CDF')

plt.xlim([0, 256])
plt.show()
plt.savefig('image_hist_cdf.png')
## Histogram equalization:
# Histogram equalization is an image processing procedure that reassigns image pixel intensities. The basic idea is to use interpolation to map the original CDF of pixel intensities to a CDF that is almost a straight line. In essence, the pixel intensities are spread out and this has the practical effect of making a sharper, contrast-enhanced image. 

# Using the flattened pixels, calculate the cdf, bins, and patches using plt.hist():
cdf, bins, patches = plt.hist(pixels, bins = 256, range = [0, 256], density = True, cumulative = True)
print(bins) # xp in np.interp()
print(cdf) # basis of fp in np.interp() and a method to normalize since the cdf lies between 0 and 1. 

# Plot bins[:-1] and cdf * 255 as a line graph
plt.plot(bins[:-1], cdf * 255)
plt.show()
# Note that the domain of pixels is between approximately 130 and 210. If we use the function created by bins[:-1] and cdf * 255, then the target of the domain is now between 0 and 255. Further interpolation assume linearity between the domain (bins[:-1]) and the target (255 * cdf). This is the desired result.

# We create new_pixels using np.interp():
new_pixels = np.interp(x = pixels, xp = bins[:-1], fp = cdf * 255)
new_pixels_sorted = np.sort(new_pixels)

plt.plot(new_pixels_sorted)
plt.show() # Notice that the plot is almost linear. 

# We've been working with a flattened image, so we need to reshape the array:
new_image = new_pixels.reshape(image.shape)

# Use the new_image above to create a subplot with the new image stacked on the pdf and cdf:
plt.subplot(2, 1, 1)
plt.imshow(new_image, cmap = 'gray')
plt.grid(False)
plt.title('Equalized Image')

plt.subplot(2, 1, 2)
plt.hist(new_pixels, bins = 64, range = [0, 256], density = True, color = 'red', alpha = 0.4)
plt.grid(False)

plt.twinx()

plt.hist(new_pixels, bins = 64, range = [0, 256], density = True, cumulative = True, color = 'blue', alpha = 0.4)
plt.grid(False)

plt.title('New Image CDF and PDF')
plt.xlim([0, 256])
plt.show()

plt.savefig('new_image_equalized_hist_cdf.png')

## Now let's try for a color image: NSFW jane image
# Don't open at work! :)
jane = plt.imread('NSFW_jane_.jpg') # To prevent accidental use, I've changed the image filename from that in the working directory. 
jane.shape # (2000, 3000, 3)

# Extract the 2d arrays for red, green, and blue and flatten:
red, green, blue = jane[:, :, 0], jane[:, :, 1], jane[:, :, 2]
red_pixels = red.flatten()
green_pixels = green.flatten()
blue_pixels = blue.flatten()

# Show the image on top and then show the histograms of the image superimposed on a single plot:
plt.subplot(2, 1, 1)
plt.imshow(jane)
plt.grid(False)
plt.title('Original Image')

plt.subplot(2, 1, 2)
plt.hist(red_pixels, bins = 64, range = [0, 256], density = True, color = 'red', alpha = .25)
plt.hist(green_pixels, bins = 64, range = [0, 256], density = True, color = 'green', alpha = .25)
plt.hist(blue_pixels, bins = 64, range = [0, 256], density = True, color = 'blue', alpha = .25)

plt.savefig('NSFW_jane_hist.png')
plt.show()

## You could use hist2d() to see the relative intensities for each color channel. Skip

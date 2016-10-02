import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import csv

##### Read company names into a dictionary
def readNamesIntoDict():
    d = dict()
    input_file = csv.DictReader(open("SP_500_firms.csv"))
    for row in input_file:
        #print(row)
        d[row['Symbol']] = [row['Name'],row['Sector']]
    return d

namesDict = readNamesIntoDict()

compNames = namesDict.keys()


##### Prices into standarad Python data structures

# Read prices into dictionary of lists

def readPricesIntoDict():
    input_file = csv.DictReader(open('SP_500_close_2015.csv', 'r')) 
    d = dict()
    for row in input_file:
        for column, value in row.items():
            d.setdefault(column, []).append(value)
    return d


prices = readPricesIntoDict()


##### Prices into pandas

# Open data with pandas 
filename = 'SP_500_close_2015.csv'
priceData = pd.read_csv(filename,index_col = 0)

print(type(priceData))
print(priceData.columns)

# Get specific data from dataframe

firstPrices =priceData.ix[0] # This is a "series" of first-day prices
print(type(firstPrices))

firstColumnPrices = priceData.ix[:,0] # First company by index 


applePrices = priceData['AAPL'] # Get by column name
msftPrices = priceData['MSFT']

# Create dataframe from series, then add another series
customPrices = applePrices.to_frame('AAPL')
customPrices = customPrices.join(msftPrices.to_frame('MSFT'))

print(customPrices)  

# Normalise data by first price
pricesScaled = priceData.divide(priceData.ix[0]) 
# Plot
priceFig = pricesScaled.plot(legend=False,figsize=(6,4))

# Save figure into working directory
plt.savefig('stocks2015.png', bbox_inches='tight')


# Loop through companies
for index,company in enumerate(priceData.columns):
        print(company,index)


# Turn into numpy matrix
priceMatrix = priceData.as_matrix()
# Into a 1D array
priceArray = priceMatrix.flatten() 

# Numpy is useful for eg math
np.sqrt(200)




########## Game On!

#Calculate daily return for each stock
dr = priceData.pct_change(1)

#omit the first line NA
dr.drop(dr.index[:1])

#max/min return, overall best/worst, most/least volatility
dr_summary = dr.describe().loc[['mean','std','min','max']]
dr_summary.T.idxmax()
#Netflix has the highest mean; Church & Dwight Co has the lowest dr, FCX has the highest dr

highest_dr = dr.idxmax().loc[['FCX']]
lowest_dr = dr.idxmin().loc[['CHD']]

dr_summary.T.idxmin()
#CHK has the lowest mean; Coca Cola has the least mean

#find the pairwise cor


from itertools import combinations
from scipy.stats.stats import pearsonr
def findCor(stockA,stockB,dr):
    dr_cor = dr.corr()
    correlation = dr_cor.at[stockA,stockB]
    print (namesDict[stockA][0]+ ' ' +namesDict[stockB][0]+' ')
    print(correlation)

findCor('AAPL','MSFT',dr)

def getCor(stock,dr):
    dr_cor = dr.corr()
    stock_list = dr_cor[stock]
    stock_list = stock_list.sort_values() #sort the correlation in asscending order
    low = stock_list.index[0]
    high = stock_list.index[-2]  #The last one is always the stock itself
    return namesDict[high][0], namesDict[low][0]

for stock in ['AMZN','MSFT','FB','AAPL','GOOG']:
    print(getCor(stock,dr))


#####Clustering
dr_cor = dr.corr()

from scipy.stats.stats import pearsonr
import itertools

dr_cor = dr_cor.where(np.triu(np.ones(dr_cor.shape)).astype(np.bool))
print (dr_cor)

dr_cor = dr_cor.stack().reset_index()
dr_cor.columns = ['Stock1','Stock2','Correlation']
print (dr_cor)

#create the list of edge(correlation,stock1,stock2)
edge = []
for i in range(len(dr_cor)):
    if dr_cor['Stock1'][i] != dr_cor['Stock2'][i]:
        edge.append((dr_cor['Correlation'][i], dr_cor['Stock1'][i], dr_cor['Stock2'][i]))

#sort the edge according to the correlation
sorted_edge = sorted(edge,key=lambda tup: tup[0])

#initialize the dict where each node points to itself
nodePointers = {}
stock1 = dr.columns
stock2 = stock1
for i in range(len(stock1)):
    nodePointers[stock1[i]] = stock2[i]




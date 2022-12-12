import math
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.cluster.vq import kmeans,vq,whiten



# Read the data from the csv file
countriesCounter = 0
varietiesCounter = 0

corrCountriesInt = 0
corrVarietiesInt = 0


dfCountries = pd.read_csv('winemag-data_first150k.csv', usecols=['country', 'points', 'price'])
dfVarieties = pd.read_csv('winemag-data_first150k.csv', usecols=['variety', 'points', 'price'])

#Store one fourth of the data in a new dataframe
dfCountriesSplit = dfCountries.sample(frac=0.25)
dfVarietiesSplit = dfVarieties.sample(frac=0.25)

#Take average of points and price for each country
dfCountriesSplit = dfCountriesSplit.groupby('country').mean()
dfVarietiesSplit = dfVarietiesSplit.groupby('variety').mean()

    

#Drop all rows with NaN values
dfCountriesSplitFinal = dfCountriesSplit.dropna() 
dfVarietiesSplitFinal = dfVarietiesSplit.dropna()


#Convert dataframe to array
dfCountriesSplitFinal = dfCountriesSplitFinal.astype(float)
dfVarietiesSplitFinal = dfVarietiesSplitFinal.astype(float)

#Normalize the data
varietiesFrame = whiten(dfVarietiesSplitFinal)
countriesFrame = whiten(dfCountriesSplitFinal)


#Remove outliers
dfCountriesSplitFinalNoOutliers = varietiesFrame[(np.abs(stats.zscore(varietiesFrame)) < 3).all(axis=1)]
dfVarietiesSplitFinalNoOutliers = countriesFrame[(np.abs(stats.zscore(countriesFrame)) < 3).all(axis=1)]


#Perform k-means clustering
centroidsCountries, _ = kmeans(dfCountriesSplitFinalNoOutliers, 3) #3 clusters
centroidsVarieties, _ = kmeans(dfVarietiesSplitFinalNoOutliers, 3)


#Assign each sample to a cluster
idxCountries, _ = vq(dfCountriesSplitFinal, centroidsCountries)
idxVarieties, _ = vq(dfVarietiesSplitFinal, centroidsVarieties)


#Store all points from each cluster
cluster1Countries = dfCountriesSplitFinal[idxCountries==0]
cluster2Countries = dfCountriesSplitFinal[idxCountries==1]
cluster3Countries = dfCountriesSplitFinal[idxCountries==2]

cluster1Varieties = dfVarietiesSplitFinal[idxVarieties==0]
cluster2Varieties = dfVarietiesSplitFinal[idxVarieties==1]
cluster3Varieties = dfVarietiesSplitFinal[idxVarieties==2]


#Show the correlation between points and price for each cluster

corrCountries1 = cluster1Countries.corr()
corrCountries2 = cluster2Countries.corr()
corrCountries3 = cluster3Countries.corr()

corrVarieties1 = cluster1Varieties.corr()
corrVarieties2 = cluster2Varieties.corr()
corrVarieties3 = cluster3Varieties.corr()


 
#Store first value in corrCountries that is not 1
for i in range(0, len(corrCountries1)):
    for j in range(0, len(corrCountries1)):
        if corrCountries1.iloc[i][j] != 1 and math.isnan(corrCountries1.iloc[i][j]) == False:
            corrCountriesHold1 = corrCountries1.iloc[i][j]
            if corrCountriesHold1 > corrCountriesInt:
                corrCountriesInt = corrCountriesHold1
            break
for i in range(0, len(corrCountries2)):
    for j in range(0, len(corrCountries2)):
        if corrCountries2.iloc[i][j] != 1 and math.isnan(corrCountries2.iloc[i][j]) == False:
            corrCountriesHold2 = corrCountries2.iloc[i][j]
            if corrCountriesHold2 > corrVarietiesInt:
                corrCountriesInt = corrCountriesHold2
            break
for i in range(0, len(corrCountries3)):
    for j in range(0, len(corrCountries3)):
        if corrCountries3.iloc[i][j] != 1 and math.isnan(corrCountries3.iloc[i][j]) == False:
            corrCountriesHold3 = corrCountries3.iloc[i][j]
            if corrCountriesHold3 > corrVarietiesInt:
                corrCountriesInt = corrCountriesHold3
            break
for i in range(0, len(corrVarieties1)):
    for j in range(0, len(corrVarieties1)):
        if corrVarieties1.iloc[i][j] != 1 and math.isnan(corrVarieties1.iloc[i][j]) == False:
            corrVarietiesHold1 = corrVarieties1.iloc[i][j]
            if corrVarietiesHold1 > corrVarietiesInt:
                corrVarietiesInt = corrVarietiesHold1
            break
for i in range(0, len(corrVarieties2)):
    for j in range(0, len(corrVarieties2)):
        if corrVarieties2.iloc[i][j] != 1 and math.isnan(corrVarieties2.iloc[i][j]) == False:
            corrVarietiesHold2 = corrVarieties2.iloc[i][j]
            if corrVarietiesHold2 > corrVarietiesInt:
                corrVarietiesInt = corrVarietiesHold2
            break
for i in range(0, len(corrVarieties3)):
    for j in range(0, len(corrVarieties3)):
        if corrVarieties3.iloc[i][j] != 1 and math.isnan(corrVarieties3.iloc[i][j]) == False:
            corrVarietiesHold3 = corrVarieties3.iloc[i][j]
            if corrVarietiesHold3 > corrVarietiesInt:
                corrVarietiesInt = corrVarietiesHold3
            break

#if corrCountries is greater than corrVariety, add 1 to countriesCounter
#if corrCountries is less than corrVariety, add 1 to varietiesCounter
#if corrCountries is equal to corrVariety, add 1 to both countriesCounter and varietiesCounter
if corrCountriesInt > corrVarietiesInt:
    countriesCounter += 1
elif corrCountriesInt < corrVarietiesInt:
    varietiesCounter += 1
else:
    countriesCounter += 1
    varietiesCounter += 1
   

#If corrCountries is greater than corrVariety, print 'Countries have a higher correlation between points and price'
#If corrCountries is less than corrVariety, print 'Varieties have a higher correlation between points and price'
#If corrCountries is equal to corrVariety, print 'Countries and varieties have the same correlation between points and price'

if countriesCounter > varietiesCounter:
    print('Countries have a higher correlation between points and price')
elif countriesCounter < varietiesCounter:
    print('Varieties have a higher correlation between points and price')
else:
    print('Countries and varieties have the same correlation between points and price')

print("Correlation Value for Countries: ", corrCountriesInt)
print("Correlation Value for Varieties: ", corrVarietiesInt)

    







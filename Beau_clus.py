# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:21:01 2020

@author: BeauChapman
"""
##############################################################################
#Reading in initial data######################################################
##############################################################################

import pandas as pd
import numpy as np
import pandas as pd 
from subprocess import check_output

dfb = pd.read_csv('https://raw.githubusercontent.com/bchap90210/bchap90210/Data-Files/acs2017_census_tract_data.csv')
print(df.columns)
print(df)
 
# Creating copy of data
cenData = dfb
print(cenData.columns)
print(cenData)
type(cenData)

stateP = cenData[['State','TotalPop','Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific','Poverty', 'ChildPoverty', 'Professional', 'Service', 'Office','Construction','Production','Drive','Carpool','Transit','Walk','OtherTransp','WorkAtHome', 'PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment']]


percentages = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific','Poverty', 'ChildPoverty', 'Professional', 'Service', 'Office','Construction','Production','Drive','Carpool','Transit','Walk','OtherTransp','WorkAtHome', 'PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment']
for i in percentages:
    stateP[i] = round(stateP['TotalPop'] * stateP[i] / 100) 

stateDF = stateP.groupby(['State']).sum()
print(stateDF)
display(stateDF)
print(stateDF.columns)
print(stateDF.dtypes)


import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

fig, ax = plt.subplots(figsize=(14,4))
fig = sns.barplot(x=stateDF.index.values, y=stateDF['TotalPop'], data=stateDF)
fig.axis(ymin=0, ymax=40000000)
plt.xticks(rotation=90)


##############################################################################
#Set Up Kmeans for Clustering#################################################
##############################################################################
import sklearn
from sklearn.cluster import KMeans
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
# from pca import pca

import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


# Set up kmeans
myKmeans = sklearn.cluster.KMeans(n_clusters=4)

# Fit the Kmeans alghorithm
result = myKmeans.fit(stateDF)

print(result)

y_kmeans = myKmeans.predict(stateDF)
print(y_kmeans)


# Define Kmeans labels
#labels = myKmeans.labels_
#print(labels)
# Get results of clustering
#Myresults = pd.DataFrame([stateDF.index,labels]).T
#print(Myresults)
# print DF
#print(stateDF.columns)
stateDF['cluster'] = y_kmeans
stateDF.head()
type(stateDF)


#X, _ = make_blobs(n_samples=10, centers=3, n_features=4)

#df = pd.DataFrame(stateDF, columns=df.iloc[:,0:30])

#kmeans = KMeans(n_clusters=4)

#y = kmeans.fit_predict(df.iloc[:,0:30])

#df['Cluster'] = y

#print(df.head())

printClusters = stateDF.index, stateDF.cluster
print(printClusters)


##############################################################################
#Bringing in Covid Cases by State#############################################
##############################################################################
# Reading in the data
df2 = pd.read_csv('CoreData.csv', header=0, encoding='latin-1')
print(df.columns)

df2 = df2.rename(columns = {'ï»¿State':'State'})

stsumDF = df2
print(stsumDF)
print(stsumDF.columns)
stsumDF['Confirmed']
# Getting rid of commas in the numeric data
#stsumDF["cases"] = stsumDF["cases"].str.replace(",","")
#stsumDF['cases']
#stsumDF['cases'] = stsumDF.cases.astype(float)

# Joining the two DFs
joinDF = stateDF.merge(stsumDF, on='State', how='left',)

joinDF.head()
joinDF.columns
joinDF.dropna(inplace=True)

# Getting rid of excess columns
#joinDF.drop(joinDF.columns[1:35], axis=1, inplace=True)
joinDF.head()

# Sorting the data
joinDF.sort_values(by=['Confirmed'])

# # Plots
# joinDF.plot(x ='cluster', y='Confirmed', kind = 'scatter', 
#                  xticks=joinDF.cluster.values)

# joinDF.plot(x ='Unemployment', y='Confirmed', kind = 'scatter')


# import matplotlib.pyplot as plt

# plt.style.use('default')
# ax = joinDF[['cluster','Confirmed']].plot(kind='scatter', x='cluster', y='Confirmed', 
#                                       title ="Covid-19 Cases By Cluster",
#                                       legend=True, fontsize=12)

# joinDF.plot(x='TotalPop', y='Confirmed', kind='scatter')





# fig, ax = plt.subplots(figsize=(14,4))
# fig = sns.scatterplot( x=joinDF.Confirmed, y=joinDF.TotalPop, data=joinDF)
# fig.axis(ymin=0, ymax=40000000)
# plt.xticks(rotation=90)
# z = np.polyfit(x, y, 1)
# p = np.poly1d(z)
# plt.plot(x,p(x),"r--")
# plt.show()



# sns.lmplot(x='TotalPop', y='Confirmed', data=joinDF)

# sns.lmplot(x ='cluster', y='cases', data=joinDF)

# sns.lmplot(x ='Unemployment', y='cases', data=joinDF)


##############################################################################
#Plots with the Clusters######################################################
##############################################################################
# plot = joinDF[['Confirmed', 'cluster']]
# plot = plot.groupby(['cluster']).mean()
# plot.head()
# #plot = plot.set_index('cluster')
# plot.columns = ['Confirmed']
# plot = plot.plot(figsize=(10, 6), kind='bar')
# plot.set_ylabel('Average Number of Cases')
# plot.set_xlabel('Cluster')
# plt.show()

# plot2 = joinDF[['TotalPop', 'cluster']]
# plot2 = plot2.groupby(['cluster']).mean()
# plot2.head()
# #plot = plot.set_index('cluster')
# plot2.columns = ['TotalPop']
# plot2 = plot2.plot(figsize=(10, 6), kind='bar')
# plot2.set_ylabel('Average Population')
# plot2.set_xlabel('Cluster')
# plt.show()


###################### MANSO ########################

joinDF = joinDF.groupby(["State"]).max().reset_index()






















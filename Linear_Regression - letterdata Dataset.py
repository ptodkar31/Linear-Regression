# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:24:59 2024

@author: Priyanka
"""
#Suppoert Vector Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
letter = pd.read_csv("C:\Data Set\letterdata.csv")

letter.dtypes

'''
letter    object
xbox       int64
ybox       int64
width      int64
height     int64
onpix      int64
xbar       int64
ybar       int64
x2bar      int64
y2bar      int64
xybar      int64
x2ybar     int64
xy2bar     int64
xedge      int64
xedgey     int64
yedge      int64
yedgex     int64

'''
#EDA
letter.shape  #(20000, 17)

plt.figure(1,figsize=(16,10))
sns.countplot(x=letter["xbox"])

sns.countplot(x=letter["ybox"])


sns.displot(letter.width)
#data isnormal and slight left skewed
sns.boxplot(letter.width)
#there are sevreal outliers

sns.displot(letter.height)
#data is normal and slight left skewed
sns.boxplot(letter.height)
#there are outlier
sns.displot(letter.onpix)
#data is normal and slight right skewed
sns.boxplot(letter.onpix)
#there are outlier
sns.displot(letter.xbar)
#data is normal
sns.boxplot(letter.xbar)
#there are outlier
sns.displot(letter.ybar)
#data is normal
sns.boxplot(letter.ybar)
#there are outlier

#now let us check the highest fire In KM?
letter.sort_values(by="ybar",ascending  = False).head(5)

highest_fire_ybar = letter.sort_values(by="ybar",ascending=True)

plt.figure(figsize=(8,6))









#############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

letter=pd.read_csv("C:/Data Set/letterdata.csv")
letter.dtypes
letter.head
letter.columns

letter.shape
plt.figure(1,figsize=(16,10))
sns.countplot(x=letter['letter']) 
#Letter U is definedmultiple times
sns.countplot(x=letter['onpix'])
#Data is right skewed and 2 has the highst value
sns.displot(letter.height)
#data is normal and slight right skewed
sns.boxplot(letter.width)
#There are several Outliers
###########################################################
#Now let us check highest fire in KM?
letter.sort_values(by='height',ascending=False).head(5)

highest_fire_area=letter.sort_values(by='width',ascending=True)

plt.figure(figsize=(8,6))
plt.title('Temperature vs Area of fire')
plt.bar(highest_fire_area['hight'],
         highest_fire_area['width'])

plt.xlabel('Height')
plt.ylabel('Width')
plt.show()

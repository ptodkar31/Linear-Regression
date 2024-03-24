# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:15:38 2024

@author: Priyanka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
forest = pd.read_csv("C:/Data Set/forestfires.csv")

forest.dtypes
'''
month             object
day               object
FFMC             float64
DMC              float64
DC               float64
ISI              float64
temp             float64
RH                 int64
wind             float64
rain             float64
area             float64
dayfri             int64
daymon             int64
daysat             int64
daysun             int64
daythu             int64
daytue             int64
daywed             int64
monthapr           int64
monthaug           int64
monthdec           int64
monthfeb           int64
monthjan           int64
monthjul           int64
monthjun           int64
monthmar           int64
monthmay           int64
monthnov           int64
monthoct           int64
monthsep           int64
size_category     object
'''

#EDA
forest.shape  #(517, 31)

plt.figure(1,figsize=(16,10))
sns.countplot(x=forest["month"])
#AUG and SEPT has highest value
sns.countplot(x=forest["day"])
#Friday Sunday and Staurday has highest value

sns.displot(forest.FFMC)
#data isnormal and slight left skewed
sns.boxplot(forest.FFMC)
#there are sevreal outliers

sns.displot(forest.RH)
#data is normal and slight left skewed
sns.boxplot(forest.RH)
#there are outlier
sns.displot(forest.wind)
#data is normal and slight right skewed
sns.boxplot(forest.wind)
#there are outlier
sns.displot(forest.rain)
#data is normal
sns.boxplot(forest.rain)
#there are outlier
sns.displot(forest.area)
#data is normal
sns.boxplot(forest.area)
#there are outlier


#now let us check the highest fire In KM?
forest.sort_values(by="area",ascending  = False).head(5)

highest_fire_area = forest.sort_values(by="area",ascending=True)

plt.figure(figsize=(8,6))

plt.title("Temperature vs area of fire")
plt.bar(highest_fire_area["temp"],highest_fire_area["area"])

plt.xlabel("Temperature")
plt.ylabel("Area per km-sq")
plt.show()
#ones the fire start almost 1000+ sq area's
#temperature goes beyound 25 and
#around 750km area is facing temp 30+
#Now let us check the highest rain in the forest
highest_rain=forest.sort_values(by='rain',ascending=True)[['month','day','rain']].head(5)
highest_rain
#highest rainn observed in the month of aug
#let us check highest and lowest temperature in month and

highest_temp = forest.sort_values(by='temp',ascending=False)[['month','day','rain']].head(5)

lowest_temp = forest.sort_values(by='temp',ascending=True)[['month','day','rain']].head(5)

print("Highest Temperature",highest_temp)
#Highest them Observation is AUG
print("Lowest temperature",lowest_temp)
#lowest temperature is in the month of Dec
forest.isna().sum()
#there is no null values for the columns

#Sal1.dtypes

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
forest.month=labelencoder.fit_transform(forest.month)
forest.day=labelencoder.fit_transform(forest.day)
forest.size_category=labelencoder.fit_transform(forest.size)

forest.dtypes
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['month'])
df_t=winsor.fit_transform(forest[["month"]])
sns.boxenplot(df_t.month)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['day'])
df_t=winsor.fit_transform(forest[["day"]])
sns.boxenplot(df_t.day)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['FFMC'])
df_t=winsor.fit_transform(forest[["FFMC"]])
sns.boxenplot(df_t.FFMC)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['RH'])
df_t=winsor.fit_transform(forest[["RH"]])
sns.boxenplot(df_t.RH)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['wind'])
df_t=winsor.fit_transform(forest[["wind"]])
sns.boxenplot(df_t.wind)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['rain'])
df_t=winsor.fit_transform(forest[["rain"]])
sns.boxenplot(df_t.rain)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['area'])
df_t=winsor.fit_transform(forest[["area"]])
sns.boxenplot(df_t.area)

tc=forest.corr()
tc
fig,ax=plt.subplots()
fig.set_size_inches(200,10)
sns.heatmap(tc,annot=True,cmap='YlGnBu')


#all the variables are moderately correlated with size_category

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test=train_test_split(forest,test_size=0.3)
train_x=train.iloc[:,:30]
train_y=train.iloc[:,30]
test_x=test.iloc[:,:30]
test_y=test.iloc[:,30]

#Kernel linear
model_linear =SVC(kernel='linear')
model_linear.fit(train_x,train_y)
pred_test_linear=model_linear.predict(test_x)
np.mean(pred_test_linear==test_y)

#RBF
model_rbf=SVC(kernel='rbf')
model_rbf.fit(train_x,train_y)
pred_test_rbf=model_rbf.predict(test_x)
np.mean(pred_test_rbf==test_y)



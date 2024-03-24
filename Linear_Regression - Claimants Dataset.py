# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:11:13 2024

@author: Priyanka
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report

claimants=pd.read_csv("C:\Data Set\claimants.csv")
#There are CLMAGE and LOSS are having continious data rest are
#verify the dataset, where CASENUM is not really useful so droping 
c1=claimants.drop('CASENUM',axis=1)
c1.head(11)
c1.describe()
#let us check whether there are null values
c1.isna().sum()
#there are several null values
#if we will used dropna() function we will loss 290 data points
#hence we will go imputation
c1.dtypes
mean_value=c1.CLMAGE.mean()
mean_value
# mean_values is 28.41 

#now let us impute the same
c1.CLMAGE = c1.CLMAGE.fillna(mean_value)
# 189
c1.CLMAGE.isna().sum()
# 0
#hence all null values of CLMAGE has been filled by mean values
#for columns where there are discrete values 
#we will apply mode

mode_CLMSEX = c1.CLMSEX.mode()
mode_CLMSEX
c1.CLMSEX= c1.CLMSEX.fillna((mode_CLMSEX)[0])
c1.CLMSEX.isna().sum()

#CLMINSUR is also categorical data hence mode impution is application
mode_CLMINSUR= c1.CLMINSUR.mode()
mode_CLMINSUR

c1.CLMINSUR = c1.CLMINSUR.fillna((mode_CLMINSUR)[0])
c1.CLMINSUR.isna().sum()

#SEATBELT  is categorical data hence go for mode imputation
mode_SEATBELT= c1.SEATBELT.mode()
mode_SEATBELT


c1.SEATBELT = c1.SEATBELT.fillna((mode_SEATBELT)[0])
c1.SEATBELT .isna().sum()

#now the person we met an accident will hire the atternery or not
#let us build the model

logit_model=sm.logit('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=c1).fit()
logit_model.summary()

#in logistic regression we do not have R sqaured values only check p=values
#SEATBELT is statistically insignificant ignore and proceed
logit_model.summary2()
#here we are going to check AIC values it stands for Akaike information criterion
#is mathematically method for evaluation how well a model fits the data
#a lower the score ,more the better model AIC scores are only useful in comparisions
#with other  AIC scores for the same dataset


#now let us go for predictions
pred=logit_model.predict(c1.iloc[:,1:])
#here we are applying all rows columns from 1, as columns 0 is ATTORNEY target values


#let us check the performance of the model
fpr,tpr,thresholds = roc_curve(c1.ATTORNEY,pred)

#we are applying actual values and predicted values so as to get
#falues positive rate,true postitive rate and thresholds
#the optimal cutoff values is point where there ishigh true positive 
#you can use the below code to get the values:

optimal_idx = np.argmax(tpr-fpr)
optimal_thresholds=thresholds[optimal_idx]
optimal_thresholds
#ROC :receiver operating characterstics curve in logistic regression are 
#determine best cutoff/threshold values


import pylab as pl
i= np.range(len(tpr))
#index for df
#here tpr is of 599 so it will create a scale from 0 to 558
roc=pd.DataFrame({'fpr':pd.Series(fpr,index=i),'tpr':pd.Series(tpr,index=i),'1-fpr':pd.Series(1-fpr,index=i),'tf':pd.Series(tpr - (1-fpr),index=i),'thresholds':pd.Series(thresholds,index=i)})
#we want to create a dataframe which comprises of columns fpr
#tpr,1-fpr,tpr-(1-fpr=tf)
#the optimal cut off would be where tpr is high and fpr is low
#tpr - (1-fpr) is zero or near to zero is the optimal cutoff point

#plot ROC curve

plt.plot(fpr,tpr)
plt.xlabel("false postitive rate");
plt.ylabel("True postitive rate")
roc.iloc[(roc.tf-0).abs().argsort()[:1]]
roc_auc=auc(fpr,tpr)
print("area under the curve:%f"% roc_auc)
#Area is 0.7601

#tpr vs 1-fpr
#plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'],color='red')
pl.plot(roc['1-fpr'],color='blue')
pl.xlabel("1-false postitive rate")
pl.ylabel("True postitive rate")
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])
#the optimal cut off point is one where tpr is high and fpr is low
#the optimal cut off point is 0.317628
#so any thing above this can be labeled as 1 else 0
#you can see from the output/chart that where TPR is crossing 1-FPR the TPR is 63%
#FPR is 36% and TPR-(1-fpr) is nearest to zero in the current example

#fillinf all the cells with zeros
c1['pred']=np.zeros(1340)
c1.loc[pred>optimal_thresholds,'pred']=1
#let us check the classificatin report
classificatin=classification_report(c1['pred'],c1["ATTORNEY"])
classificatin

#splitting the data into train and test
train_data,test_data=train_test_split(c1,test_size=0.3)

#model building
model=sm.report('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=train_data).fit()
model.summary()

# p values are below the condition of 0.05 but SEATBELT has got in statisticallyin siginificant
model.summary2()

#AIC values is 1110.3782 AIC score are useful in comparisions with other
#lower the AIC Score better the model

#let us go for predictions
test_pred =logit_model.predict(test_data)

test_data['test_pred']=np.zeros(402)
test_data.ioc[test_pred>optimal_thresholds,'test_pred']=1
# Confusion matrix
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.ATTORNEY)
confusion_matrix
accracy_test=(143+151)/(402)
accracy_test
# It is 0.6940298

#Clasification repirt
classification_test=classification_report(test_data['test_pred'],test_data.ATTORNEY)
classification_test
# Accuracy os 0.73

# ROC curve and AUC
fpr,tpr,thresholds=metrics.roc_curve(test_data['ATTORNEY'],test_pred)

# Plot the ROC Curve
plt.plot(fpr,tpr);plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')

# Area under the curve
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test

# Prediction on train Data
train_pred=logit_model.predict(train_data)
train_data['train_pred']=np.zeros(938)
train_data.loc[train_pred>optimal_thresholds,'train_pred']=1


# Confusion matrix
confusion_matrix=pd.crosstab(train_data.train_pred,train_data.ATTORNEY)
confusion_matrix

accuracy_train=(315+347)/(938)
accuracy_train
# It is 0.7217484000

# Classification Report
classification_train=classification_report(train_data['train_pred'],train_data['ATTORNEY'])
classification_train
# ccuracy is 0.69

#ROC Curve and AUC
fpr,tpr,thresholds=metrics.roc_curve(train_data['ATTORNEY'],train_pred)

# Plot the ROC Curve
plt.plot(fpr,tpr);plt.xlabel('False Positiove Rate');plt.ylabel('True Positive Rate')
# Area under the curve
roc_auc_train=metrics.auc(fpr,tpr)


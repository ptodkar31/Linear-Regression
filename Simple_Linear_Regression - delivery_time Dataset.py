# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:32:45 2024

@author: Priyanka
"""

"""
A logistics company recorded the time taken for delivery and the time taken 
for the sorting of the items for delivery. Build a Simple Linear Regression model
to find the relationship between delivery time and sorting time with 
delivery time as the target variable.Apply necessary transformations and 
record the RMSE and correlation coefficient values for different models.

Business Problem:
What is the business objective?
Delivery time is a crucial measure in transportation.
Accurate delivery time prediction is also fundamental for operation 
and advanced logistic systems.
prediction of delivery time with the increase of demand volume and e-commerce 
activities, logistic services face more challenges in order to maintai efficiency 
and customer retention

Are there any constraints?
Reliable long-term predictions remain challenging.There are unseen factors 
which contribute in delibery time.
"""


import pandas as pd
import numpy as np
import seaborn as sns
time=pd.read_csv("C:\Data Set\Simple_Linear_Regression\delivery_time.csv")
time.dtypes
'''
Delivery Time    float64
Sorting Time       int64
'''
time.columns="del_time","sort_time"

#EDA
time.describe()
'''
        del_time  sort_time
count  21.000000  21.000000
mean   16.790952   6.190476
std     5.074901   2.542028
min     8.000000   2.000000
25%    13.500000   4.000000
50%    17.830000   6.000000
75%    19.750000   8.000000
max    29.000000  10.000000
'''

#Average delivery time is 16.71 and min is 8 and max is 29
#Average sort_time is 6.1 and min is 2.0 and max is 10
import matplotlib.pyplot as plt

sns.distplot(time.del_time)
#Data is normal but right skewed
plt.boxplot(time.del_time)
#No outliers but slight right skewed

sns.distplot(time.sort_time)
#Data is normal distributed 
plt.boxplot(time.sort_time)
#No outliers 


#Bivariant analysis
sns.regplot(x=time['sort_time'],y=time['del_time'])

#Data is linearly scattered,direction positive,
#Now let us check the correlation coeficient
np.corrcoef(x=time['sort_time'],y=time['del_time'])
'''
array([[1.        , 0.82599726],
       [0.82599726, 1.        ]])
'''
#The correlation coeficient is 0.8259<0.85 hence the correlation is moderate
#Let us check the direction of correlation
cov_output=np.cov(time.sort_time,y=time.del_time)[0,1]
cov_output
#10.655809523809523,it is positive means correlation will be positive


# let us apply to various models and check the feasibility
import statsmodels.formula.api as smf
#first simple linear model
model=smf.ols('del_time~sort_time',data=time).fit()

model.summary()
#R-sqarred=0.682<0.85,model is not fit as good
#p=0.01<0.05 hence acceptable
#bita-0=6.5827
#bita-1=1.6490
pred1=model.predict(pd.DataFrame(time.sort_time))
pred1
'''
0     23.072933
1     13.178814
2     16.476853
3     21.423913
4     23.072933
5     16.476853
6     18.125873
7     11.529794
8     23.072933
9     21.423913
10    19.774893
11    13.178814
12    18.125873
13    11.529794
14    11.529794
15    13.178814
16    16.476853
17    18.125873
18     9.880774
19    18.125873
20    14.827833
'''

#Regression line
sns.regplot(x=time['sort_time'],y=time['del_time'])
plt.plot(time.sort_time,pred1,'r')
plt.legend(['Predicted line','Observed data'])
plt.show()


#Error calculations
res1=time.del_time-pred1
np.mean(res1)
#-3.891067362495787e-15
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#2.7916503270617654



plt.scatter(x=np.log(time.sort_time),y=time.del_time)
#Data is not linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=np.log(time.sort_time),y=time.del_time)
#The correlation coeficient is 0.833<0.85 hence the correlation moderate
#r=0.8217
model2=smf.ols('time.del_time~np.log(time.sort_time)',data=time).fit()
#Y is wt_gained and X =log(cal_consumed)
model2.summary()
#R-sqarred=0.695<0.85,there is scope of improvement
#p=0.642>0.05 hence not acceptable
#bita-0=-6955.6501
#bita-1=np.log(cal_consumed)   948.37
pred2=model.predict(pd.DataFrame(time.sort_time))
pred2


#Regression line
plt.scatter(x=np.log(time.sort_time),y=time.del_time)
plt.plot(np.log(time.sort_time),pred2,'r')
plt.legend(['Predicted line','Observed data_model2'])
plt.show()

#Error calculations
res2=time.del_time-pred2
np.mean(res2)
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#2.7916503270617654
#This model has very poor r value,R-squarred value 
#Hence let us try another model


#Now let us make logY and X as is
#x=logistic['sort_time'],y=np.log(logistic['del_time'])
plt.scatter(x=(time.sort_time),y=np.log(time.del_time))
#Data is not linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=(time.sort_time),y=np.log(time.del_time))
#The correlation coeficient is 0.8431<0.85 hence the correlation is moderate
#r=0.8431
model3=smf.ols('np.log(del_time)~sort_time',data=time).fit()
#Y is log(AT) and X =Waist
model3.summary()
#R-sqarred=0.711<0.85
#p=0.000<0.05 hence acceptable
#bita-0=2.1214   
#bita-1=   0.1056
pred3=model3.predict(pd.DataFrame(time.sort_time))
pred3_at=np.exp(pred3)
pred3_at


#Regression line
plt.scatter(time.sort_time,np.log(time.del_time))
plt.plot(time.sort_time,pred3,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()

#Error calculations
res3=time.del_time-pred3_at
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#2.9402503230562007,higher than earlier two models

#Hence let us try another model

x = time['sort_time'] + time['sort_time'] * time['sort_time']
y = np.log(time['del_time'])


#polynomial model
#Here r can not be calculated
model4=smf.ols('np.log(del_time)~sort_time+I(sort_time*sort_time)',data=time).fit()
#Y=np.log(cal.wt_gained) and X=cal.cal_consumed
model4.summary()
#R-sqarred=0.765<0.85
#p=0.022 <0.05 hence acceptable
#bita-0=1.6997 
#bita-1=   0.2659 
pred4=model4.predict(pd.DataFrame(time.sort_time))
pred4
pred4_at=np.exp(pred4)
pred4_at



#Regression line
plt.scatter(time.sort_time,np.log(time.del_time))
plt.plot(time.sort_time,pred4,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()


#error calculations
res4=time.del_time-pred4_at
res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
# 2.799041988740927
#model but higher than second model,which was 103.30


data={"model":pd.Series(["SLR","Log_model","Exp_model","Poly_model"])}
data
table_rmse=pd.DataFrame(data)
table_rmse

#We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(time,test_size=0.3)

plt.scatter(train.sort_time,train.del_time)
plt.scatter(test.sort_time,test.del_time)


#Now let us check the correlation coeficient
np.corrcoef(x=train.sort_time,y=train.del_time)
#The correlation coeficient is 0.9274>0.85 hence the correlation is good
#r=0.9274
final_model=smf.ols('del_time~sort_time',data=time).fit()
#Y is del_timeand X =sort_time
final_model.summary()
#R-sqarred=0.682<0.85,there is scope of improvement
#p=0.01<0.05 hence acceptable
#bita-0= 6.5827 
#bita-1=   1.6490
test_pred=final_model.predict(pd.DataFrame(test))
test_pred

test_res=test.del_time-test_pred

test_res_sqr=test_res*test_res
test_mse=np.mean(test_res_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#3.649192264685347


train_pred=final_model.predict(pd.DataFrame(train))
train_pred

train_res=train.del_time-train_pred

train_res_sqr=train_res*train_res
train_mse=np.mean(train_res_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#2.243137363283044

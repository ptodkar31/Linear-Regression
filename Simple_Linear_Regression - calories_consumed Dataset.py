# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:30:17 2024

@author: Priyanka
"""
"""
A certain food-based company conducted a survey with the help of a fitness company 
to find the relationship between a person’s weight gain and the number of calories 
they consumed in order to come up with diet plans for these individuals. 
Build a Simple Linear Regression model with calories consumed as the target variable.
Apply necessary transformations and record the RMSE and correlation coefficient 
values for different models. 

Business Problem:
What is the business objective?
If people consumed only the number of calories needed every day, 
they would probably have healthy lives.
Calorie consumption that is too low or too high will eventually lead to health problems.
prediction of The number of calories in food tells us how much potential energy they contain. 
It is not only calories that are important,but also the substance from which the calories are taken.

Are there any constraints?
It also appears that weight regain occurs regardless of the type of diet used 
for weight loss, although some diets are linked to less regain than others.
"""

import pandas as pd
import numpy as np
import seaborn as sns
calories=pd.read_csv("C:\Data Set\Simple_Linear_Regression\calories_consumed.csv")
calories.dtypes
'''
Weight gained (grams)    int64
Calories Consumed        int64
'''
calories.columns="wt_gained","cal_consumed"

#EDA
calories.describe()
'''
wt_gained  cal_consumed
count    14.000000     14.000000
mean    357.714286   2340.714286
std     333.692495    752.109488
min      62.000000   1400.000000
25%     114.500000   1727.500000
50%     200.000000   2250.000000
75%     537.500000   2775.000000
max    1100.000000   3900.000000
'''
#Average weight gained is 357.71 and min is 62.00 and max is 1100 gms
#Average Calories consumed is 2340 and min is 1400 and max is 3900


import matplotlib.pyplot as plt
plt.bar(height=calories.wt_gained,x=np.arange(1,110,1))
sns.distplot(calories.wt_gained)

#Data is normal but right skewed
plt.boxplot(calories.wt_gained)
#No outliers but right skewed
plt.bar(height=calories.cal_consumed,x=np.arange(1,110,1))
sns.distplot(calories.cal_consumed)
#Data is normal distributed 
plt.boxplot(calories.cal_consumed)
#No outliers but slight right skewed


#Bivariant analysis
plt.scatter(x=calories.cal_consumed,y=calories.wt_gained)
#Data is linearly scattered,direction positive,
#Now let us check the correlation coeficient
np.corrcoef(x=calories.cal_consumed,y=calories.wt_gained)
#The correlation coeficient is 0.94699101>0.85 hence the correlation is strong
#Let us check the direction of correlation
cov_output=np.cov(calories.cal_consumed,calories.wt_gained)[0,1]
cov_output
#237669.4505494506,it is positive means correlation will be positive

# let us apply to various models and check the feasibility
import statsmodels.formula.api as smf
#first simple linear model
model=smf.ols('wt_gained~cal_consumed',data=calories).fit()
#Y is AT and X is waist
model.summary()
#R-sqarred=0.897>0.85,model is good
#p=00<0.05 hence acceptable
#bita-0=-625.7524 
#bita-1=0.4202
pred1=model.predict(pd.DataFrame(calories.cal_consumed))
pred1
'''
0        4.482599
1      340.607908
2      802.780209
3      298.592245
4      424.639236
5       46.498263
6      -37.533065
7      172.545254
8      550.686227
9     1012.858527
10      75.909227
11     172.545254
12     508.670563
13     634.717554
'''

#Regression line
plt.scatter(calories.cal_consumed,calories.wt_gained)
plt.plot(calories.cal_consumed,pred1,'r')
plt.legend(['Predicted line','Observed data'])
plt.show()


#error calculations
res1=calories.wt_gained-pred1
np.mean(res1)
#3.004580711214138e-13
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#103.30250194726935
#RMSE is very high


#let us try another model
#x=log(cal.cal_consumed)
plt.scatter(x=np.log(calories.cal_consumed),y=calories.wt_gained)
#Data is linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=np.log(calories.cal_consumed),y=calories.wt_gained)

'''
array(
      [[1.        , 0.89872528],
       [0.89872528, 1.        ]]
      )
'''

#The correlation coeficient is 0.89872528>0.85 hence the correlation is good
#r=0.8217
model2=smf.ols('wt_gained~np.log(cal_consumed)',data=calories).fit()
#Y is wt_gained and X =log(cal_consumed)
model2.summary()
#R-sqarred=0.808<0.85,there is scope of improvement
#p=00<0.05 hence acceptable
#bita-0=-6955.6501
#bita-1=np.log(cal_consumed)   948.37
pred2=model.predict(pd.DataFrame(calories.cal_consumed))
pred2

#Regression line
plt.scatter(np.log(calories.cal_consumed),calories.wt_gained)
plt.plot(np.log(calories.cal_consumed),pred2,'r')
plt.legend(['Predicted line','Observed data_model2'])
plt.show()


#error calculations
res2=calories.wt_gained-pred2
np.mean(res2)
#3.004580711214138e-13
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
# 103.30250194726935
#Better as compared to earlier,which was 2027.04
#Hence let us try another model


#Now let us make logY and X as is
plt.scatter(x=(calories.cal_consumed),y=np.log(calories.wt_gained))
#Data is linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=(calories.cal_consumed),y=np.log(calories.wt_gained))
'''
array([[1.        , 0.93680369],
       [0.93680369, 1.        ]])
'''
#The correlation coeficient is 0.8185<0.85 hence the correlation is moderate
#r=0.9368
model3=smf.ols('np.log(wt_gained)~cal_consumed',data=calories).fit()
#Y is log(AT) and X =Waist
model3.summary()
#R-sqarred=0.878>0.85
#p=0.000<0.05 hence acceptable
#bita-0=2.8387   
#bita-1=   0.0011
pred3=model3.predict(pd.DataFrame(calories.cal_consumed))
pred3_at=np.exp(pred3)
pred3_at
'''
0       93.603577
1      231.816603
2      806.661188
3      206.972681
4      290.808810
5      104.839263
6       83.572027
7      147.305340
8      408.603511
9     1421.833419
10     113.497427
11     147.305340
12     364.813232
13     512.584083
'''

#Regression line
plt.scatter(calories.cal_consumed,np.log(calories.wt_gained))
plt.plot(calories.cal_consumed,pred3,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()


#error calculations
res3=calories.wt_gained-pred3_at
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#118.04515720118094
# Better as compared to first model but higher than second model,which was 103.30
#Hence let us try another model


#Now let us make Y=np.log(cal.wt_gained) and X=cal.cal_consumed,X*X=cal.cal_consumed*cal.cal_consumed
#polynomial model
#Here r can not be calculated
model4=smf.ols('np.log(wt_gained)~cal_consumed+I(cal_consumed*cal_consumed)',data=calories).fit()
#Y=np.log(cal.wt_gained) and X=cal.cal_consumed
model4.summary()
#R-sqarred=0.0.878>0.85
#p=0.022 <0.05 hence acceptable
#bita-0=2.8287
#bita-1=   0.0011 
pred4=model4.predict(pd.DataFrame(calories.cal_consumed))
pred4
'''
0     4.538170
1     5.446795
2     6.692653
3     5.333334
4     5.673616
5     4.651865
6     4.424441
7     4.992750
8     6.013597
9     7.257612
10    4.731432
11    4.992750
12    5.900303
13    6.240083
'''

pred4_at=np.exp(pred4)
pred4_at
'''
0       93.519497
1      232.013330
2      806.459099
3      207.127381
4      291.085209
5      104.780251
6       83.466143
7      147.341113
8      408.951512
9     1418.864795
10     113.457933
11     147.341113
12     365.148180
13     512.900975
'''

#Regression line
plt.scatter(calories.cal_consumed,np.log(calories.wt_gained))
plt.plot(calories.cal_consumed,pred4,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()


#error calculations
res4=calories.wt_gained-pred4_at
res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#117.41450013144271
#Better as compared to third model but higher than second model,which was 103.30



data={"model":pd.Series(["SLR","Log_model","Exp_model","Poly_model"])}
data

table_rmse=pd.DataFrame(data)
table_rmse

#We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(calories,test_size=0.3)

plt.scatter(np.log(train.cal_consumed),train.wt_gained)
plt.scatter(np.log(test.cal_consumed),test.wt_gained)

#Data is linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=np.log(train.cal_consumed),y=train.wt_gained)
'''
array([[1.      , 0.907505],
       [0.907505, 1.      ]])
'''
#The correlation coeficient is 0.9224386>0.85 hence the correlation is good
#r=0.9224386
final_model=smf.ols('wt_gained~np.log(cal_consumed)',data=calories).fit()
#Y is wt_gained and X =log(cal_consumed)
final_model.summary()
#R-sqarred=0.808<0.85,there is scope of improvement
#p=00<0.05 hence acceptable
#bita-0=-6955.6501
#bita-1=np.log(cal_consumed)   948.37
test_pred=final_model.predict(pd.DataFrame(test))
test_pred
'''
0    -19.998702
1    385.377115
4    464.453875
9    886.181334
7    204.185731
'''

test_res=test.wt_gained-test_pred
test_res_sqr=test_res*test_res
test_mse=np.mean(test_res_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#169.54564196820786



train_pred=final_model.predict(pd.DataFrame(train))
train_pred
'''
11    204.185731
8     571.931596
13    637.362484
12    537.441550
3     343.220320
6     -85.429591
5      41.207806
2     756.063670
10     81.817081
'''

train_res=train.wt_gained-train_pred
train_res_sqr=train_res*train_res
train_mse=np.mean(train_res_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#122.30500297858323
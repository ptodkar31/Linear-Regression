# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:31:59 2024

@author: Priyanka
"""

"""
A certain university wants to understand the relationship between students’ SAT scores and 
their GPA. Build a Simple Linear Regression model with GPA as the target variable and record 
the RMSE and correlation coefficient values for different models.


Business Problem:
What is the business objective?
That said, keep in mind that your GPA is not at all worthless! While the SAT might
have won the match, the GPA manages to steal several rounds. It is the only numerical 
data that can reveal hard work, self-discipline, and consistency. Together with your 
transcript, your GPA can show improvement over time and intellectual growth. 
These qualities are quite valuable when you are being compared to another student 
with a similar SAT score but lesser GPA. It is still important to take rigorous
classes and earn good grades to bolster your transcript and GPA.

Are there any constraints?
If you have high SAT scores, are you likely to get better grades in college?  
Perhaps unsurprisingly,most research studies find that your SAT scores 
do predict college success - to an extent.The relationship isn't particularly strong, 
hich means that if you have high SAT scores,you're only slightly more likely to have 
higher college grades than a student who had lower scores. Research shows a similar 
relationship between SAT scores and high school grades.        

"""

           
import pandas as pd
import numpy as np
import seaborn as sns
sat=pd.read_csv("C:\Data Set\Simple_Linear_Regression\SAT_GPA.csv")
sat.dtypes
'''
SAT_Scores      int64
GPA           float64
'''

#EDA
sat.describe()
'''
SAT_Scores         GPA
count  200.000000  200.000000
mean   491.810000    2.849500
std    174.893834    0.541076
min    202.000000    2.000000
25%    349.750000    2.400000
50%    480.500000    2.800000
75%    641.500000    3.400000
max    797.000000    3.900000
'''
#Average Sat score is 491.81 and min is 202 and max is 797
#Average GPA is 2.84 and min is 2 and max is 3.9
import matplotlib.pyplot as plt


sns.distplot(sat.SAT_Scores)
#Data is normal 
plt.boxplot(sat.SAT_Scores)
#No outliers 

sns.distplot(sat.GPA)
#Data is normal distributed ,bimodal
plt.boxplot(sat.GPA)
#No outliers 


#Bivariant analysis
sns.regplot(x=sat.SAT_Scores,y=sat.GPA)

#Data is not linearly scattered,direction positive,
#Now let us check the correlation coeficient
np.corrcoef(x=sat.SAT_Scores,y=sat.GPA)
'''
array(
      [[1.        , 0.29353828],
       [0.29353828, 1.        ]]
      )
'''
#The correlation coeficient is 0.2935<0.85 hence the correlation is very poor
#Let us check the direction of correlation
cov_output=np.cov(sat.SAT_Scores,sat.GPA)[0,1]
cov_output
#27.777793969849263 it is positive means correlation will be positive


# let us apply to various models and check the feasibility
import statsmodels.formula.api as smf
#first simple linear model
model=smf.ols('GPA~SAT_Scores',data=sat).fit()
#x=sat.SAT_Scores,y=sat.GPA
model.summary()
#R-sqarred=0.086<<0.85,model is not best fit 
#p=0.00<0.05 hence acceptable

pred1=model.predict(pd.DataFrame(sat.SAT_Scores))
pred1
'''
0      2.589947
1      2.597212
2      3.054002
3      2.929588
4      2.769757
         ...
195    2.865111
196    2.826061
197    3.014044
198    3.075797
199    2.638078
'''



#Regression line
sns.regplot(x=sat.SAT_Scores,y=sat.GPA)
plt.plot(sat.SAT_Scores,pred1,'r')
plt.legend(['Predicted line','Observed data'])
plt.show()


#Error calculations
res1=sat.GPA-pred1
np.mean(res1)
#5.55111
res_sqr1=res1*res1
mse1=np.mean(res_sqr1)
rmse1=np.sqrt(mse1)
rmse1
#0.5159457227723684



#let us try another model
sns.regplot(x=np.log(sat.SAT_Scores),y=sat.GPA)
#Data is not at all linearly scattered,direction positive,strength:poor
#Now let us check the correlation coeficient
np.corrcoef(x=np.log(sat.SAT_Scores),y=sat.GPA)
#The correlation coeficient is 0.2777<<0.85 hence the correlation very poor
#r=0.2777
model2=smf.ols('GPA~np.log(SAT_Scores)',data=sat).fit()
#Y is GPA and X =log(SAT_Scores)
model2.summary()
#R-sqarred=0.077<<<0.85
#p=0.412 >0.05 hence not acceptable

pred2=model.predict(pd.DataFrame(sat.SAT_Scores))
pred2
'''
0      2.589947
1      2.597212
2      3.054002
3      2.929588
4      2.769757
           ...
195    2.865111
196    2.826061
197    3.014044
198    3.075797
199    2.638078
'''

#Regression line
plt.scatter(x=np.log(sat.SAT_Scores),y=sat.GPA)
plt.plot(np.log(sat.SAT_Scores),pred2,'r')
plt.legend(['Predicted line','Observed data_model2'])
plt.show()


#Error calculations
res2=sat.GPA-pred2
np.mean(res2)
#5.55111
res_sqr2=res2*res2
mse2=np.mean(res_sqr2)
rmse2=np.sqrt(mse2)
rmse2
#0.515945722772368
#This model has same R-squarred value 
#Hence let us try another model


#Now let us make logY and X as is
plt.scatter(x=(sat.SAT_Scores),y=np.log(sat.GPA))
#Data is  linearly scattered,direction positive,strength:good
#Now let us check the correlation coeficient
np.corrcoef(x=(sat.SAT_Scores),y=np.log(sat.GPA))
#The correlation coeficient is 0.2940<0.85 hence the correlation is very poor
#r=0.2940
model3=smf.ols('np.log(GPA)~SAT_Scores',data=sat).fit()

model3.summary()
#R-sqarred=0.086<<0.85
#p=0.000<0.05 hence acceptable

pred3=model3.predict(pd.DataFrame(sat.SAT_Scores))
pred3_at=np.exp(pred3)
pred3_at
'''
0      2.555671
1      2.562188
2      3.007152
3      2.878816
4      2.721951
         ...
195    2.814476
196    2.776210
197    2.965323
198    3.030217
199    2.599158

'''

#Regression line
plt.scatter(sat.SAT_Scores,np.log(sat.GPA))
plt.plot(sat.SAT_Scores,pred3,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()


#Error calculations
res3=sat.GPA-pred3_at
res_sqr3=res3*res3
mse3=np.mean(res_sqr3)
rmse3=np.sqrt(mse3)
rmse3
#0.5175875893834133

#Hence let us try another model
#Now let us make Y=np.log(sat.GPA) and X=sat.SAT_Scores,X*X=sat.SAT_Scores*sat.SAT_Scores
#polynomial model
#Here r can not be calculated
model4=smf.ols('np.log(GPA)~SAT_Scores+I(SAT_Scores*SAT_Scores)',data=sat).fit()

model4.summary()
#R-sqarred=0.094<<0.85
#p=0.00<0.05 hence acceptable

#I(yrs_exp * yrs_exp)    -0.0066
pred4=model4.predict(pd.DataFrame(sat.SAT_Scores))
pred4
'''
0      0.971516
1      0.971245
2      1.112092
3      1.042931
4      0.987920
        ...
195    1.016160
196    1.002957
197    1.087366
198    1.126581
199    0.971186
'''

pred4_at=np.exp(pred4)
pred4_at

'''
0      2.641946
1      2.641231
2      3.040712
3      2.837523
4      2.685642
         ...
195    2.762567
196    2.726333
197    2.966451
198    3.085089
199    2.641076
'''

#Regression line
plt.scatter(sat.SAT_Scores,np.log(sat.GPA))
plt.plot(sat.SAT_Scores,pred4,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()

#Error calculations
res4=sat.GPA-pred4_at
res_sqr4=res4*res4
mse4=np.mean(res_sqr4)
rmse4=np.sqrt(mse4)
rmse4
#0.5144912487746157  
#This is the best model


data={"model":pd.Series(["SLR","Log_model","Exp_model","Poly_model"])}
data
table_rmse=pd.DataFrame(data)
table_rmse


#We have to generalize the best model
from sklearn.model_selection import train_test_split
train,test=train_test_split(sat,test_size=0.2)

plt.scatter(train.SAT_Scores,train.GPA)
plt.scatter(test.SAT_Scores,test.GPA)


#Now let us check the correlation coeficient
np.corrcoef(train.SAT_Scores,train.GPA)
#The correlation coeficient is 0.27433<<0.85 hence the correlation is very poor
#r=0.27433

final_model=smf.ols('np.log(GPA)~SAT_Scores+I(SAT_Scores*SAT_Scores)',data=sat).fit()

final_model.summary()
#R-sqarred=0.094<0.85
#p=0.00<0.05 hence acceptable


test_pred=final_model.predict(pd.DataFrame(test))
test_pred_at=np.exp(test_pred)
test_pred_at
'''
185    2.680321
165    2.860928
39     2.640350
198    3.085089
118    2.646661
150    2.640674
180    2.640717
24     2.849660
32     2.850895
94     2.718132
162    2.813328
92     2.784740
128    3.180392
181    2.733363
126    2.737395
29     3.159153
145    3.090836
53     3.098567
164    2.875149
31     2.817803
123    2.906508
19     2.893972
108    2.641016
8      2.645141
114    2.659470
158    2.667643
84     2.640331
88     2.747438
37     2.667643
103    2.642368
194    2.640775
14     2.704972
146    2.659118
2      3.040712
77     2.657072
74     2.677788
61     2.664756
18     2.950806
132    2.728644
182    3.110316
'''

test_res=test.GPA-test_pred_at

test_res_sqr=test_res*test_res
test_mse=np.mean(test_res_sqr)
test_rmse=np.sqrt(test_mse)
test_rmse
#0.5586330876387567


train_pred=final_model.predict(pd.DataFrame(train))
train_pred_at=np.exp(train_pred)
train_pred_at
'''
99     3.037139
13     2.676308
102    3.098567
186    2.646866
91     2.661656
         ....
41     3.064401
127    3.069984
95     2.680837
38     2.732568
191    2.748299
'''

train_res=train.GPA-train_pred_at

train_res_sqr=train_res*train_res
train_mse=np.mean(train_res_sqr)
train_rmse=np.sqrt(train_mse)
train_rmse
#0.51670.5028506982006457
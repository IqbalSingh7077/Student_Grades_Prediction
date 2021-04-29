
"""
Created on Fri Apr 23 23:49:15 2021

@author: ryuzaki
"""

import os 
os.chdir('/home/ryuzaki/E drive/Dekstop/projects/kaggle/Student_grades')

# importing the important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
#pip install pandas-bokeh
import pandas_bokeh
#importing the dataset

df=pd.read_csv('student-mat.csv')

#Structure of the dataset
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
df.head()
df.isnull().sum()#No missing values
df.shape#(395,33)
df.dtypes
df.describe()


##******* EDA ********##

plt.style.use('fivethirtyeight')

# G1, G2, G3 & age (distribution of continous variables)

plt.figure(figsize = (20,10))
plt.subplot(3,2,1)
sns.distplot(df.G1)
plt.subplot(3,2,2)
sns.distplot(df.G2)
plt.subplot(3,2,3)
sns.distplot(df.G3) 
plt.subplot(3,2,4)
sns.distplot(df.age) 
plt.subplot(3,2,5)
sns.distplot(df.absences)
# boxplot of continous variables

cont = ['age','G1','G2','G3','absences']
plt.figure(figsize = (20,10))
plt.title('Boxplot of Continous Vars')
for i in enumerate(cont):
    plt.subplot(3,2,i[0]+1)
    sns.boxplot(x=i[1], data=df)
'''
age and G2 have one outlier each, but these outliers can be ignored
absences has so many outliers, which will treated further
'''

# Countplots of ordinal and categorical variables
df.columns
cat = ['school', 'sex','address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health']

plt.figure(figsize = (40,20))
plt.title('Boxplot of Continous Vars')
for i in enumerate(cat):
    plt.subplot(4,7,i[0]+1)
    sns.countplot(x=i[1], data=df)




# 1.school
df.school.value_counts()
'''
GP    349
MS     46
Name: school, dtype: int

will reject this variable as data is more dominant towards GP with 89 %
'''

# 2. sex
df.sex.value_counts()
'''
F    208
M    187
Name: sex, dtype: int6

'''
sns.kdeplot(df.loc[df['sex'] == 'F', 'G3'], label='Female', shade = True)
sns.kdeplot(df.loc[df['sex'] == 'M', 'G3'], label='Male', shade = True)
plt.title('Grades distribution as per gender', fontsize = 20)
plt.legend()
plt.show()

'''
gender doesn't seem to have much impact on the final scores'
'''

# t_test
S_F= df[df.sex== 'F']
S_M= df[df.sex== 'M']

len(S_F)
len(S_M)


statistics, p=scipy.stats.ttest_ind(S_F.G3, S_M.G3)
p
'''
p-value = 0.039865332341527636, can be an important predictor
'''


# 3. age 
df.age.describe()
'''
count    395.000000
mean      16.696203
std        1.276043
min       15.000000
25%       16.000000
50%       17.000000
75%       18.000000
max       22.000000
Name: age, dtype: float64
'''
# correlation 
cor=df[['age','G3']].corr()
'''
          age        G3
age  1.000000 -0.161579
G3  -0.161579  1.000000
'''
sns.regplot(x='age', y='G3', data=df)
plt.title('Scatter plot of age & G3')
plt.xlabel('Age')
plt.ylabel('Final Grade')

'''
may not be a good predictor as correlation is -0.161579, close to 0, rejecting this variable
'''



# 4. address

df.address.value_counts()
'''
U    307
R     88
Name: address, dtype: int64

rejecting the variable as the data is more dominant towards U with over 78%
'''

sns.kdeplot(df.loc[df['address']=='U','G3'], label = 'Urban', shade = True)
sns.kdeplot(df.loc[df['address']=='R','G3'], label = 'Rural', shade = True)
plt.title("Impact of home address type on grades")
plt.legend()
plt.plot()

'''
address type doesn't seem to have a large impact on grades but students from urban areas scored
more between the range of 12 to 16 than.
'''

# 5. famsize
df.famsize.value_counts()
'''
GT3    281
LE3    114
Name: famsize, dtype: int64
'''
sns.boxplot(x="G3", y='famsize', data=df)
sns.swarmplot(x="G3",y='famsize',data=df,orient='h')

# t-test
G_3=df[df['famsize']=='GT3']
L_3=df[df['famsize']=='LE3']
len(G_3)
len(L_3)

statistics, p=scipy.stats.ttest_ind(G_3.G3, L_3.G3)
p
'''
p-value = 0.1062048278385956, might not be a good predictor, rejecting the var
'''




# 6. Pstatus

df.Pstatus.value_counts()
'''
T    354
A     41
Name: Pstatus, dtype: int64

Rejecting this variable as data is more dominant towards T by 90%
'''


# 7. Medu
df.Medu.value_counts()
'''
4    131
2    103
3     99
1     59
0      3
Name: Medu, dtype: int64
'''
mylabels = ['Higher Education','5th to 9th grade','secondary education','primary education','None']
col = [ "#ffffff","#d0e1f9","#4d648d","#283655","#1e1f26"]

# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


plt.figure(figsize=(15,12))
plt.pie(df.Medu.value_counts(),autopct = lambda pct: func(pct, df.Medu),explode =(0.1,0,0,0,0) 
        ,shadow= True, colors = col)
plt.legend(mylabels)
plt.title('Pie chart of Mothers education level ')
plt.show() 
'''
from the pie chart we can see that around 33% of the education level is higher education 
'''

#Anova test
import pingouin as pg
pg.anova(dv='G3',between='Medu',data=df)

'''
p-value = 0.000092, Accepting the variable.
'''


# 8.Fedu 
df.Fedu.value_counts()
'''
2    115
3    100
4     96
1     82
0      2
Name: Fedu, dtype: int64
'''
mylabels = ['5th to 9th grade','secondary education','Higher Education','primary education','None']

plt.figure(figsize=(15,12))
plt.pie(df.Fedu.value_counts(),autopct = lambda pct: func(pct, df.Fedu),explode = (0.1,0,0,0,0) 
        ,shadow= True, colors = col)
plt.legend(mylabels)
plt.title('Pie chart of Fathers education level ')
plt.show() 

'''
from the pie chart we can see that 29.1% of the fathers studied till 5th to 9th grade & only
24.3% have gone for higher education
'''
#Anova test
pg.anova(dv='G3',between='Fedu',data=df)

'''
p-value = 0.022197, Accepting the variable.
'''


# 9. Mjob
df.Mjob.value_counts()
'''
other       141
services    103
at_home      59
teacher      58
health       34
Name: Mjob, dtype: int64

86% of the mothers are working professionals out of which, 26% of the mothers are into services
14% into Teaching & 8% into health care.
35% of the mothers are into other professions and only 14% are house wives.
'''

# Anova Test
#pip install pingouin
import pingouin as pg
aov = pg.anova(dv = 'G3', between = 'Mjob', data=df, detailed = False)
aov
'''
p-value = 0.005195, can be an important variable 
'''


# 9. Fjob
df.Fjob.value_counts()
'''
other       217
services    111
teacher      29
at_home      20
health       18
Name: Fjob, dtype: int64

around 95% fathers are working professionals out of which, 28% are into services, 7% into teaching
& 4% into health care.
54% are into other professions, and only 5% are house husbands.
'''

#anova test
aov = pg.anova(dv='G3', between = 'Fjob', data=df,detailed= False)
aov

'''
p-value = 0.268314, rejecting the variable as p-value > 0.05
'''


#11. reason
df.reason.value_counts()
grop_df = df[['reason','G3']]
grouped = grop_df.groupby(['reason']).mean()
grouped.plot(kind = 'line')
'''
from the line plot we can assume that schools reputation plays a good role in scoring good marks,
althoug more people prefer school according to thier preference in a course.
'''

# Anova test
aov = pg.anova(dv='G3',between ='reason',data=df)
aov
'''
p-value = 0.102337, rejecting the variable as p > 0.05.
'''



# 12. guardian
df.guardian.value_counts()
'''
mother    273
father     90
other      32
Name: guardian, dtype: int64

most of the gurdians are mothers
'''
sns.boxplot(x='G3',y='guardian',data=df)

# Anova test
aov = pg.anova(dv='G3', between = 'guardian', data = df)
aov
'''
p-value = 0.205133, rejecting the variable 
'''


# 13. traveltime
df.traveltime.value_counts()
'''
1    257
2    107
3     23
4      8
Name: traveltime, dtype: int64
'''

group_df = df[['G3','traveltime']]
grouped = group_df.groupby(['traveltime']).mean()
grouped.plot(kind = 'line')
'''
from line plot we can observe that average final scores are reducing with increase in travel time
                   G3
traveltime           
1           10.782101
2            9.906542
3            9.260870
4            8.750000
'''

#Anova test
pg.anova(dv='G3', between = 'traveltime', data=df)
'''
p-value = 0.139379, Rejecting the variable!
'''

# 14. studytime
df.studytime.value_counts()
'''
2    198
1    105
3     65
4     27
Name: studytime, dtype: int64
'''
group_df = df[['G3','studytime']]
grouped = group_df.groupby(['studytime']).mean()
grouped.plot(kind ='line')
'''
from the line plot and grouped data we can see that there is no sharp increase in the average
final scores.
'''

#Anova test
pg.anova(dv='G3', between = 'studytime', data=df)
'''
p-value = 0.160723, Rejecting the variable!
'''



# 15. failures
df.failures.value_counts()
'''
0    312
1     50
2     17
3     16
Name: failures, dtype: int64
'''
group=df.groupby('failures').mean()
group[['studytime','G3']]
'''
          studytime         G3
failures                      
0          2.102564  11.253205
1          1.860000   8.120000
2          1.882353   6.235294
3          1.437500   5.687500

from the above table we can see that, average study time for 0 faliuers in around 2 hours, whereas
it is 1 hour 43 min for 3 failures. even the average scores reduced drastically.
'''
# Anova test
aov = pg.anova(dv='G3', between = 'failures', data = df)
aov
'''
can be an important predictor as p-value is 1.642166e-12, ehich is < 0.05
'''


# 16. schoolsup
df.schoolsup.value_counts()
'''
no     344
yes     51
Name: schoolsup, dtype: int64

rejecting the variable as data is more dominant towards 'no' by 88%.
'''


# 17. famsup
df.famsup.value_counts()
'''
yes    242
no     153
Name: famsup, dtype: int64
'''
group_df = df[['famsup','G3']]
grouped = group_df.groupby(['famsup']).mean()
grouped
'''
from the table we can see that, family educational support makes little to no difference.
'''
#t-test
NO = df[df['famsup']=='no']
YES = df[df['famsup']=='yes']
statistics, p=scipy.stats.ttest_ind(NO.G3, YES.G3)
p
'''
p-value = 0.43771108589489893, rejecting the variable as p-value > 0.05
'''


# 18. paid
df.paid.value_counts()
'''
no     214
yes    181
Name: paid, dtype: int64
'''
plt.figure(figsize=(12,8))
sns.kdeplot(df.loc[df['paid']=='yes','G3'], label = 'Yes', shade=True)
sns.kdeplot(df.loc[df['paid']=='no','G3'], label= 'No', shade=True)
plt.legend()
plt.title('Does you extra paid classes affect final scores?')

'''
from the plot we can conclude that, extra paid classes does have an impact on final scores, as
more students were able to score between the range of 7 to 15, those who took extra classes.
'''

# t-test
NO = df[df['paid'] == 'no']
YES = df[df.paid == 'yes']

stats, p = scipy.stats.ttest_ind(NO.G3, YES.G3)
p
'''
p-value = 0.04276506403357553, can be an important predictor
'''


# 19. activities
df.activities.value_counts()
'''
yes    201
no     194
Name: activities, dtype: int6
'''
plt.figure(figsize = (12,8))
sns.kdeplot(df.loc[df['activities']== 'yes','G3'], label = 'Yes', shade = True)
sns.kdeplot(df.loc[df['activities']== 'no', 'G3'], label = 'no', shade = True)
plt.legend()
plt.title('Does extra-curricular activities impact final scores?')

'''
from the plot we can observe that extra-curricular activities doesnt affectfinal scores much.
'''

# t-test
NO = df[df.activities == 'no']
YES = df[df.activities == 'yes']

stats, p = scipy.stats.ttest_ind(NO.G3, YES.G3)
p
'''
p-value = 0.7497402737748432, rejecting tha variable
'''

# 20. nursery
df.nursery.value_counts()
'''
yes    314
no      81
Name: nursery, dtype: int64

rejecting the variable as data is more dominant towards yes by 80%.
'''


# 21. higher
df.higher.value_counts()
'''
yes    375
no      20
Name: higher, dtype: int64

Rejecting the variable as data is more dominant towards yes by 95%.
'''

df[['traveltime','studytime','failures','G3']].groupby(df['higher']).mean()
'''
interesting observation, those who wants to go for higher education, 
1- less traveling time around 1.43 on average
2- more study time, 2.06 on average
3- more average final scores, 10.608
4- less failures, 0.28 on average
when compared to those who dont want to pursue higher education.
'''

# 22. internet
df.internet.value_counts()
'''
yes    329
no      66
Name: internet, dtype: int64

Rejcting the variable as data is more dominant towards yes by 84%.
'''

# 23. romantic
df.romantic.value_counts()
'''
no     263
yes    132
Name: romantic, dtype: int64
'''

plt.figure(figsize = (12,8))
sns.kdeplot(df.loc[df['romantic']=='yes','G3'], label = 'Yes', shade = True)
sns.kdeplot(df.loc[df['romantic']=='no', 'G3'], label = 'No', shade = True)
plt.legend()
plt.title('Does romantic relationship affect final scores?')

df.groupby(df.romantic).mean()
'''
both from the table and the plot we can observe that romantic relation doesnt have much impact
on the final scores of students.
'''
# t-test
YES = df[df.romantic == 'yes']
NO = df[df.romantic == 'no']

stats, p = scipy.stats.ttest_ind(YES.G3, NO.G3)
p
'''
p- values = 0.00971272639411926, can be an important predictor
'''


# 24. famrel 
df.famrel.value_counts()
'''
4    195
5    106
3     68
2     18
1      8
Name: famrel, dtype: int64
'''
sns.boxplot(x="famrel", y="G3", data = df)

#Anova test
pg.anova(dv='G3', between = 'famrel', data=df)
'''
p-value = 0.810487, rejecting the variable
'''


# 25. freetime
df.freetime.value_counts()
'''
3    157
4    115
2     64
5     40
1     19
Name: freetime, dtype: int64
'''

plt.plot(df[['G3','failures','studytime']].groupby(df['freetime']).mean())
plt.legend()

sns.boxplot(x='freetime', y='G3', data=df)
'''
from the plots, we can assume that free time doesnt affect grades much but a lil bit
'''
#Anova test
pg.anova(dv='G3', between = 'freetime', data=df)
'''
p-value = 0.065744, rejecting the variable
'''

# 26. goout
df.goout.value_counts()
'''
3    130
2    103
4     86
5     53
1     23
Name: goout, dtype: int64
'''

sns.boxplot(x='goout', y='G3', data=df)
df[['G3', 'studytime']].groupby(df['goout']).mean()
'''
from the plot above we can assume that people those who go out with friends quite often, score
little bit less than those who dont, and even the study time differs a bit
'''

# Anova test
pg.anova(dv='G3',between = 'goout', data=df)

'''
p-value = 0.01438, accepting the variable
'''


# 27. Dalc
df.Dalc.value_counts()
'''
1    276
2     75
3     26
5      9
4      9
Name: Dalc, dtype: int64
'''
sns.boxplot(x='Dalc', y='G3', data=df)
df[['G3']].groupby(df['Dalc']).mean()
'''
from the plot we can assume that workday alcohol consumption doesnt affect final grades.
'''
#Anova test
pg.anova(dv='G3', between='Dalc', data=df)
'''
p-value = 0.177864, Rejecting the variable
'''


# 28. Walc
df.Walc.value_counts()
'''
1    151
2     85
3     80
4     51
5     28
Name: Walc, dtype: int6
'''
sns.boxplot(x=df.Walc, y=df.G3)
df[['G3']].groupby(df['Walc']).mean()

#Anova test
pg.anova(dv='G3', between = 'Walc', data=df)
'''
p-value = 0.56975, rejecting the variable 
'''


# 29. health
df.health.value_counts()
'''
5    146
3     91
4     66
1     47
2     45
Name: health, dtype: int64
'''

sns.boxplot(x=df.health, y=df.G3)
df[['G3','studytime']].groupby(df['health']).mean()
'''
from the plot and table we can assume that, health doesnt affect final scores and study time much
'''
#Anova test
pg.anova(dv='G3', between = 'health', data=df)
'''
p-value = 0.211085, rejecting the variable!
'''


# 30. absences
df.absences.describe()
df.absences.value_counts()
sns.boxplot(df.absences)
df.absences.plot(kind = 'hist')

mean = np.mean(df.absences) 
std = np.std(df.absences) 
print('Mean of the absences :', mean) 
print('STD Deviation of absences:', std)
threshold = 3
outlier = [] 
for i in df.absences: 
    z = (i-mean)/std 
    if z > threshold: 
        outlier.append(i) 
print('Outlier in the dataset is (absences):', outlier)
print((df['absences']>= 30).sum())

'''
we will every data point over 30 as a outlier and treat it
'''
df["absences"] = np.where(df["absences"] >30, 30,df['absences'])
print(df['absences'].skew())
df.absences.describe()
sns.boxplot(df.absences)


# 31 G1 & G2
df.G1.describe()
df.G2.describe()


# correlation test for G1, G2 & absences
cor = df[['G1','G2','absences','G3']].corr()

plt.figure(figsize=(12,8))
sns.heatmap(cor, cmap = 'YlGnBu')
plt.title("correlation heatmap")

'''
from the heatmap we can see that G1 & G2 are highly correlated with G3, whereas absences doesn't 
show any strong relation.
G1 & G2 are highly correlated with each other hence we will use one at a time to see how our model 
performs.
'''


## ************** MODELING ************##

df1 = df.copy()

# choosing the accepted variables
df1 = df1.loc[:,['sex','Medu','Fedu','Mjob','failures','paid','romantic','goout','G1','G2','G3']]
df1.head()


# Encoding
dumies = pd.get_dummies(df1[['sex','Mjob','paid','romantic']])
dumies.head(3)

df1 = df1.drop(columns=['sex','Mjob','paid','romantic'])

df1 = pd.concat([df1,dumies], axis=1)
df1.head()


# Scaling 

from sklearn.preprocessing import StandardScaler
std_scale= StandardScaler()
df1['G2']=std_scale.fit_transform(df1[['G2']])
df1['G1']=std_scale.fit_transform(df1[['G1']])


# Train Test split

Y = df1['G3']
X = df1.drop(['G3'], axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Model 1
# Linear Regression
from sklearn.linear_model import LinearRegression 
lm = LinearRegression() 

lm.fit(X_train,y_train) 
print(lm.intercept_)
predictions_train = lm.predict(X_train)
predictions = lm.predict(X_test) 

from sklearn import metrics

# Train set
print('MAE:', metrics.mean_absolute_error(y_train, predictions_train))
print('MSE:', metrics.mean_squared_error(y_train, predictions_train))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, predictions_train)))
'''
MAE: 1.1425841204128337
MSE: 3.3853761863057805
RMSE: 1.8399391800561726
'''
# Test set
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
'''
MAE: 1.363713935142439
MSE: 4.913741707965253
RMSE: 2.216696124407956
'''
from sklearn.metrics import r2_score
# Train set
print(r2_score(y_train, predictions_train))
'''
0.838826274068298
'''
# Test set
print(r2_score(y_test, predictions))

'''
0.7603642828164666
'''

sns.residplot(y_test, predictions)

sns.kdeplot(predictions, label = 'pred', shade = True)
sns.kdeplot(y_test, label = 'actual', shade = True)


# Ridge Regression
from sklearn.linear_model import Ridge

# Model 1


r2_s = []

al = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

for a in al:
    RM = Ridge( alpha = a)
    RM.fit (X_train, y_train)
    y_hat = RM.predict(X_test)
    r2_s.append(r2_score(y_hat, y_test))
    
r2_s

RM = Ridge( alpha = 0.0001)
RM.fit (X_train, y_train)

y_hat_train = RM.predict(X_train)
y_hat = RM.predict(X_test)

print("Train Data accuracy:", r2_score(y_train, y_hat_train))
print("Test Data accuracy:", r2_score(y_hat, y_test))
'''
Train Data accuracy: 0.8388262740681147
Test Data accuracy: 0.7476635905442173
'''


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
random_regressor = RandomForestRegressor(n_estimators=10, random_state=123)
random_regressor.fit(X_train, y_train)
y_pred = random_regressor.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

'''
0.693740115025161
'''


'''
Out of Random Forest and linear regression, Linear regression performed the best with r2 of 0.76
'''





# Using Back-elimination method

df2 = df.copy()

# importing ols
from statsmodels.formula.api import ols

fit = ols('''G3 ~ school+sex+age+address+famsize+Pstatus+Medu+Fedu+
          Mjob+Fjob+reason+guardian+traveltime+studytime+
       failures+schoolsup+famsup+paid+activities+nursery+
       higher+internet+romantic+famrel+freetime+goout+Dalc+
       Walc+health+absences+G1+G2''', data=df2).fit()
       
fit.summary()

fit1 = ols('''G3 ~ famrel+absences+G1+G2''', data=df2).fit()
       
fit1.summary()

df3 = df2.loc[:,['famrel','absences','G1','G2','G3']]

# Scaling 

from sklearn.preprocessing import StandardScaler
std_scale= StandardScaler()
df3['G2']=std_scale.fit_transform(df3[['G2']])
df3['G1']=std_scale.fit_transform(df3[['G1']])
df3['absences']=std_scale.fit_transform(df3[['absences']])

df3.head()

# Train Test split
Y = df3['G3']
X = df3.drop(['G3'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model 2
# Linear Regression
from sklearn.linear_model import LinearRegression 
lm = LinearRegression() 

lm.fit(X_train,y_train) 
print(lm.intercept_)
predictions_train = lm.predict(X_train)
predictions = lm.predict(X_test) 

from sklearn import metrics

# Train set
print('MAE:', metrics.mean_absolute_error(y_train, predictions_train))
print('MSE:', metrics.mean_squared_error(y_train, predictions_train))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, predictions_train)))
'''
MAE: 1.1273622394206033
MSE: 3.396247854981722
RMSE: 1.842891167427345
'''
# Test set
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
'''
MAE: 1.303553592892538
MSE: 4.163105697228046
RMSE: 2.0403690100636322
'''
from sklearn.metrics import r2_score
# Train set
print(r2_score(y_train, predictions_train))
'''
0.8383086868782299
'''
# Test set
print(r2_score(y_test, predictions))

'''
0.796971660547618
'''


# Ridge Regression
from sklearn.linear_model import Ridge

r2_s = []

al = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

for a in al:
    RM = Ridge( alpha = a)
    RM.fit (X_train, y_train)
    y_hat = RM.predict(X_test)
    r2_s.append(r2_score(y_hat, y_test))
    
r2_s

RM = Ridge( alpha = 0.0001)
RM.fit (X_train, y_train)

y_hat_train = RM.predict(X_train)
y_hat = RM.predict(X_test)

print("Train Data accuracy:", r2_score(y_train, y_hat_train))
print("Test Data accuracy:", r2_score(y_hat, y_test))
'''
Train Data accuracy: 0.8419460358318387
Test Data accuracy: 0.7680180745339673
'''


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
random_regressor = RandomForestRegressor(n_estimators=10, random_state=123)
random_regressor.fit(X_train, y_train)
y_pred = random_regressor.predict(X_test)
y_pred_train = random_regressor.predict(X_train)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
print(r2_score(y_train, y_pred_train))
'''
0.9635098291965452 - accuracy on Train Dataset 
0.842015957787641 - Accuracy on Test Dataset

it shows that there is no overfitting
'''

'''
In back-elimination method, Random Forest clearly gave us the better accuracy
'''



# Model 3
# Variables from both Model1 & Model2
'''
For this model we will choose all the variables from model1 & model2 and combine them!
we will compare the accuracy of our models
'''

df4 = df.loc[:,['sex','Medu','Fedu','Mjob','failures','paid','famrel','absences','romantic','goout','G1','G2','G3']]


# Encoding
dumies = pd.get_dummies(df4[['sex','Mjob','paid','romantic']])
dumies.head(3)

df4 = df4.drop(columns=['sex','Mjob','paid','romantic'])

df4 = pd.concat([df4,dumies], axis=1)
df4.head()


# Scaling 

from sklearn.preprocessing import StandardScaler
std_scale= StandardScaler()
df4['G2']=std_scale.fit_transform(df4[['G2']])
df4['G1']=std_scale.fit_transform(df4[['G1']])
df4['absences']=std_scale.fit_transform(df4[['absences']])

# Train Test split

Y = df4['G3']
X = df4.drop(['G3'], axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Model 3
# Linear Regression
from sklearn.linear_model import LinearRegression 
lm = LinearRegression() 

lm.fit(X_train,y_train) 
print(lm.intercept_)
predictions_train = lm.predict(X_train)
predictions = lm.predict(X_test) 

from sklearn import metrics

# Train set
print('MAE:', metrics.mean_absolute_error(y_train, predictions_train))
print('MSE:', metrics.mean_squared_error(y_train, predictions_train))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, predictions_train)))
'''
MAE: 1.1327881131744604
MSE: 3.180939002151406
RMSE: 1.7835187137093365
'''
# Test set
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
'''
MAE: 1.3931023659036532
MSE: 4.859666430803195
RMSE: 2.204465112176465
'''
from sklearn.metrics import r2_score
# Train set
print(r2_score(y_train, predictions_train))
'''
0.8523379330026574
'''
# Test set
print(r2_score(y_test, predictions))

'''
0.7630476526224806
'''

sns.residplot(y_test, predictions)

sns.kdeplot(predictions, label = 'pred', shade = True)
sns.kdeplot(y_test, label = 'actual', shade = True)


# Ridge Regression
from sklearn.linear_model import Ridge

r2_s = []

al = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

for a in al:
    RM = Ridge( alpha = a)
    RM.fit (X_train, y_train)
    y_hat = RM.predict(X_test)
    r2_s.append(r2_score(y_hat, y_test))
    
r2_s

RM = Ridge( alpha = 0.0001)
RM.fit (X_train, y_train)

y_hat_train = RM.predict(X_train)
y_hat = RM.predict(X_test)

print("Train Data accuracy:", r2_score(y_train, y_hat_train))
print("Test Data accuracy:", r2_score(y_hat, y_test))
'''
Train Data accuracy: 0.852337933002479
Test Data accuracy: 0.7438754933202991
'''

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
random_regressor = RandomForestRegressor(n_estimators=10, random_state=123)
random_regressor.fit(X_train, y_train)
y_pred = random_regressor.predict(X_test)
y_pred_train = random_regressor.predict(X_train)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
print(r2_score(y_train, y_pred_train))
'''
0.9810182475957447 - Train data accuracy
0.8002282530553558-  Test data accuracy
'''


'''
Out of Random Forest and linear regression, Linear regression performed the best with r2 of 0.76
'''


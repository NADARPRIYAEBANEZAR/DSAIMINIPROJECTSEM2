# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 21:37:52 2021

@author: Priya
"""
# hides all warnings
import warnings
warnings.filterwarnings('ignore')

# imports
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
# seaborn
import seaborn as sns
# utils
import utils
from sklearn import preprocessing

##############################################################
# Read Data 
##############################################################

# read dataset
df = pd.read_csv('./data/student-mat.csv')


##############################################################
# Exploratory Data Analytics
##############################################################

# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())


##############################################################
# Dependent Variable 
##############################################################

# store dep variable  
# change as required
depVars = 'G3'
print("\n*** Dep Vars ***")
print(depVars)


##############################################################
# Data Transformation
##############################################################


print(df['school'].unique())
lesh = preprocessing.LabelEncoder()
df['school'] = lesh.fit_transform(df['school'])
print(df['school'].unique())
print("Done ...")


print(df['sex'].unique())
lesex = preprocessing.LabelEncoder()
df['sex'] = lesex.fit_transform(df['sex'])
print(df['sex'].unique())
print("Done ...")

print(df['address'].unique())
lead = preprocessing.LabelEncoder()
df['address'] = lead.fit_transform(df['address'])
print(df['address'].unique())
print("Done ...")

print(df['famsize'].unique())
lefam = preprocessing.LabelEncoder()
df['famsize'] = lefam.fit_transform(df['famsize'])
print(df['famsize'].unique())
print("Done ...")

print(df['Pstatus'].unique())
lep = preprocessing.LabelEncoder()
df['Pstatus'] = lep.fit_transform(df['Pstatus'])
print(df['Pstatus'].unique())
print("Done ...")

print(df['Mjob'].unique())
leMjob= preprocessing.LabelEncoder()
df['Mjob'] = leMjob.fit_transform(df['Mjob'])
print(df['Mjob'].unique())
print("Done ...")

print(df['Fjob'].unique())
lef = preprocessing.LabelEncoder()
df['Fjob'] = lef.fit_transform(df['Fjob'])
print(df['Fjob'].unique())
print("Done ...")

print(df['reason'].unique())
leRes = preprocessing.LabelEncoder()
df['reason'] = leRes.fit_transform(df['reason'])
print(df['reason'].unique())
print("Done ...")

print(df['guardian'].unique())
leguar = preprocessing.LabelEncoder()
df['guardian'] = leguar.fit_transform(df['guardian'])
print(df['guardian'].unique())
print("Done ...")

print(df['schoolsup'].unique())
leschool = preprocessing.LabelEncoder()
df['schoolsup'] = leschool.fit_transform(df['schoolsup'])
print(df['schoolsup'].unique())
print("Done ...")

print(df['famsup'].unique())
lefamsup = preprocessing.LabelEncoder()
df['famsup'] = lefamsup.fit_transform(df['famsup'])
print(df['famsup'].unique())
print("Done ...")

print(df['paid'].unique())
lepaid = preprocessing.LabelEncoder()
df['paid'] = lepaid.fit_transform(df['paid'])
print(df['paid'].unique())
print("Done ...")

print(df['activities'].unique())
leact = preprocessing.LabelEncoder()
df['activities'] = leact.fit_transform(df['activities'])
print(df['activities'].unique())
print("Done ...")

print(df['nursery'].unique())
lenur = preprocessing.LabelEncoder()
df['nursery'] = lenur.fit_transform(df['nursery'])
print(df['nursery'].unique())
print("Done ...")

print(df['higher'].unique())
lehig = preprocessing.LabelEncoder()
df['higher'] = lehig.fit_transform(df['higher'])
print(df['higher'].unique())
print("Done ...")

print(df['internet'].unique())
leint = preprocessing.LabelEncoder()
df['internet'] = leint.fit_transform(df['internet'])
print(df['internet'].unique())
print("Done ...")

print(df['romantic'].unique())

lerom = preprocessing.LabelEncoder()
df['romantic'] = lerom.fit_transform(df['romantic'])
print(df['romantic'].unique())
print("Done ...")

# Recheck info
print("\n*** Structure ***")
print(df.info())

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outlier
colNames = df.columns.tolist()
for colName in colNames:
      colType =  df[colName].dtype  
      df[colName] = utils.HandleOutliers(df[colName])
      if df[colName].isnull().sum() > 0:
          df[colName] = df[colName].astype(np.float64)
      else:
          df[colName] = df[colName].astype(colType)    
print("Done ...")  
    
# Recheck outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# Recheck outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required
print("None")

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# check mean
print('\n*** Mean In Columns ***')
print(df.mean())

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required
print('\n*** Handle Nulls ***')
print("None")

# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.2f}'.format
dfc = df.corr()
print("Done ...")

# handle multi colinearity if required
# drop col not required

##############################################################
# Visual Data Analytics
##############################################################

# check relation with corelation - heatmap
print("\n*** Heat Map ***")
plt.figure(figsize=(8,8))
ax = sns.heatmap(df.corr(), annot=True, cmap="PiYG")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()

# boxplot
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
for colName in colNames:
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# histograms
# https://www.qimacros.com/histogram-excel/how-to-determine-histogram-bin-interval/
# plot histograms
print('\n*** Histograms ***')
colNames = df.columns.tolist()
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# scatterplots
# plot Sscatterplot
print('\n*** Scatterplot ***')
colNames = df.columns.tolist()
colNames.remove(depVars)
print(colName)
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.regplot(data=df, x=depVars, y=colName, color= 'b', scatter_kws={"s": 5})
    plt.title(depVars + ' v/s ' + colName)
    plt.show()

# class count plot
# change as required
colNames = ['sex','age','famsize','Medu','Fedu','Mjob','Fjob','reason','traveltime','studytime','famsup','paid','activities','romantic','famrel','freetime','goout','Dalc','Walc','health']
print("\n*** Distribution Plot ***")
for colName in colNames:
    plt.figure()
    sns.countplot(df[colName],label="Count")
    plt.title(colName)
    plt.show()

###############################
# Split Train & Test
###############################

# split into data & target
print("\n*** Prepare Data ***")
dfTrain = df.sample(frac=0.8, random_state=707)
dfTest=df.drop(dfTrain.index)
print("Train Count:",len(dfTrain.index))
print("Test Count :",len(dfTest.index))

##############################################################
# Model Creation & Fitting 
##############################################################

# all cols except dep var 
print("\n*** Regression Data ***")
allCols = dfTrain.columns.tolist()
print(allCols)
allCols.remove(depVars)
print(allCols)
print("Done ...")

# regression summary for feature
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(dfTrain[allCols])
y = dfTrain[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# *** Regression Summary ***
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                     G3   R-squared:                       0.846
# Model:                            OLS   Adj. R-squared:                  0.834
# Method:                 Least Squares   F-statistic:                     69.87
# Date:                Wed, 14 Jul 2021   Prob (F-statistic):          2.24e-104
# Time:                        23:51:29   Log-Likelihood:                -637.31
# No. Observations:                 316   AIC:                             1323.
# Df Residuals:                     292   BIC:                             1413.
# Df Model:                          23                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# school     -6.673e-17   1.44e-15     -0.046      0.963   -2.89e-15    2.76e-15
# sex           -0.0071      0.252     -0.028      0.978      -0.502       0.488
# age           -0.2111      0.090     -2.336      0.020      -0.389      -0.033
# address       -0.0728      0.295     -0.247      0.805      -0.654       0.508
# famsize        0.1110      0.242      0.458      0.647      -0.366       0.588
# Pstatus       -0.0728      0.295     -0.247      0.805      -0.654       0.508
# Medu           0.0268      0.144      0.186      0.853      -0.257       0.311
# Fedu          -0.1065      0.133     -0.800      0.425      -0.369       0.156
# Mjob          -0.0428      0.104     -0.412      0.681      -0.247       0.162
# Fjob          -0.0275      0.129     -0.213      0.832      -0.282       0.227
# reason         0.0704      0.093      0.761      0.447      -0.112       0.253
# guardian      -0.0728      0.295     -0.247      0.805      -0.654       0.508
# traveltime     0.0520      0.167      0.312      0.755      -0.276       0.380
# studytime     -0.1724      0.150     -1.149      0.251      -0.467       0.123
# failures    4.497e-17   6.23e-17      0.722      0.471   -7.76e-17    1.68e-16
# schoolsup   1.523e-17   9.03e-17      0.169      0.866   -1.62e-16    1.93e-16
# famsup         0.1063      0.240      0.442      0.658      -0.366       0.579
# paid           0.2361      0.235      1.003      0.317      -0.227       0.699
# activities    -0.2206      0.223     -0.991      0.322      -0.659       0.217
# nursery       -0.0728      0.295     -0.247      0.805      -0.654       0.508
# higher        -0.0728      0.295     -0.247      0.805      -0.654       0.508
# internet      -0.0728      0.295     -0.247      0.805      -0.654       0.508
# romantic      -0.0202      0.240     -0.084      0.933      -0.492       0.451
# famrel         0.3572      0.121      2.953      0.003       0.119       0.595
# freetime      -0.0342      0.118     -0.289      0.773      -0.267       0.199
# goout          0.0917      0.113      0.810      0.419      -0.131       0.314
# Dalc          -0.0383      0.162     -0.237      0.813      -0.357       0.280
# Walc           0.0387      0.124      0.312      0.756      -0.206       0.283
# health         0.0958      0.082      1.172      0.242      -0.065       0.257
# absences       0.0606      0.017      3.595      0.000       0.027       0.094
# G1             0.1975      0.064      3.078      0.002       0.071       0.324
# G2             0.9671      0.056     17.267      0.000       0.857       1.077
# ==============================================================================
# Omnibus:                      146.617   Durbin-Watson:                   2.124
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              629.945
# Skew:                          -1.995   Prob(JB):                    1.62e-137
# Kurtosis:                       8.650   Cond. No.                     1.08e+16
# ==============================================================================

#when drop columns based on pValue After removing the columns based on pValues out 0f 33 columns only 5 columns remain
#Since there is noscope of getting more data the business owner has decided that columns based on pValues will not be dropped

# train data
print("\n*** Regression Data For Train ***")
X_train = dfTrain[allCols].values
y_train = dfTrain[depVars].values
# print
print(X_train.shape)
print(y_train.shape)
print(type(X_train))
print(type(y_train))
print("Done ...")

# test data
print("\n*** Regression Data For Test ***")
X_test = dfTest[allCols].values
y_test = dfTest[depVars].values
print(X_test.shape)
print(y_test.shape)
print(type(X_test))
print(type(y_test))
print("Done ...")


###############################
# Auto Select Best Regression
###############################

# imports 
print("\n*** Import Regression Libraries ***")
# normal linear regression
from sklearn.linear_model import LinearRegression 
# ridge regression from sklearn library 
from sklearn.linear_model import Ridge 
# import Lasso regression from sklearn library 
from sklearn.linear_model import Lasso 
# import model 
from sklearn.linear_model import ElasticNet 
print("Done ...")
  
# empty lists
print("\n*** Init Empty Lists ***")
lModels = []
lModelAdjR2 = []
lModelRmses = []
lModelScInd = []
print("Done ...")

# list model name list
print("\n*** Init Models Lists ***")
lModels.append(("LinearRegression", LinearRegression()))
lModels.append(("RidgeRegression ", Ridge(alpha = 10)))
lModels.append(("LassoRegression ", Lasso(alpha = 1)))
lModels.append(("ElasticNet      ", ElasticNet(alpha = 1)))
print("Done ...")

# imports
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# iterate through the models list
for vModelName, oModelObject in lModels:
    # create model object
    model = oModelObject
    # print model vals
    print("\n*** "+vModelName)
    # fit or train the model
    model.fit(X_train, y_train) 
    # predict train set 
    p_train = model.predict(X_train)
    dfTrain[vModelName] = p_train
    # predict test set 
    p_test = model.predict(X_test)
    dfTest[vModelName] = p_test
    # r-square  
    r2 = r2_score(y_train, p_train)
    print("R-Square:",r2)
    # adj r-square  
    adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
              (X_train.shape[0] - X_train.shape[1] - 1)))
    lModelAdjR2.append(adj_r2)
    print("Adj R-Square:",adj_r2)
    # mae 
    mae = mean_absolute_error(y_test, p_test)
    print("MAE:",mae)
    # mse 
    mse = mean_squared_error(y_test, p_test)
    print("MSE:",mse)
    # rmse 
    rmse = np.sqrt(mse)
    lModelRmses.append(rmse)
    print("RMSE:",rmse)
    # scatter index
    si = rmse/y_test.mean()
    lModelScInd.append(si)
    print("SI:",si)

# print key metrics for each model
print("\n*** Model Summary ***")
msg = "%10s %16s %10s %10s" % ("Model Type", "AdjR2", "RMSE", "SI")
print(msg)
for i in range(0,len(lModels)):
    # print accuracy mean & std
    msg = "%16s %10.3f %10.3f %10.3f" % (lModels[i][0], lModelAdjR2[i], lModelRmses[i], lModelScInd[i])
    print(msg)

# *** Model Summary ***
# Model Type            AdjR2       RMSE         SI
# LinearRegression      0.829      1.887      0.170
# RidgeRegression       0.829      1.885      0.170
# LassoRegression       0.810      1.950      0.176
# ElasticNet            0.812      1.938      0.175

# find model with best adj-r2 & print details
print("\n*** Best Model ***")
vBMIndex = lModelAdjR2.index(max(lModelAdjR2))
print("Index       : ",vBMIndex)
print("Model Name  : ",lModels[vBMIndex][0])
print("Adj-R-Sq    : ",lModelAdjR2[vBMIndex])
print("RMSE        : ",lModelRmses[vBMIndex])
print("ScatterIndex: ",lModelScInd[vBMIndex])

# *** Best Model ***
# Index       :  0
# Model Name  :  LinearRegression
# Adj-R-Sq    :  0.8288479306701959
# RMSE        :  1.8870074989811774
# ScatterIndex:  0.16998129124231814

##############################################################
# predict from new data 
##############################################################

# create model from full dataset
print(allCols)

# regression summary for feature
print("\n*** Regression Summary Again ***")
X = sm.add_constant(df[allCols])
y = df[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# now create linear regression model
print("\n*** Regression Model ***")
X = df[allCols].values
y = df[depVars].values
mname = lModels[vBMIndex][0]
model = lModels[vBMIndex][1]
model.fit(X,y)
print(mname)
print(model)

# read dataset
dfp = pd.read_excel('./data/students_mar_prd.xlsx')

print("\n*** Structure ***")
print(dfp.info())

##############################################################
# Data Transformation
##############################################################

# convert string / categoric to numeric

print(dfp['school'].unique())
dfp['school'] = lesh.transform(dfp['school'])
print(dfp['school'].unique())
print("Done ...")


print(dfp['sex'].unique())
dfp['sex'] = lesex.transform(dfp['sex'])
print(dfp['sex'].unique())
print("Done ...")


print(dfp['address'].unique())
dfp['address'] = lead.transform(dfp['address'])
print(dfp['address'].unique())
print("Done ...")

print(dfp['famsize'].unique())
dfp['famsize'] = lefam.transform(dfp['famsize'])
print(dfp['famsize'].unique())
print("Done ...")

print(dfp['Pstatus'].unique())
dfp['Pstatus'] = lep.transform(dfp['Pstatus'])
print(dfp['Pstatus'].unique())
print("Done ...")

print(dfp['Mjob'].unique())
dfp['Mjob'] = leMjob.transform(dfp['Mjob'])
print(dfp['Mjob'].unique())
print("Done ...")

print(dfp['Fjob'].unique())
dfp['Fjob'] = lef.transform(dfp['Fjob'])
print(dfp['Fjob'].unique())
print("Done ...")

print(dfp['reason'].unique())
dfp['reason'] = leRes.transform(dfp['reason'])
print(dfp['reason'].unique())
print("Done ...")

print(dfp['guardian'].unique())
dfp['guardian'] = leguar.transform(dfp['guardian'])
print(dfp['guardian'].unique())
print("Done ...")

print(dfp['schoolsup'].unique())
dfp['schoolsup'] = leschool.transform(dfp['schoolsup'])
print(dfp['schoolsup'].unique())
print("Done ...")

print(dfp['famsup'].unique())
dfp['famsup'] = lefamsup.transform(dfp['famsup'])
print(dfp['famsup'].unique())
print("Done ...")

print(dfp['paid'].unique())
dfp['paid'] = lepaid.transform(dfp['paid'])
print(dfp['paid'].unique())
print("Done ...")

print(dfp['activities'].unique())
dfp['activities'] = leact.transform(dfp['activities'])
print(dfp['activities'].unique())
print("Done ...")

print(dfp['nursery'].unique())
dfp['nursery'] = lenur.transform(dfp['nursery'])
print(dfp['nursery'].unique())
print("Done ...")

print(dfp['higher'].unique())
dfp['higher'] = lehig.transform(dfp['higher'])
print(dfp['higher'].unique())
print("Done ...")

print(dfp['internet'].unique())
dfp['internet'] = leint.transform(dfp['internet'])
print(dfp['internet'].unique())
print("Done ...")

print(dfp['romantic'].unique())
dfp['romantic'] = lerom.transform(dfp['romantic'])
print(dfp['romantic'].unique())
print("Done ...")

# Recheck info
print("\n*** Structure ***")
print(df.info())

# check nulls
print('\n*** Columns With Nulls ***')
print(dfp.isnull().sum()) 

# split X & y
print("\n*** Split Predict Data ***")
X_pred = dfp[allCols].values
y_pred = dfp[depVars].values
print(X_pred)
print(y_pred)

# predict
print("\n*** Predict Data ***")
p_pred = model.predict(X_pred)
# read dataset again because we dont want transformed data
dfp = pd.read_excel('./data/students_mar_prd.xlsx')
# upate predict
dfp['predict'] = p_pred
#write data file
dfp.to_csv("./data/PRD_STUDENTSMARKSRESULT.csv", index=False)
print("Done ... ")


# no y_pred values given
# so show predicted values
print("\n*** Print Predict Data ***")
for idx in dfp.index:
     print(dfp['school'][idx], dfp['predict'][idx])
print("Done ... ")

# *** Print Predict Data ***
# MS 3.2464216850812213
# GP 9.951614504926575
# GP 12.874981306312389
# MS 13.997668200412813
# MS 14.108945694544422
# GP 10.764954094931767
# GP 12.378002780413189
# MS 17.433965875047114
# MS 11.288930618209356
# GP 12.11295950557816
# GP 13.376806530493734
# MS 12.539713437267734
# Done ... 
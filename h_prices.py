#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:50:59 2019

@author: brijesh
"""



categorial = ['MSSubClass',
 'MSZoning',
 'Street',
 'Alley',
 'LotShape',
 'LandContour',
 'Utilities',
 'LotConfig',
 'LandSlope',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'ExterQual',
 'ExterCond',
 'Foundation',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating',
 'HeatingQC',
 'CentralAir',
 'Electrical',
 'KitchenQual',
 'Functional',
 'FireplaceQu',
 'GarageType',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PavedDrive',
 'PoolQC',
 'Fence',
 'MiscFeature',
 'SaleType',
 'SaleCondition']


numeric = ['LotFrontage',
 'LotArea',
 'OverallQual',
 'OverallCond',
 'YearBuilt',
 'YearRemodAdd',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageYrBlt',
 'GarageCars',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'MiscVal',
 'MoSold',
 'YrSold']
# reading data
import pandas as pd

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
'''
allcols = data_train.columns
numeric = []
for col in allcols:
    print(col)
    inp = int(input('Is it numeric'))
    if inp==1:
        numeric.append(col)
len(numeric)




numeric.remove('SalePrice')
numeric
categorical = []
for col in allcols:
    if col not in numeric:
        categorical.append(col)
'''
y = data_train.SalePrice.values
data_train.drop(['SalePrice'], axis=1, inplace = True)
data = pd.concat([data_train,data_test],axis=0)

cat_data = data.loc[:,categorial]


cat_data = pd.get_dummies(cat_data)

num_data = data.loc[:,numeric]

data = pd.concat([cat_data,num_data],axis = 1)

all_X = data.iloc[:,:].values

all_X.shape

X = all_X[:1460,:]
submit_X = all_X[1460:,:]


# Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Imputing 
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.transform(X_test)
submit_X = my_imputer.transform(submit_X)


# without scaling linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
linear_reg = LinearRegression()
linear_reg.fit(X_train,y_train)

y_pred = linear_reg.predict(X_test)
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))


#Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
submit_X = sc.transform(submit_X)

#applying pca and predicting
from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
submit_X = pca.transform(submit_X)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
linear_reg = LinearRegression()
linear_reg.fit(X_train,y_train)

y_pred = linear_reg.predict(X_test)
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))



# applying lda and linear regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 20)
X_train = lda.fit_transform(X_train,y_train)
X_test = lda.transform(X_test)
submit_X = pca.transform(submit_X)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
linear_reg = LinearRegression()
linear_reg.fit(X_train,y_train)

y_pred = linear_reg.predict(X_test)
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))




# Xgboost (best accuracy)
from xgboost import XGBRegressor
cls_ = XGBRegressor()
cls_.fit(X_train,y_train)
y_pred = cls_.predict(X_test)
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))


#support vector machines
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

'''parameters = [{'C':[1,10,100,1000,10000,100000], 'kernel':['linear']},
               {'C':[1,10,100,1000,10000,100000], 'kernel':['poly'],'degree':[1,2,3]}
               ]'''
parameters = [{'C':[1,10,100,1000,10000,100000], 'kernel':['rbf','linear']}
               ]
grid_search = GridSearchCV(estimator=SVR(),
                           param_grid=parameters,
                           #scoring='accuracy',
                           cv=10)
grid_search.fit(X_train,y_train)
grid_search.best_params_



clf = SVR(kernel='linear', C=10000.0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))








# creating dataframe and converting to csv file
y_submit = cls_.predict(submit_X)
type_submit = pd.read_csv('sample_submission.csv')
Id = type_submit.Id.values
d = {'Id':Id, 'SalePrice': y_submit}
submit_df = pd.DataFrame(d) 
submit_df.set_index('Id').to_csv('submit1.csv')
submit_df.shape

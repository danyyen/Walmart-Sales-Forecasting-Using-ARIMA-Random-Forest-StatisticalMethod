# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 17:00:49 2025

@author: ND
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#reading the required data files
sales_data = pd.read_csv('sales_data.csv')
stores_data = pd.read_csv('stores.csv')
features_data = pd.read_csv('features.csv')


# -------- EXPLORATORY DATA ANALYSIS
# view first 5 rows
sales_data.head()

# check for missing vales in a dataframe.. 0 means no missing value and 1 means yes for missing values
sales_data.isna().sum()

#check data column data types
sales_data.dtypes

#check shape of data
sales_data.shape

#check dataframe characteristics
sales_data.describe()

#we can see that sales have negative values, now we wanna see if this is a systematic issue or random
sales_data_check = sales_data.loc[sales_data['Weekly_Sales'] <= 0]
sales_data_check.shape

#we can see that negative sales applies to all stores
sales_data_check.nunique()

#check the proportion for negative sales to record to total records
sales_data_check.shape[0]/sales_data.shape[0]

#convert store and dept column to string data type
sales_data['Store'] = sales_data['Store'].astype(str)
sales_data['Dept'] = sales_data['Dept'].astype(str)

#converts Isholiday column from true/fales to to interger 1/0 
sales_data['IsHoliday'] = sales_data['IsHoliday']*1

#Since the negtative sales affects a small percentage of the data, lets drop these records
sales_data2 = sales_data.loc[sales_data['Weekly_Sales']>0]
sales_data2.shape

sales_data2.dtypes

#Creating a function that does all of the above tasks in one go
def get_basic_stats(dfname):
    print("Shape of dataframe is " + str(dfname.shape))
    print("Below are datatypes of columns in DF")
    print(dfname.dtypes.sort_values())
    print("Below are missing values in each column")
    print(dfname.isna().sum().sort_values())
    print("Below are the number of unique values taken by a column")
    print(dfname.nunique().sort_values())
    print("Below are some records in DF")
    print(dfname.head())
    print("Basic Stats for numeric variables")
    print(dfname.describe())
    
#apply function to features data set
get_basic_stats(features_data)

#apply function to features data set
get_basic_stats(stores_data) 

#coverting variable to correct datatype
features_data['Store'] = features_data['Store'].astype(str)
stores_data['Store'] = stores_data['Store'].astype(str)

features_data2 = features_data.drop(['IsHoliday'], axis=1)


#-----MISSING VALUE IMPUTATION 
#replace missing values with mean
features_data2['Unemployment'].fillna(features_data2['Unemployment'].mean(), inplace=True)
features_data2['CPI'].fillna(features_data2['CPI'].mean(), inplace=True)


#Markdown variables are missing in over half of the cases, and the most logical reason for that seems to be the absence \
#of any markdown in those cases 
#replace all missing values with 0
features_data2.fillna(0, inplace=True)

#group stores data by type and calculate mean of store size
stores_data.groupby('Type').agg({'Size':'mean'})

        #DATA PREPARATION ----- 
#Stores unique key = Store
#Features Data - Store+Date
#Store+Dept+Date

#grouping by store and date then compare number of rows without grouping
features_data2.groupby(['Store', 'Date']).size().shape[0] - features_data2.shape[0]

#grouping by store and date then compare number of rows without grouping
sales_data.groupby(['Store', 'Dept', 'Date']).size().shape[0] - sales_data.shape[0]

#Merging datasets
sales_features_data = pd.merge(sales_data2, features_data2, on=['Store', 'Date'], how='inner')
sales_features_data.shape

#merge sales data and features data using store column
combined_data = pd.merge(sales_features_data, stores_data, on='Store', how='inner')
combined_data.shape


#--------Dividing data into Train and Test set
#getting the date variable in the correct format
combined_data['Date2'] = pd.to_datetime(combined_data['Date'], format = '%Y-%m-%d')

#match the number of weeks of difference with the number of distinct weeks in the data
print(combined_data['Date2'].nunique())
((combined_data['Date2'].max() - combined_data['Date2'].min()).days / 7) + 1 #1 week was added to this

#(combined_data['Date2'].max() - combined_data['Date2'].min())/7 


#Getting the date that comes at 70% of the data and using it to divide the original data
unique_dates = pd.DataFrame(combined_data['Date2'].unique(), columns = ['date'])
unique_dates.sort_values('date', inplace = True)
splitter = round(unique_dates.shape[0]*0.7, 0)
split_date=unique_dates.iloc[int(splitter)-1]['date']
print(split_date)

#Using that date to make the split
combined_data_train = combined_data.loc[combined_data['Date2'] <= split_date]
combined_data_test = combined_data.loc[combined_data['Date2'] > split_date]
combined_data_train.shape[0]/combined_data.shape[0]


# ----------UNIVARIATE analysis
plt.figure(figsize=(20,10))
plt.hist(list(combined_data_train['Store']), bins = combined_data_train['Store'].nunique() )
plt.show()

#identify all categorical columns in the dataframe
cat_columns = combined_data_train.select_dtypes(include=object).columns.tolist()
#univariate analysis of categorical variables images
for var in cat_columns:
    plt.figure(figsize = (20,10))
    plt.hist(list(combined_data_train[var]), bins = combined_data_train[var].nunique() )
    plt.title(var)
    plt.show()
    
#get all columns with number datatype
num_columns = combined_data_train.select_dtypes(np.number).columns.tolist()

#create list for columns with mark in them
mark_columns = [x for x in num_columns if 'Mark' in x]
mark_columns

#remove markdown columns from the list
nonmark_columns = list(set(num_columns).difference(set(mark_columns)))
nonmark_columns

#display all number variables
combined_data_train.boxplot(column = mark_columns, figsize=(20,10))
plt.show()

for var in nonmark_columns:
    combined_data_train.boxplot(column = var, figsize=(20,10))
    plt.show()
    
    
#-----------Bivariate Analysis
#Keeping the numeric predictors
train_num = combined_data_train[num_columns]
train_num = train_num.drop(['Weekly_Sales'], axis=1)
train_num.head()

#Checking the highly correlated variables
corr_values = train_num.corr().unstack().reset_index()
print(corr_values.shape)
corr_values2 = corr_values[corr_values['level_0'] > corr_values['level_1'] ]
print(corr_values2.shape)
corr_values2.columns = ['var1', 'var2', 'corr_value']
corr_values2['corr_abs'] = corr_values2['corr_value'].abs()
corr_values2.sort_values( 'corr_abs', ascending=False, inplace=True)
corr_values2.head(10)

#creating a heat map to see the degree of correlation visually
plt.figure(figsize=(12, 12))
vg_corr = train_num.corr().abs()
sns.heatmap(vg_corr, xticklabels = vg_corr.columns.values,yticklabels = vg_corr.columns.values, annot = True)
plt.show()

#Plotting each numeric predictor against the other
sns.pairplot(train_num)
plt.show()


#---------Variable trend with Dependent Variable
#Looking at average weekly Sales by STORE, DEPT...directly taking average will lead to incorrect result..
for var in ['Store', 'Dept']:
    print(var)
    agg_temp = combined_data_train.groupby([var, 'Date'], as_index=False).agg({'Weekly_Sales': 'sum'})
    agg_temp2 = agg_temp.groupby([var], as_index=False).agg({'Weekly_Sales': 'mean'})
    agg_temp2[var] = pd.to_numeric(agg_temp2[var])
    agg_temp2.sort_values(var, inplace=True)
    print(agg_temp2.head())
    plt.figure(figsize=(20,10))
    plt.title('Avg Weekly Sales by ' + var)
    plt.barh(agg_temp2[var], agg_temp2['Weekly_Sales'])
    plt.show()

#Using scatter plot to check trend of predictors with dependent variable
var_list = [x for x in combined_data_train.columns]
columnstodrop = ['Weekly_Sales', 'Date']
var_list = list(set(var_list).difference(set(columnstodrop)))
for varname in var_list:
    plt.figure(figsize=(20,10))
    plt.scatter(combined_data_train[varname], combined_data_train['Weekly_Sales'])
    plt.xlabel(varname); plt.ylabel("Weekly_Sales")
    plt.show()
    
# Select only numeric columns
numeric_df = combined_data_train.select_dtypes(include='number')

# Compute correlation matrix
all_corr = numeric_df.corr().unstack().reset_index()

# Filter for correlations with 'Weekly_Sales'
corr_table = all_corr[all_corr['level_1'] == 'Weekly_Sales']
corr_table = corr_table[corr_table['level_0'] != 'Weekly_Sales']

# Rename and sort
corr_table.columns = ['var1', 'var2', 'corr_value']
corr_table['corr_abs'] = corr_table['corr_value'].abs()
corr_table = corr_table.sort_values(by='corr_abs', ascending=False)

print(corr_table)


#----------DATE TRENDS
#Adding the week number and year column 
combined_data_train.loc[:, 'week_number'] = combined_data_train['Date2'].dt.isocalendar().week
combined_data_train.loc[:, 'year_number'] = combined_data_train['Date2'].dt.year

#Checking the holiday weeks with highest total sales 
temper = combined_data_train.groupby('Date', as_index=False).agg({'Weekly_Sales':'sum'})
print(temper['Weekly_Sales'].mean())
temper = combined_data_train.groupby(['Date', 'IsHoliday', 'week_number', 'year_number'], as_index=False).agg({'Weekly_Sales':'sum'})
temper.loc[temper['IsHoliday'] == 1].sort_values(by = 'Weekly_Sales', ascending=False)

#Checking the weeks with highest total sales..please note that most of them are non holidays
temper.sort_values(by = 'Weekly_Sales', ascending=False).head(20)

#Checking weekly Sales of different years juxtaposed
another_agg = combined_data_train.groupby(['week_number', 'year_number', 'IsHoliday'], as_index=False).agg({ 'Weekly_Sales' : 'sum'})
plt.figure(figsize=(20,10))
plt.title('Total Juxtaposed Weekly Sales')
sns.lineplot(x="week_number", y="Weekly_Sales", hue="year_number", data = another_agg, marker = 'o')
plt.show()

#Checking total weekly Sales of different Stores
for var in ['Store', 'Dept']:
    agg_temp = combined_data_train.groupby(['Date2', var], as_index=False).agg({ 'Weekly_Sales' : 'sum'})
    print(agg_temp.sort_values(by='Weekly_Sales', ascending=False).head(10))
    plt.figure(figsize=(20,10))
    plt.title('Total Weekly Sales by ' + var)
    agg_temp[var] = pd.to_numeric(agg_temp[var])#The hue variable has to be numerical 
    sns.lineplot(x="Date2", y="Weekly_Sales", hue=var, data = agg_temp, marker = 'o')
    plt.show()

#Checking weekly Sales of different years juxtaposed..by Store and by Dept
for var in ['Store', 'Dept']:
    another_agg = combined_data_train.groupby([var, 'week_number', 'year_number', 'IsHoliday'], as_index=False).agg({ 'Weekly_Sales' : 'sum'})
    for idd in another_agg[var].unique() :
        temper = another_agg.loc[ another_agg[var] == idd ]
        print(temper.shape)
        plt.figure(figsize=(20,10))
        plt.title('Total Juxtaposed Weekly Sales by ' + var + " " + idd)
        sns.lineplot(x="week_number", y="Weekly_Sales", hue="year_number", data = temper, marker = 'o')
        plt.show()
        

#---------FEATURE CREATION
#marking the holiday names
conditions = [combined_data_train['IsHoliday'] == 0, combined_data_train['week_number'] == 52, combined_data_train['week_number'] == 47, \
             combined_data_train['week_number'] == 36, combined_data_train['week_number'] == 6]
choices = ['no_holiday', 'christmas', 'thanksgiving', 'labor_day', 'super_bowl']
combined_data_train['holiday_type' ] = np.select(conditions, choices, default=np.nan)
combined_data_train['holiday_type'].value_counts()

#function to create sales rush weeks 
def season_variables(df):
    df['week_number'] = df['Date2'].dt.isocalendar().week
    df['year_number'] = df['Date2'].dt.year

    conditions = [
        df['IsHoliday'] == 0,
        df['week_number'] == 52,
        df['week_number'] == 47,
        df['week_number'] == 36,
        df['week_number'] == 6
    ]
    choices = ['no_holiday', 'christmas', 'thanksgiving', 'labor_day', 'super_bowl']
    df['holiday_type'] = np.select(conditions, choices, default=np.nan)

    print(df['holiday_type'].value_counts())

    christmas_df = pd.DataFrame(df[df['holiday_type'] == 'christmas']['Date2'].unique(), columns=['christmas_week'])
    christmas_df['minus1'] = christmas_df['christmas_week'] - pd.to_timedelta(7, unit='d')
    christmas_df['minus2'] = christmas_df['christmas_week'] - pd.to_timedelta(14, unit='d')
    christmas_df['minus3'] = christmas_df['christmas_week'] - pd.to_timedelta(21, unit='d')

    df['christmas_week'] = df['Date2'].isin(christmas_df['christmas_week']).astype(int)
    df['christmas_minus1'] = df['Date2'].isin(christmas_df['minus1']).astype(int)
    df['christmas_minus2'] = df['Date2'].isin(christmas_df['minus2']).astype(int)
    df['christmas_minus3'] = df['Date2'].isin(christmas_df['minus3']).astype(int)
    df['thanksgiving_week'] = (df['holiday_type'] == 'thanksgiving').astype(int)

    print(df['christmas_week'].value_counts())
    print(df['christmas_minus1'].value_counts())
    print(df['christmas_minus2'].value_counts())
    print(df['christmas_minus3'].value_counts())
    print(df['thanksgiving_week'].value_counts())

    return df

combined_data_test = season_variables(combined_data_test)

#Creating another feature..Prediction based on Store and Dept
store_dept_median = combined_data_train.groupby(['Store', 'Dept'], as_index=False).agg({'Weekly_Sales' :  ['median']})
print(store_dept_median.shape)
store_dept_median.columns = ["_".join(x) for x in store_dept_median.columns.ravel()]
store_dept_median.columns = ['Store', 'Dept', 'Weekly_Sales_median']
print(store_dept_median.head())

#Adding the feature into train and test data
print(combined_data_train.shape)
combined_data_train2 = pd.merge(combined_data_train, store_dept_median, on=['Store', 'Dept'], how='left')
print(combined_data_train2.shape)
print(combined_data_test.shape)
combined_data_test2 = pd.merge(combined_data_test, store_dept_median, on=['Store', 'Dept'], how='left')
print(combined_data_test2.shape)

#Filling the few missing values for new feature in test data with mean
combined_data_test2['Weekly_Sales_median'].fillna(combined_data_test2['Weekly_Sales_median'].mean() , inplace=True)


#---------------MODELLING
#-------------Prediction 1

#Prediction based on Store, Dept and Week
simple_prediction = combined_data_train.groupby(['Store', 'Dept', 'week_number'], as_index=False).agg({'Weekly_Sales' :  ['mean', 'count' ]})
print(simple_prediction.shape)
simple_prediction.columns = ["_".join(x) for x in simple_prediction.columns.ravel()]
simple_prediction = simple_prediction.drop(['Weekly_Sales_count'], axis=1)
simple_prediction.columns = ['Store', 'Dept', 'week_number', 'Sales_prediction']
simple_prediction['prediction_type'] = 1
simple_prediction.head()

pd.merge(combined_data_test, simple_prediction, on = ['Store', 'Dept', 'week_number']).shape

#Prediction based on Dept..since previous prediction will miss required combinations..
#this is being done to provide approximation for sales in a dept irrespective of size 
moresimple_prediction = combined_data_train.groupby(['Dept'], as_index=False).agg({'Weekly_Sales' :  ['mean', 'count' ]})
print(moresimple_prediction.shape)
moresimple_prediction.columns = ["_".join(x) for x in moresimple_prediction.columns.ravel()]
print(moresimple_prediction['Weekly_Sales_count'].describe())
moresimple_prediction = moresimple_prediction.drop(['Weekly_Sales_count'], axis=1)
moresimple_prediction.columns = ['Dept', 'Sales_prediction']
moresimple_prediction['prediction_type'] = 2
moresimple_prediction.head()

#making sure there is no more than one entry for this combo
def score_direct(df_to_score, pred_df1, pred_df2):
    """
    We will use the 2 set of predictions made to estimate sales on the test data 
    """
    df_to_score['week_number'] =  df_to_score['Date2'].dt.isocalendar().week
    df_input = df_to_score[['Store', 'Dept', 'Date2',  'week_number']]
    print(df_input.shape)
    #checking if there is only 1 entry per combination being used to predict (i.e. Store, Dept, Week Number)
    print( df_input.groupby(['Store', 'Dept', 'week_number']).size().shape )
    #Using prediction1 and separate out combinations that remain unscored
    df_scored = pd.merge(df_input, pred_df1, on= ['Store', 'Dept', 'week_number'], how='left')
    df_present = df_scored[ pd.isnull(df_scored['prediction_type']) ==False ]
    df_missed =  df_scored[ pd.isnull(df_scored['prediction_type']) ]
    df_missed = df_missed.drop(['Sales_prediction', 'prediction_type'], axis=1)
    #using prediction2 to score the unscored test data
    df_missed_scored = pd.merge(df_missed, pred_df2, on= ['Dept'], how='left')
    print(df_missed_scored['prediction_type'].isna().sum())    
    #getting the scored test data in single DF
    df_full_scored = pd.concat( [df_present, df_missed_scored ] , axis=0)
    print(df_full_scored.shape)
    print(df_full_scored['prediction_type'].value_counts() )
    return df_full_scored

test_full_scored = score_direct(combined_data_test, simple_prediction, moresimple_prediction)
print(test_full_scored)


#---------PREDICTION 2 - RANDOM FOREST REGRESSION
#doing one hot encoding of the categorical variables 
cat_columns = ['Type']
combined_data_train2 = pd.concat([combined_data_train2, pd.get_dummies(combined_data_train2[cat_columns], drop_first=True)], axis=1)
combined_data_test2 = pd.concat([combined_data_test2, pd.get_dummies(combined_data_test2[cat_columns], drop_first=True)], axis=1)

columnstodrop = ['Type' , 'Store', 'Dept', 'Date', 'Date2', 'Weekly_Sales', 'holiday_type', 'week_number', 'year_number']
X_train = combined_data_train2.drop(columnstodrop, axis = 1 )
X_test = combined_data_test2.drop(columnstodrop, axis = 1 )
print(X_train.shape)
print(X_test.shape)

#MarkDown4, 5  found to be over 60% correlated to other markdown variables
X_train = X_train.drop(['MarkDown4', 'MarkDown5'], axis = 1 )
X_test = X_test.drop(['MarkDown4', 'MarkDown5'], axis = 1 )

Y_train = combined_data_train2['Weekly_Sales']
Y_test = combined_data_test2['Weekly_Sales']


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# RFR with hyper-parameter tuning
import time
aa = time.time()
rfr = RandomForestRegressor(random_state=42,  n_jobs = -1, max_depth = 5)#criterion = 'mse',
param_grid = { 
'n_estimators': [50, 100, 200]
}
CV_rfr = GridSearchCV(estimator = rfr, param_grid=param_grid, cv= 3)
CV_rfr.fit(X_train, Y_train)
rfr_best = CV_rfr.best_estimator_
print(rfr_best)
print(time.time() - aa)

def get_FI(modelname, dfname):
    importance_list = pd.DataFrame(modelname.feature_importances_, columns=['importance'])
    varnames_list = pd.DataFrame(dfname.columns.tolist(), columns=['feature'])
    feature_importance = pd.concat([varnames_list, importance_list], axis=1)
    feature_importance = feature_importance.sort_values(by=['importance'], ascending=False)
    feature_importance['cum_importance'] = feature_importance['importance'].cumsum()
    return feature_importance

get_FI(rfr_best, X_train)


# ----------- Prediction 3 - ARIMA, another statistical technique
#Taking a example randomly to showcase ARIMA
particular_data = sales_data2.loc[(sales_data2['Store'] == '13') & (sales_data2['Dept'] == '17'), \
                                          ['Date', 'Weekly_Sales']   ]
particular_data['Date'] =  pd.to_datetime(particular_data['Date'])
particular_data.index = particular_data.Date
particular_data = particular_data.drop('Date', axis=1)
particular_data.head()

particular_data_train, particular_data_test  = particular_data[particular_data.index <= split_date],  particular_data[particular_data.index > split_date]
print(particular_data_train.shape)
print(particular_data_test.shape)

#time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(particular_data, model='multiplicative')
fig = result.plot()

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(particular_data_train)

#Running ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# Fit model
model = ARIMA(particular_data_train, order=(0, 1, 0))
model_fit = model.fit()

# Summary
print(model_fit.summary())

# Plot residuals
residuals = model_fit.resid
residuals.plot(title="Residuals")
plt.show()

residuals.plot(kind='kde', title="Residuals KDE")
plt.show()

print(residuals.describe())

# Install pyramid using
#!pip install pmdarima
#Hyper Parameter Tuning of ARIMA model..No seasonality taken.
# searches for the best arima model within p and q values
from pmdarima import auto_arima
auto_model = auto_arima(particular_data_train, start_p=1, d=0, start_q=1,
                           max_p=3, max_q=3, m=1,
                           start_P=0, seasonal=False,
                           D=None, trace=True,
                           error_action='ignore'
                           )#suppress_warnings=True
auto_model.fit(particular_data_train)
print(auto_model)
test_arima = auto_model.predict(n_periods= len(particular_data_test))

sales_data_train = combined_data_train[['Store', 'Dept', 'Date2', 'Weekly_Sales']]
sales_data_test = combined_data_test[['Store', 'Dept', 'Date2', 'Weekly_Sales']]

#checking the number of records for each store, dept combination
temper = sales_data_train.groupby(['Store', 'Dept'], as_index=False).agg({'Date2': 'count' })
temper = temper.rename(columns= {'Date2' : 'counter'})
temper.head()
temper2 = temper.groupby(['counter'], as_index=False).agg({'Store': 'count' })
temper2

temper.shape

# -------
from sklearn.metrics import mean_absolute_error
def arima_modeler(df_train, df_test):
    # infer split date from training data automatically
    split_date = df_train['Date2'].max()
    
    # Pre-filter valid Store-Dept combinations
    combos_train = df_train.groupby(['Store', 'Dept']).agg(
        Date2_min=('Date2', 'min'),
        Date2_max=('Date2', 'max'),
        Date2_count=('Date2', 'count')
    ).reset_index()

    combos_test = df_test[['Store', 'Dept']].drop_duplicates()
    combos_common = pd.merge(combos_train, combos_test, on=['Store', 'Dept'])
    
    combos_common['total_weeks'] = ((combos_common['Date2_max'] - combos_common['Date2_min']).dt.days) / 7 + 1
    combos_common['date_gap'] = combos_common['total_weeks'] - combos_common['Date2_count']
    
    # Keep only full-series combinations
    combos_filtered = combos_common[
        (combos_common['date_gap'] == 0) &
        (combos_common['Date2_max'] == split_date) &
        (combos_common['Date2_count'] == 100)  # Assumes 100 weeks is expected
    ].sort_values(['Store', 'Dept']).reset_index(drop=True)

    # Filter the training and test sets
    df_train_filtered = df_train.merge(combos_filtered[['Store', 'Dept']], on=['Store', 'Dept'])
    df_test_filtered = df_test.merge(combos_filtered[['Store', 'Dept']], on=['Store', 'Dept'])

    # Output containers
    all_preds = []
    scores = []

    for _, row in combos_filtered.iterrows():
        store, dept = row['Store'], row['Dept']
        print(f"Training ARIMA model for Store {store}, Dept {dept}")

        train_data = df_train_filtered[(df_train_filtered['Store'] == store) & (df_train_filtered['Dept'] == dept)]
        test_data = df_test_filtered[(df_test_filtered['Store'] == store) & (df_test_filtered['Dept'] == dept)]

        y_train = train_data.set_index('Date2')['Weekly_Sales']
        y_test = test_data.set_index('Date2')['Weekly_Sales']

        try:
            model = auto_arima(
                y_train, start_p=1, start_q=1, max_p=3, max_q=3,
                seasonal=False, trace=False, error_action='ignore', suppress_warnings=True
            )
            y_pred = model.predict(n_periods=len(y_test))

            mae = mean_absolute_error(y_test, y_pred)
            print(f"MAE for Store {store}, Dept {dept}: {mae:.2f}")

            preds_df = test_data.copy()
            preds_df['Predicted_Sales'] = y_pred
            all_preds.append(preds_df)

            scores.append({'Store': store, 'Dept': dept, 'MAE': mae})

        except Exception as e:
            print(f"Model failed for Store {store}, Dept {dept}: {e}")

    # Combine results
    scored_df = pd.concat(all_preds, ignore_index=True)
    scores_df = pd.DataFrame(scores)

    return scored_df, scores_df

partialtest_scored_arima = arima_modeler(sales_data_train, sales_data_test)


#---------------------------------------------------------------------------------------
partialtest_scored_arima.shape

sales_data_test_arima = pd.merge(sales_data_test, partialtest_scored_arima.drop(['Weekly_Sales'], axis=1), on=['Store', 'Dept', 'Date2'], how='left' )
sales_data_test_arima.isna().sum()

#Making the prediction for cases where ARIMA couldnt make the prediction due to lack of data  
sales_data_test_missing =sales_data_test_arima.loc[pd.isnull(sales_data_test_arima['score']) ]
sales_data_test_mp = pd.merge(sales_data_test_missing.drop('score', axis=1) , moresimple_prediction.rename(columns = {'Sales_prediction':'score'}) , on='Dept' )
sales_data_test_mp.head()

sales_data_test_mp.drop('prediction_type', axis=1, inplace=True)
fullyscored_test_arima = pd.concat([partialtest_scored_arima, sales_data_test_mp ], axis=0)
fullyscored_test_arima.shape


#-------------- MODEL EVALUATION ----------------
#Getting the sales numbers in the data
test_scored_stats = pd.merge(test_full_scored, sales_data_test, on = ['Store', 'Dept', 'Date2'])
test_scored_stats.shape

def rf_scorer(X_df, Y_df, model_name):
    """
    obtain the prediction and compare with the actual
    """
    scored_rf = pd.DataFrame({ 'Sales_prediction' : model_name.predict(X_df)} ) 
    scored_actuals_rf = pd.concat( [scored_rf, Y_df.reset_index() ] , axis=1)
    scored_actuals_rf = scored_actuals_rf.drop(['index'], axis = 1)
    scored_actuals_rf['error_abs_rf'] = (scored_actuals_rf['Sales_prediction'] - scored_actuals_rf['Weekly_Sales']).abs()
    print(scored_actuals_rf['error_abs_rf'].describe())
    return scored_actuals_rf

#Creating absolute error metric in each dataframe
test_scored_stats['error_abs_st'] = (test_scored_stats['Weekly_Sales'] - test_scored_stats['Sales_prediction']).abs()
test_rf = rf_scorer(X_test, Y_test, rfr_best)
fullyscored_test_arima['error_abs_ar'] = (fullyscored_test_arima['Weekly_Sales'] - fullyscored_test_arima['score']).abs()

#Getting Mean Absolute Error
from sklearn.metrics import mean_squared_error
import math
print( "MAE Prediction Stats =   " + str((mean_absolute_error(test_scored_stats['Weekly_Sales'],  test_scored_stats['Sales_prediction'])) ) )  
print( "MAE Prediction Random Forest =   " + str((mean_absolute_error(test_rf['Weekly_Sales'],  test_rf['Sales_prediction'])) )  )
print( "MAE Prediction ARIMA  = " + str((mean_absolute_error(fullyscored_test_arima['Weekly_Sales'],  fullyscored_test_arima['score'])) )  )

#Getting RMSE
from sklearn.metrics import mean_squared_error
import math
print( "RMSE Prediction Stats =   " + str(math.sqrt(mean_squared_error(test_scored_stats['Weekly_Sales'],  test_scored_stats['Sales_prediction'])) ) )  
print( "RMSE Prediction Random Forest =   " + str(math.sqrt(mean_squared_error(test_rf['Weekly_Sales'],  test_rf['Sales_prediction'])) )  )
print( "RMSE Prediction ARIMA  = " + str(math.sqrt(mean_squared_error(fullyscored_test_arima['Weekly_Sales'],  fullyscored_test_arima['score'])) )  )

#Plotting the absolute error with the weekly sales for all the 3 predictions
temper1 = test_scored_stats.sort_values(by=['Weekly_Sales'])
temper2 = test_rf.sort_values(by=['Weekly_Sales'])
temper3 = fullyscored_test_arima.sort_values(by=['Weekly_Sales'])

plt.figure(figsize=(20,10))
plt.title('Error Comparison of different Models on Test data')
plt.xlabel('Weekly Sales')
plt.ylabel('Absolute Error')
plt.plot(temper1['Weekly_Sales'], temper1['error_abs_st'], label ='Statistical', alpha = 0.7)
plt.plot(temper2['Weekly_Sales'], temper2['error_abs_rf'], label = 'RFR', alpha = 0.7)
plt.plot(temper3['Weekly_Sales'], temper3['error_abs_ar'], label = 'ARIMA', alpha = 0.7)
plt.legend()


# ------------- Discussion and Conclusion ------------
# Macro Economic Indicators did not help in prediction of weekly sales. The sales data is very similar over different years \
# and this consistency helped to predict the sales. 
# the statistical approach will be used for sales forecasting because of its low RMSE

















































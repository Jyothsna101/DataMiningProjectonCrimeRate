#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[9]:


import pandas as pd
import numpy as np
import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt


from featurewiz import featurewiz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[ ]:


pip install featurewiz


# # Loading the dataset

# In[2]:


crimes_data = pd.read_csv("/Users/jyothsnaakula/Documents/6040/Data/crime_and_incarceration_by_state.csv")
crimes_data


# # Descriptive Analysis 

# In[3]:


#displaying the number of rows and columns of the dataset
print("Total number of Rows and Columns:",crimes_data.shape)

#displaying the data field values
print("\nColumn Names:\n",crimes_data.columns)


# In[4]:


#displaying the data types
print("\nData types:\n", crimes_data.dtypes)


# In[5]:


#information about the dataframe
crimes_data.info()


# # Statistical Analysis  

# In[6]:


#describing the dataset
round(crimes_data.describe(),1)


# # Data Profiling Report

# In[7]:


crimesdata_profilereport = crimes_data.profile_report(title='Crimes Data Analysis Report', explorative = True)
crimesdata_profilereport


# In[120]:


#saving the profile report 
crimesdata_profilereport.to_file(output_file="Crimes Data Analysis Report.html")


# # Total Prisoner Count in each state from 2001-2016

# In[193]:


crimes_data_group = crimes_data.groupby("jurisdiction")["prisoner_count"].sum()
crimes_data_group


# # Data Cleaning

# In[194]:


#checking for null values in the each column of the dataset
for x in range(17):
    print("%-45s %10d" % (crimes_data.columns.values[x], crimes_data.iloc[:,x].isna().sum()))


# # Detecting Outliers

# In[195]:


#outliers - boxplot

#prisoner count
sns.boxplot(x = "prisoner_count", data=crimes_data, color="lightblue" )
plt.xlabel("Prisoner Count")
plt.title("Count of Prisoners per State")


# In[196]:


#outliers - boxplot

#violent crime total
sns.boxplot(x = "violent_crime_total", data=crimes_data, color="lightblue" )
plt.xlabel("Violent Crime Count")
plt.title("Count of Violent Crime per State")


# In[197]:


#distribution plot


#plotting distribution for the features
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(crimes_data['agg_assault'])
plt.subplot(1,2,2)
sns.distplot(crimes_data['murder_manslaughter'])
plt.show()


# # Cleaning Data

# In[198]:


#checking for datatype of each variable

crimes_data.dtypes


# In[199]:


#data cleaning

#dropping rape_revised column as it has maximum null values of the whole data
crimes_data = crimes_data.drop("rape_revised", axis = 1)


# In[200]:


crimes_data


# In[201]:


#remaining all row values are null for rows with missing values, hence dropping all those 17 values

crimes_data = crimes_data.dropna(subset=['crime_reporting_change'])
crimes_data.shape


# In[202]:


#checking for null values in the each column of the dataset
for x in range(16):
    print("%-45s %10d" % (crimes_data.columns.values[x], crimes_data.iloc[:,x].isna().sum()))


# In[203]:


#distribution plot for rape_legacy

plt.figure(figsize=(16,5))
sns.distplot(crimes_data['rape_legacy'])
plt.show()


# In[204]:


#input for rape_legacy fied value

#missing data - approx 20%
#data is skewed
#input missing values with median


crimes_data['rape_legacy'] = crimes_data['rape_legacy'].fillna(crimes_data['rape_legacy'].median())
print("Imputed Values!")


# In[205]:


#checking for null values in the each column of the dataset
for x in range(16):
    print("%-45s %10d" % (crimes_data.columns.values[x], crimes_data.iloc[:,x].isna().sum()))


# # Correlation Plot

# In[237]:


# correlation plot

plt.figure(figsize = (15,9))
ax = plt.subplot()
sns.heatmap(crimes_data.corr(),annot=True, fmt='.1f', ax=ax, cmap="YlGnBu")
ax.set_title('Correlation Plot')


# # Feature Selection & Extraction

# In[139]:


#automatic feature extraction

target = 'prisoner_count'

features, train = featurewiz(crimes_data, target, corr_limit=0.7, verbose=2, sep=",",
header=0,test_data="", feature_engg="", category_encoders="")


# In[140]:


print(features)


# # Label Encoding

# In[207]:


labelencoder = LabelEncoder()
crimes_data['jurisdiction Labels'] = labelencoder.fit_transform(crimes_data["jurisdiction"])
crimes_data['includes_jails Labels'] = labelencoder.fit_transform(crimes_data["includes_jails"])
crimes_data.head(10)


# In[208]:


#new dataframe for training model

new_crimes_data = pd.DataFrame()

new_crimes_data['jurisdiction'] = crimes_data['jurisdiction Labels']
new_crimes_data['includes_jails'] = crimes_data['includes_jails Labels']
new_crimes_data['state_population'] = crimes_data['state_population']
new_crimes_data['violent_crime_total'] = crimes_data['violent_crime_total']
new_crimes_data['murder_manslaughter'] = crimes_data['murder_manslaughter']
new_crimes_data['agg_assault'] = crimes_data['agg_assault']

new_crimes_data.head(10)


# In[238]:


new_crimes_data.columns


# In[209]:


#extracting features and labels from the dataset

x = new_crimes_data
y = crimes_data['prisoner_count']


# In[210]:


#splitting data into training and testing set

X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=42, test_size=0.2)
print(X_train.shape)
print(X_test.shape)


# # Linear Regression Model

# In[211]:


#linear regression model

linear_regressionmodel = LinearRegression()
linear_regressionmodel.fit(X_train, y_train)


# In[212]:


#predict the result for the model

predicted_value_LR = linear_regressionmodel.predict(X_test)


# In[267]:


#accuracy of the LR for training and testing set

print('Accuracy of Linear Regressor model on training set: {:.2f}'.format(linear_regressionmodel.score(X_train, y_train)))
print('Accuracy of Linear Regressor model on test set:     {:.2f}'.format(linear_regressionmodel.score(X_test, y_test)))

result_LR = linear_regressionmodel.score(X_test, y_test)
result_LR = round(result_LR,4)
result_LR


# # Model Evaluation

# In[305]:


#evaluating the model
print("Model Evaluation of Linear Regression.")
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, predicted_value_LR),1))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, predicted_value_LR),1))  
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, predicted_value_LR)),1))
print("R-Squared value:", metrics.r2_score(y_test, predicted_value_LR))


# # Decision Tree Regressor Model

# In[280]:


#decision tree regressor

decisiontree_model = DecisionTreeRegressor(max_depth=5)
decisiontree_model.fit(X_train, y_train)


# In[281]:


#predict the result for the model

predicted_value_dt = decisiontree_model.predict(X_test)


# In[282]:


#accuracy of the DT for training and testing set

print('Accuracy of Decision Tree Regressor model on training set: {:.2f}'.format(decisiontree_model.score(X_train, y_train)))
print('Accuracy of Decision Tree Regressor model on test set:     {:.2f}'.format(decisiontree_model.score(X_test, y_test)))

result_DT = decisiontree_model.score(X_test, y_test)
result_DT = round(result_DT,3)
result_DT


# In[285]:


#plotting the decision tree

feature_names = x.columns

plt.figure(figsize=(85,10))
a = tree.plot_tree(decisiontree_model,
                   feature_names = feature_names,
                   class_names = crimes_data['prisoner_count'],
                   rounded = True,
                   filled = True,
                   fontsize=12)
plt.show()


# # Feature Importance

# In[286]:


#feature importance for decision tree regressor

(pd.Series(decisiontree_model.feature_importances_, index=x.columns)
   .nlargest(8)
   .plot(kind='barh'))


# # Model Evaluation

# In[301]:


#evaluating the model
print("Model Evaluation of Decision Tree Regressor.")
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, predicted_value_dt),1))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, predicted_value_dt),1))  
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, predicted_value_dt)),1))
print("R-Squared value:", metrics.r2_score(y_test, predicted_value_dt))


# # Random Forest Regressor Model

# In[293]:


#random forest regressor

randomforest_model = RandomForestRegressor(n_estimators = 5000, max_depth=5)
randomforest_model.fit(X_train, y_train)


# In[294]:


#predict the result for the model

predicted_value_rf = randomforest_model.predict(X_test)


# In[295]:


#accuracy of the RF for training and testing set

print('Accuracy of Random Forest Regressor model on training set: {:.2f}'.format(randomforest_model.score(X_train, y_train)))
print('Accuracy of Random Forest Regressor model on test set:     {:.2f}'.format(randomforest_model.score(X_test, y_test)))

result_RF = randomforest_model.score(X_test, y_test)
result_RF = round(result_RF,3)
result_RF


# # Feature Importance

# In[296]:


#feature importance for random forest regressor

(pd.Series(randomforest_model.feature_importances_, index=x.columns)
   .nlargest(8)
   .plot(kind='barh'))


# # Model Evaluation

# In[300]:


#evaluating the model
print("Model Evaluation of Random Forest Regressor.")
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, predicted_value_rf),1))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, predicted_value_rf),1))  
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, predicted_value_rf)),1))
print("R-Squared value:", metrics.r2_score(y_test, predicted_value_rf))


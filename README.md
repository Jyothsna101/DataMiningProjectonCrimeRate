# DataMiningProjectonCrimeRate

#Dataset: Crime and Incarceration in the United States

#About Dataset:
    
    1. With 240,593 inmates housed in state or federal facilities in 1975, the US reached a new high. Over the course of the following 34 years, 
    the United States set new records in every category. There are currently more over 1,550,000 prisoners in the US. The United States is home to 
    more than a quarter of all prisoners in the globe.

    2. According to the U.S. Department of Education, state and local government spending on jails and prisons (which is not included in this dataset) 
    has grown roughly three times as quickly as that on primary and secondary education over this time. Does the substantial investment in incarceration 
    increase public safety? To aid academics in examining this link, this dataset combines data on crime and incarceration.
    
#Data Exploration:

![image](https://user-images.githubusercontent.com/52540495/178807947-f273bdeb-0ab0-4058-9439-aead16061048.png)

<img width="233" alt="Screen Shot 2022-07-13 at 2 32 15 PM" src="https://user-images.githubusercontent.com/52540495/178805813-babc79c5-94f4-4408-898b-d3bd7e1d0d28.png">

#Exploratory Data Analysis:

The exploratory data analysis is performed on the crime and incarceration data of United States for 50 states from year 2001 to 2016 which 
contains 17 data field values and 816 rows of data having categorical, numerical, and boolean type data. The different parameters of the dataset 
are state, year, prisoner count, crimes estimated, violent crime total, rape legacy, and other various types of crime data. 

<img width="475" alt="image" src="https://user-images.githubusercontent.com/52540495/178806582-738304c6-e662-42e0-a2f1-4903e9560ec0.png">

<img width="508" alt="image" src="https://user-images.githubusercontent.com/52540495/178806599-be8b5e37-6d0c-4210-bb9d-39be99a6c2af.png">

#Data Profiling:

<img width="468" alt="image" src="https://user-images.githubusercontent.com/52540495/178806652-87bea27a-04d8-4e57-9b24-d186ee559c7e.png">

Here, we can observe that there are 871 missing values which is 6.3% missing cells in the dataset having no duplicate rows, which has 13 numerical data,
1 categorical data, and 3 boolean data values. 

#Data Cleaning:

<img width="284" alt="image" src="https://user-images.githubusercontent.com/52540495/178806681-a3954694-9d52-44c1-97a5-debaffb971e1.png">

 It is observed that the rape revised column has maximum of null values of the entire dataset and thus the column can be dropped out as it is not an 
 important feature that needs to be considered while the analysis of the data. Apart from that, the remaining columns have 17 null values which when 
 explored was observed that the same row values are missing for all those field values. Thus, the following methods are performed to clean the data.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/52540495/178806698-680be5e4-53be-4b63-a2a6-cd9d38d43124.png">

As mentioned, the rape revised column has maximum null values and is not an important feature for consideration, hence the field is dropped. 
Apart from this, for the 17 null values, since each of those fields have same null values for each row, all the 17 values are dropped which results 
to 799 rows of data and 16 field values. But as observed below, the rape legacy still has some missing values despite performing the mentioned 
cleaning steps. 

<img width="280" alt="image" src="https://user-images.githubusercontent.com/52540495/178808354-be862e51-96fe-463a-93bd-8bcf613c271a.png">

 In order to check for the rape legacy column of missing values, since the missing data in the field is approximately 20% or less, mean, median or mode 
 can be applied to the data column to impute the missing values. Now, the distribution plot for rape legacy is taken into consideration to check 
 if the data is normally distributed or the data is skewed which would help to determine the imputation method. 
 
 <img width="490" alt="image" src="https://user-images.githubusercontent.com/52540495/178808553-8bade3d0-4539-4c5f-ab70-1db28c797437.png">
 
 <img width="492" alt="image" src="https://user-images.githubusercontent.com/52540495/178808623-c3ecc2c2-f047-45c1-97db-388948735704.png">

 The distribution plot for aggravated assaults and murder manslaughter show that the data for both the field values is skewed. 
 The boxplot gives an understanding of the outliers present in the data values for prisoner count and violent crime field values as 
 shown in the below figure. As observed from the boxplot, we can see that there are outliers present in the field values of prisoner count and 
 violent crime total data.  
  
#Data Visulization: 

<img width="501" alt="image" src="https://user-images.githubusercontent.com/52540495/178806934-c054b16f-c88f-43d2-b839-030dda174527.png">

The next visual is the average prisoner count in the year which helps in understanding the prisoner count during the years from 2001 to 2016 and the previous year analysis with respect to the prisoner count would be important to analyze as this would help understand the data and information for the previous years in order to compare it with the future reports change. As observed, the highest average prisoner count of 29,895 is in the year 2009 and the lowest average prisoner count of 26,075 is in the year 2001. Also, after the year 2009 the average count of prisoners decreased until 2016.

Average Prisoner Count in the year: 

<img width="486" alt="image" src="https://user-images.githubusercontent.com/52540495/178806954-915130e2-50d4-4501-91f5-6da6414f0f80.png">

Murder Manslaughter per State: 

<img width="499" alt="image" src="https://user-images.githubusercontent.com/52540495/178806974-fffe595f-3693-492c-9475-b11d6fc85599.png">

The above visual represents the murder manslaughter per state which shows that Mississippi and Oklahoma have the highest count of murder manslaughter as compared to the other states and this analysis will be helpful in pointing to the researchers based on a particular crime type. As mentioned earlier, violent crime, murder manslaughter, and aggravated assault are the interested crime types which need to be considered and hence this is one of the analysis that would be further useful in the comparison of the crime stating reports.

#PreModeling:

#Correlation Plot 

<img width="468" alt="image" src="https://user-images.githubusercontent.com/52540495/178807067-e8a26b29-3394-490a-adca-a5f070894513.png">

The correlation plot helps to understand the correlation between each of the independent variables of the dataset and to plot the correlation values of the parameters to know the collinearity between the variables such that the variables with high collinearity value can be dropped in order to avoid multicollinearity and overfitting of the model. From the correlation plot it is observed that since there are different crime variables having the same value, there is a high collinearity that is existing between the variables with a correlation value of 0.9. This would result in multicollinearity and the model would outperform resulting in inaccurate model and results. Thus, some of the features having high collinearity are dropped before considering them for training of the model. 

#Model Building:

Linear regression is a basic predictive analytics approach that predicts an output variable using historical data. The core concept is that if we can fit a linear regression model to observed data, we can use it to predict future values. The implementation of the Linear Regressor Model consist of various features such as jurisdiction, includes_jails, state_population, violent_crime_total, murder_manslaughter, and agg_assault for the prediction of the prisoner count in each state. The accuracy obtained for both the training and testing set of the Linear Regression model is 92% and 91.01% respectively. Since this is a regressor type of model, the model evaluation is based on the MAE, MSE, RMSE and R-Squared values in order to determine the prediction error of the model implemented.

<img width="287" alt="image" src="https://user-images.githubusercontent.com/52540495/178807118-4fba86bb-88f1-49d6-bd30-f8f99444d9d6.png">

<img width="180" alt="image" src="https://user-images.githubusercontent.com/52540495/178807134-f98189c2-bb5f-4df2-a7cf-e76e67183f49.png">

#Decision Tree Regressor Model 

The independent variables considered for the implementation of the Decision Tree Regressor are 'jurisdiction', 'includes_jails', 'state_population', 'violent_crime_total', 'murder_manslaughter', and 'agg_assault'. The decision tree model is implemented with a train and test data split of 80-20 with a maximum branch depth of 5. The accuracy of the Decision Tree model obtained for the prediction of the prisoner count is 99% for both the training and testing set, which is a good accuracy overall, but machine learning models having a 99% accuracy is not effective. This is because the model is overfitted due to the multicollinearity of the parameters and passing less data values for the training of the model. The feature importance, and model evaluation for the Decision Tree Regressor model is as follows. 

<img width="333" alt="image" src="https://user-images.githubusercontent.com/52540495/178807180-6044a83a-3465-4082-9d27-63bb673d3b53.png">

<img width="223" alt="image" src="https://user-images.githubusercontent.com/52540495/178807192-07a4bfd9-8bda-4c33-afe1-06a80a03965d.png">

#Random Forest Regressor Model 

 For the prediction of the prisoner count, the features selected for the training of the model is same as that selected for the Linear Regression model and Decision Tree model. The data is split into 80-20 ratio for training and testing of the model where the Random Forest Regressor is implemented with a minimum of 5000 tress and maximum depth branch of 5. The accuracy of the model obtained is 99% for training data and 98.8% for test dataset. The feature importance and model evaluation for the random forest regressor model is as follows.
 
<img width="557" alt="image" src="https://user-images.githubusercontent.com/52540495/178807236-2d1421d6-1b36-467a-b134-cdfd5b0ca741.png">

#Regressor Model Comparison 

<img width="656" alt="Screen Shot 2022-07-13 at 2 41 37 PM" src="https://user-images.githubusercontent.com/52540495/178807477-f8a26094-b4e4-443e-9c47-353ca088700f.png">

Thus, based on the model evaluation and comparison, it is observed that the accuracy for Random Forest Regressor is more as compared to other two models. The MAE and MSE values represent the difference between the actual and predicted values extracted by the mean error over the dataset. The RMSE is the error rate where the R-Squared represents how well the value fits compared to the original values, and the higher the value of R-Squared the better the model is. Thus, based on the evaluation metrices, it is observed that the root mean squared error is less in Random Forest model which implies that the prediction rate error is less when compared it to the other models. Hence, Random Forest Regressor model would be recommended to use for the prediction of the prisoner count in each state.  


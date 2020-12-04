#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn import linear_model
import seaborn as sns
from scipy.stats import ttest_ind


# In[84]:


covid_data = pd.read_csv("Copy of COVID_Data.csv")
mobility_data = pd.read_csv("US_Mobility_Data.csv")


# First step is loading the data into python We will use both covid tracking data and Google's mobility data. Now we have to clean the data to get only entries that we want.

# In[85]:


covid_usa = covid_data[covid_data['location']=='United States']
covid_usa = covid_usa[['date','total_cases','new_cases','total_deaths','new_deaths','reproduction_rate']]

mobility_data = mobility_data.fillna(0)
mobility_data = mobility_data[mobility_data['sub_region_1']==0]
mobility_data = mobility_data[['country_region_code','country_region','sub_region_1',
                                   'date','retail_and_recreation_percent_change_from_baseline',
                                   'parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline',
                                   'workplaces_percent_change_from_baseline']]


# Here we have simplified the data into something that we can actually use. The COVID Data has been simplified to just contain values pertaining to the United States. We also took certain columns from both the COVID Data and Mobility Data to have only columns we wish to analyze. 

# Now we must merge the two dataframes together on the common column of date. 

# In[86]:


mobility_data = mobility_data.reset_index(drop=True)
covid_usa = covid_usa.reset_index(drop=True)
data_list = [covid_usa,mobility_data]
combined_data = covid_usa.append(mobility_data)
combined_data = combined_data.groupby(['date']).sum()
combined_data.fillna(0,inplace=True)
combined_data['Time']= np.arange(len(combined_data))
combined_data.head(300)


# Here is the combined dataset, on the column of date. Obviously all of the NaN values have been changed to zero, and the join was done as an outer. The next step would be to plot some scatters of our data to see how it is distributed. There was also a Time column added that counts the entries and is supposed to simulate date. The ultimate goal will be to try and see how well reproduction rate correlates with each of the mobility data, and then see if we can predict reproduction with the mobility. 

# In[87]:


plt.plot(combined_data['Time'],combined_data['reproduction_rate'])
plt.xlabel('Time (in days)')
plt.ylabel('Reproduction Rate')
plt.title('Reproduction Rate Versus Time')
plt.show()


# In[88]:


plt.plot(combined_data['Time'],combined_data['workplaces_percent_change_from_baseline'])
plt.xlabel('Time (in days)')
plt.ylabel('Workplace Mobility')
plt.title('Workplace Mobility Versus Time')
plt.show()


# In[89]:


plt.plot(combined_data['Time'],combined_data['transit_stations_percent_change_from_baseline'])
plt.xlabel('Time (in days)')
plt.ylabel('Transit Mobility')
plt.title('Transit Mobility Versus Time')
plt.show()


# In[90]:


plt.plot(combined_data['Time'],combined_data['retail_and_recreation_percent_change_from_baseline'])
plt.xlabel('Time (in days)')
plt.ylabel('Retail & Recreation Mobility')
plt.title('Retail & Recreation Mobility Versus Time')
plt.show()


# In[91]:


plt.plot(combined_data['Time'],combined_data['parks_percent_change_from_baseline'])
plt.xlabel('Time (in days)')
plt.ylabel('Park Mobility')
plt.title('Park Mobility Versus Time')
plt.show()


# While none of the mobility graphs map exactly the same as the reproduction rate of the virus; it can be observed that they are somewhat similar. What we can do now is run some correlations on the data, to see which of the mobility columns correlate the best with reproduction rate. 

# In[92]:


transit_corr = combined_data[['reproduction_rate','transit_stations_percent_change_from_baseline']].corr()
workplace_corr = combined_data[['reproduction_rate','workplaces_percent_change_from_baseline']].corr()
parks_corr = combined_data[['reproduction_rate','parks_percent_change_from_baseline']].corr()
retail_corr = combined_data[['reproduction_rate','retail_and_recreation_percent_change_from_baseline']].corr()


# In[93]:


sns.heatmap(transit_corr,annot=True)


# In[94]:


sns.heatmap(workplace_corr,annot=True)


# In[95]:


sns.heatmap(parks_corr,annot=True)


# In[96]:


sns.heatmap(retail_corr,annot=True)


# From the four correlation heatmaps above we can see that the highest correlation with reproduction rate occurs with transit mobility, or the movement of the population in the USA on average on public transportation. Now that we have this, it may be more effective for us to take the 7-Day average for each of these columns, then try to graph the two of them on the same plot. It also would help if we normalized the columns. 

# In[97]:


combined_data['Reproduction 7-Day Avg'] = combined_data['reproduction_rate'].rolling(window=7).mean()
combined_data['Transit Mobility 7-Day Avg'] = combined_data['transit_stations_percent_change_from_baseline'].rolling(window=7).mean()

cols_to_norm = ['Reproduction 7-Day Avg','Transit Mobility 7-Day Avg']
combined_data[cols_to_norm] = combined_data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# In[98]:


plt.plot('Time','Reproduction 7-Day Avg',data=combined_data,color='skyblue')
plt.plot('Time','Transit Mobility 7-Day Avg',data=combined_data,color='red')
plt.xlabel('Time (in days)')
plt.ylabel('Normalized Column Values')
plt.title('Comparison of 7-Day Average Reproduction Rate and Transit Mobility')
plt.legend()


# As you can see, the first step was to create two new columns: Reproduction Rate 7-Day Rolling Average and Transit Mobility 7-Day Rolling Average. Once the two columns were created using panda's pd.rolling function, setting the window to 7 for a 7-Day roll, and taking the mean, we now have to normalize the two columns to bring them to the same scale. mobility's range was much larger than the reproduction rate. Looking at the plot of the two lines it can be said that reproduction rate almost acts opposite of mobility in transits. The next step in our process is to create a linear regression model to hopefully be able to make a prediction of where the reproduction rate will go. After that we could try and average out all of the mobilities and see if that is a better predictor. But to start, we can make a model to predict reproductive rate using: transit mobility, new cases,total cases, and new deaths. 

# In[99]:


combined_data = combined_data.fillna(0)
predictors_1 = combined_data[['total_cases','new_cases','new_deaths','Transit Mobility 7-Day Avg']]
response_1 = combined_data['Reproduction 7-Day Avg']

regression_1 = linear_model.LinearRegression()
model_1 = regression_1.fit(predictors_1,response_1)

res1 = ttest_ind(predictors_1, response_1).pvalue

table_1 = {'Variable':['total_cases','new_cases','new_deaths','Transit Mobility 7-Day Avg'],
           'Coefficients':[model_1.coef_],
           'P-Values':[res1]
        
          }

print(
    'Variable:     ','total_cases    ','new_cases    ','new_deaths     ','Transit Mobility 7-Day Avg ' 
    '\nCoeficients: ',model_1.coef_,
    '\nP=Values:    ',res1,
    '\nIntercept: ',model_1.intercept_,
    '\nR-Squared: ',model_1.score(predictors_1,response_1))


# Off this first model we can see what our Beta values will be and the intercept. But it is clear that the R-Squared value is not too impressive, clocking in at only about 19.64%. This means that only 19.64% of the variation in reproduction rate is explained by these predictors. 

# There are three steps here to try and make our model better. First, we will make a model consisting solely of our transit mobility 7-Day Average predictor. Next, we will get the seven day averages of each of the mobility data columns and make models of that, and average them together and use that as a predictor. lastly, we will take the 7-Day average of new cases and new deaths. 

# In[100]:


predictors_2 = combined_data[['Transit Mobility 7-Day Avg']]
response_2 = combined_data['Reproduction 7-Day Avg']

regression_2 = linear_model.LinearRegression()
model_2 = regression_2.fit(predictors_2,response_2)

res2 = ttest_ind(predictors_2, response_2).pvalue

table_1 = {'Variable':['total_cases','new_cases','new_deaths','Transit Mobility 7-Day Avg'],
           'Coefficients':[model_2.coef_],
           'P-Values':[res2]
        
          }

print(
    'Variable:     ','total_cases    ','new_cases    ','new_deaths     ','Transit Mobility 7-Day Avg ' 
    '\nCoeficients: ',model_2.coef_,
    '\nP=Values:    ',res2,
    '\nIntercept: ',model_2.intercept_,
    '\nR-Squared: ',model_2.score(predictors_2,response_2))


# Above is a model that just consists of the sole predictor of 'Transit Mobility 7-Day Avg' to predict the Reproduction Rate 7-Day Average. It is apparent that getting rid of the other predictors did not have a detrimental affect on the R-Squared value, now coming in slightly lower than before at 0.1753. The next step would be to take the 7-Day averages of all other mobility columns, and using each of them as separate predictors for the reproduction rate. 

# In[101]:


combined_data['Park Mobility 7-Day Avg'] = combined_data['parks_percent_change_from_baseline'].rolling(window=7).mean()
combined_data['Workplace Mobility 7-Day Avg'] = combined_data['workplaces_percent_change_from_baseline'].rolling(window=7).mean()
combined_data['Retail Mobility 7-Day Avg'] = combined_data['retail_and_recreation_percent_change_from_baseline'].rolling(window=7).mean()
combined_data = combined_data.fillna(0)


# In[102]:


cols_to_norm_2 = ['Park Mobility 7-Day Avg','Workplace Mobility 7-Day Avg','Retail Mobility 7-Day Avg']
combined_data[cols_to_norm_2]=combined_data[cols_to_norm_2].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

combined_data['Mean Mobility'] = combined_data[['Park Mobility 7-Day Avg','Workplace Mobility 7-Day Avg','Retail Mobility 7-Day Avg','Transit Mobility 7-Day Avg']].mean(1)


# Above is the pre-processing for the mobility data so we can get each of the mobility's 7-Day averages, and also create another new column taking the average of all four normalized mobilities. Let's now create a plot with all four of the mobilities, and the average to see what that looks like. 

# In[103]:


plt.plot('Time','Park Mobility 7-Day Avg',data=combined_data,color='green')
plt.plot('Time','Retail Mobility 7-Day Avg',data=combined_data,color='blue')
plt.plot('Time','Workplace Mobility 7-Day Avg',data=combined_data,color='red')
plt.plot('Time','Transit Mobility 7-Day Avg',data=combined_data,color='skyblue')
plt.plot('Time','Mean Mobility',data=combined_data,color='black',linewidth=3)
plt.xlabel('Time (in days)')
plt.title('Plotted 7-Day Average Mobilities')
plt.legend(bbox_to_anchor=(0,0),loc="upper right")


# Now that we have all the desired columns, and have plotted all of the mobilities, we can now create another predictive model using all of the mobilities. 

# In[65]:


predictors_3 = combined_data[['Park Mobility 7-Day Avg','Retail Mobility 7-Day Avg','Transit Mobility 7-Day Avg','Workplace Mobility 7-Day Avg']]
response_3 = combined_data['Reproduction 7-Day Avg']

regression_3 = linear_model.LinearRegression()
model_3 = regression_3.fit(predictors_3,response_3)

res3 = ttest_ind(predictors_3, response_3).pvalue

table_1 = {'Variable':['total_cases','new_cases','new_deaths','Transit Mobility 7-Day Avg'],
           'Coefficients':[model_3.coef_],
           'P-Values':[res3]
        
          }

print(
    'Variable:     ','total_cases    ','new_cases    ','new_deaths     ','Transit Mobility 7-Day Avg ' 
    '\nCoeficients: ',model_3.coef_,
    '\nP=Values:    ',res3,
    '\nIntercept: ',model_3.intercept_,
    '\nR-Squared: ',model_3.score(predictors_3,response_3))


# Using all four of the mobilities in our model actually gives us the best R-Squared value thus far, clocking in at 23.18% roughly. The same process was followed as before to make this linear model. Give the predictors their own list from the data, giving the response its assigned column, and then using the LinearRegression() function. The next step would be to try and see how well the column averaging all of the mobilities together fairs in terms of creating an affecting model. 

# In[67]:


predictors_4 = combined_data[['Mean Mobility']]
response_4 = combined_data['Reproduction 7-Day Avg']

regression_4 = linear_model.LinearRegression()
model_4 = regression_4.fit(predictors_4,response_4)

res4 = ttest_ind(predictors_4, response_4).pvalue

table_1 = {'Variable':['total_cases','new_cases','new_deaths','Transit Mobility 7-Day Avg'],
           'Coefficients':[model_4.coef_],
           'P-Values':[res4]
        
          }

print(
    'Variable:     ','total_cases    ','new_cases    ','new_deaths     ','Transit Mobility 7-Day Avg ' 
    '\nCoeficients: ',model_4.coef_,
    '\nP=Values:    ',res4,
    '\nIntercept: ',model_4.intercept_,
    '\nR-Squared: ',model_4.score(predictors_4,response_4))


# 
# 
# 

# In[63]:


x_data = combined_data['Time'].values.reshape(-1,1)
y_data = combined_data['Reproduction 7-Day Avg'].values.reshape(-1,1)

linear_regressor = linear_model.LinearRegression()

linear_regressor.fit(x_data,y_data)
Y_Pred = linear_regressor.predict(x_data)

plt.scatter(x_data,y_data)
plt.plot(x_data,Y_Pred,color='red')
plt.show()


# Above is a linear regression thatpython best sees fit to the Reproduction Data. It is clear to see that even the best fit line concerning no other variables still has a great deal of variance around the line. This is due to the spike which could be attributed to the viral spread, then the sharp dip attributed to social distancing guidlelines. 

# In[ ]:





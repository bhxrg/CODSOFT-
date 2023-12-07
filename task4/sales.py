#!/usr/bin/env python
# coding: utf-8

# ##   Sales prediction-Codsoft task 4

# In[36]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# In[37]:


adv=pd.read_csv("C:\\Users\\prasanna\\OneDrive\\Desktop\\5thsem\\datasets(codsoft)\\advertising.csv")


# In[38]:


adv.head(5)


# In[39]:


#check for null values
adv.isnull().sum()


# In[40]:


adv.info()


# In[41]:


adv.describe()


# ### outlier detection using boxplot

# In[42]:


sns.boxplot(x='Radio', data=adv)
plt.show()


# In[43]:


sns.boxplot(x='TV', data=adv)
plt.show()


# In[44]:


sns.boxplot(x='Newspaper', data=adv)
plt.show()


# #### SINCE NEWSPAPER HAS OUTLIERS WE TREAT THE OUTLIERS

# In[45]:


# Outlier treatment with Z Score for 'newspaper' column in DataFrame 'adv'
threshold = 3
outlier = []

for i in adv['Newspaper']:
    z = (i - adv['Newspaper'].mean()) / adv['Newspaper'].std()
    if z > threshold:
        outlier.append(i)

print('Outliers in the Newspaper column:', outlier)
print('Minimum value of the outliers:', min(outlier))

# Calculate the median for the 'newspaper' column excluding outliers
median_newspaper = adv.loc[adv['Newspaper'] <= min(outlier), 'Newspaper'].median()
print('Median of the column excluding outliers:', median_newspaper)

# Replace outliers with median in 'newspaper' column
adv['out_treated_newspaper'] = adv['Newspaper'].apply(lambda x: median_newspaper if x >= min(outlier) else x)

print('Maximum value after outlier treatment:', max(adv['out_treated_newspaper']))

# Drop the original 'newspaper' column
adv.drop(['Newspaper'], axis=1, inplace=True)


# In[46]:


sns.boxplot(x='out_treated_newspaper', data=adv)
plt.show()


# #### train testing data 

# In[47]:


y=adv['Sales']
X=adv.drop('Sales', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
train_data = pd.concat([X_train,y_train], axis = 1)
test_data = pd.concat([X_test,y_test], axis = 1)

X_train = sm.add_constant(X_train)
model = sm.OLS(y_train,X_train)
reg_model = model.fit()
print(reg_model.summary())

### pretty accurate by seeing adj. R-square its 90% accurate


# In[48]:


# Assuming 'train_data' and 'test_data' are your original DataFrames with columns 'Sales', 'Newspaper', 'TV', and 'Radio'

# Selecting the independent variables for prediction
independent_vars = ['out_treated_newspaper', 'TV', 'Radio']

# Creating the training dataset for prediction
train_data_for_pred = train_data[independent_vars]
train_data_for_pred = sm.add_constant(train_data_for_pred)  # Add a constant for the intercept

# Predict using the regression model (reg_model2) assuming it's already trained
train_pred = reg_model.predict(train_data_for_pred)

# Creating the test dataset for prediction
test_data_for_pred = test_data[independent_vars]
test_data_for_pred = sm.add_constant(test_data_for_pred)  # Add a constant for the intercept

# Predict for the test dataset
test_pred = reg_model.predict(test_data_for_pred)


# In[49]:


# Calculating residuals for the training data
train_residuals = train_data['Sales'] - train_pred

# Calculating residuals for the test data
test_residuals = test_data['Sales'] - test_pred


# In[59]:


train_residuals = pd.DataFrame(train_residuals)
train_residuals.columns = ['Residuals']

plt.subplots(figsize=(12, 6))
ax = plt.subplot(111)  # To remove spines
plt.scatter(x=train_residuals.index, y=train_residuals.Residuals, alpha=0.5)
plt.plot(np.repeat(0, train_residuals.index.max()), color='darkorange', linestyle='--')
ax.spines['right'].set_visible(False)  # Removing the right spine
ax.spines['top'].set_visible(False)  # Removing the top spine
plt.title('Residuals')
plt.show() 


# In[64]:


## iteration1

X = adv[['TV','out_treated_newspaper']] 
Y = adv['Sales']

#Train Model
X = sm.add_constant(X) 
model = sm.OLS(Y, X).fit()
model_sum = model.summary()
print(model_sum)

#Sales Prediction
sales_pred = model.predict(X) 

#the adj r square has decreased than previous thus this cant be considered


# In[66]:


# iteration 2
X = adv[['TV','Radio']] 
Y = adv['Sales']

#Train Model
X = sm.add_constant(X) 
model = sm.OLS(Y, X).fit()
model_sum = model.summary()
print(model_sum)

#Sales Prediction
sales_pred = model.predict(X) 

# better than the previous iteration


# In[68]:


# iteration 3
X = adv[['out_treated_newspaper','Radio']] 
Y = adv['Sales']

#Train Model
X = sm.add_constant(X) 
model = sm.OLS(Y, X).fit()
model_sum = model.summary()
print(model_sum)

#Sales Prediction
sales_pred = model.predict(X) 

# worst than the previous iteration and original so not considering this


# In[75]:


# iteration 4
X = adv['Radio']
Y = adv['Sales']

#Train Model
X = sm.add_constant(X) 
model = sm.OLS(Y, X).fit()
model_sum = model.summary()
print(model_sum)

#Sales Prediction
sales_pred = model.predict(X) 


# In[76]:


# iteration 5
X = adv['TV']
Y = adv['Sales']

#Train Model
X = sm.add_constant(X) 
model = sm.OLS(Y, X).fit()
model_sum = model.summary()
print(model_sum)

#Sales Prediction
sales_pred = model.predict(X) 

# in TV and Radio TV drives Sales most 


# ### Radio & TV drives the Sales the most 

# In[60]:


'''White test :- H0: homoskedasticity Ha: heteroskedasticity'''

white_test = het_white(reg_model.resid,  reg_model.model.exog)
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, white_test)))


# ### data visualization

# In[69]:


plt.figure(figsize=(10,5))
sns.pairplot(adv)
plt.show()


# In[80]:


import seaborn as sns
import matplotlib.pyplot as plt
selected_columns = ['Sales', 'TV', 'Radio', 'out_treated_newspaper']
data_for_heatmap = adv[selected_columns]

# Compute the correlation matrix
correlation_matrix = data_for_heatmap.corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Heatmap: Sales, TV, Radio, and out_treated_newspaper')
plt.show()


# In[72]:


import matplotlib.pyplot as plt
sales = adv['Sales']
tv = adv['Radio']

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(tv, sales, color='skyblue')
plt.xlabel('Radio')
plt.ylabel('Sales')
plt.title('Bar Chart: Sales vs Radio')
plt.show()


# In[79]:


import matplotlib.pyplot as plt
sales = adv['Sales']
tv = adv['TV']

# Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.bar(tv, sales, color='skyblue')
plt.ylabel('TV')
plt.xlabel('Sales')
plt.title('Horizontal Bar Chart: Sales vs TV')
plt.show()



# In[ ]:





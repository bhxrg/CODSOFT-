#!/usr/bin/env python
# coding: utf-8

# ## Iris flower classification -Codsoft task 3

# In[45]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


# In[34]:


iris=pd.read_csv("C:\\Users\\prasanna\\OneDrive\\Desktop\\5thsem\\datasets(codsoft)\\IRIS.csv")


# In[35]:


iris.head()


# In[7]:


iris.info()


# In[8]:


#checking for null values
iris.isnull().sum()

#no null values thus we can proceed further


# In[9]:


iris.species.value_counts()


# In[10]:


iris.shape


# In[11]:


iris.describe()


# ### data visualisation

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.boxplot(x='species', y='sepal_length', data=iris, palette='Greens_d')
plt.show()
##there is an outlier in iris-virginica

plt.figure(figsize=(10, 5))
sns.boxplot(x='species', y='petal_length', data=iris, palette='Greens_d')
plt.show()
## there is an outlier in iris-versicolor and many in iris-setosa

plt.figure(figsize=(10, 5))
sns.boxplot(x='species', y='sepal_width', data=iris, palette='Greens_d')
plt.show()
## there are two outlier in iris-virginica

plt.figure(figsize=(10, 5))
sns.boxplot(x='species', y='petal_width', data=iris, palette='Greens_d')
plt.show()
## there is outlier in iris-setosa


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.set_palette("cubehelix")
sns.scatterplot(x="petal_length", y="petal_width", hue="species", size="species", sizes=(20, 200), data=iris)
plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Scatter Plot of Petal Length vs Petal Width')
plt.show()


# In[14]:


plt.figure(figsize=(10,5))
sns.pairplot(iris)
plt.show()


# In[42]:


#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
iris['species'] = lb_make.fit_transform(iris['species'])
iris.sample(3)


# In[44]:


import seaborn as sns
import pandas as pd

# Select the first 8 columns of the iris dataset
subset_iris = iris.iloc[:, :6]

# Compute the correlation matrix
correlation_matrix = subset_iris.corr()

# Create a correlation matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix for First 8 Columns of Iris Dataset')
plt.show()


# ### analysing and training data

# In[30]:


#dependent and independent variable seperation
y = iris['species']
X = iris.drop(['species'], axis=1)


# In[24]:


from sklearn.model_selection import KFold,train_test_split,cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[26]:


model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3)
# model = DecisionTreeClassifier()
model = model.fit(iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], iris.species)


# In[27]:


dt_feature_names = list(X.columns)
dt_target_names = [str(s) for s in y.unique()]
plt.figure(figsize = (10,8))
plot_tree(model,feature_names = dt_feature_names, class_names = dt_target_names, filled = True)

y_pred = model.predict(X)
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))


# In[29]:


print('accuracy is',accuracy_score(y,y_pred))


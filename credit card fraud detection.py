#!/usr/bin/env python
# coding: utf-8

# ## Credit Card fraud detection-Codsoft task 5

# In[39]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[40]:


card_data=pd.read_csv("C:\\Users\\prasanna\\OneDrive\\Desktop\\5thsem\\datasets(codsoft)\\creditcard.csv")


# In[41]:


card_data.head()


# In[42]:


card_data.info()


# In[43]:


card_data.describe()


# In[44]:


card_data.isnull().sum()


# In[45]:


card_data['Class'].value_counts()
# 0 for non fraudulent 1 for fraudulent


# In[46]:


card_data.groupby('Class').mean()


# In[47]:


sns.boxplot(x='Time', data=ccard_data)
plt.show()


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt

# Randomly select a sample of 1000 rows from the 'adv' DataFrame
sample_data = card_data.sample(n=1000, random_state=42)  # You can adjust the sample size as needed

# Create a boxplot for the 'Amount' column using the sampled data
plt.figure(figsize=(8, 6))
sns.boxplot(x='Amount', data=sample_data, color='skyblue')
plt.xlabel('Amount')
plt.title('Boxplot of Amount (Sampled Data)')
plt.show()


# In[50]:


df = card_data.copy() 


# ### data visualization

# In[52]:


import seaborn as sns
import matplotlib.pyplot as plt

def class_amount_graph(class_num):
    # Select the subset of data where "Class" is equal to class_num
    fraudulent_data = df[df["Class"] == class_num]

    # Set the style of the plot
    sns.set(style="whitegrid", palette="pastel")

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the first histogram
    sns.histplot(fraudulent_data["Amount"], bins=25, color='red', kde=True, ax=axes[0])
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Distribution of Amount for Class {class_num} (Fraudulent)")

    # Plot the second histogram
    sns.histplot(fraudulent_data["Amount"], bins=25, color='blue', kde=True, ax=axes[1])
    axes[1].set_xlabel("Amount")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Distribution of Amount for Class {class_num} (Non-Fraudulent)")

    # Adjust the layout to prevent overlapping
    plt.tight_layout()

    # Show the plots
    plt.show()

# Call the function for class 1 and class 0
class_amount_graph(1)
class_amount_graph(0)


# In[54]:


print(df.Class.value_counts())
print("\n")
# Portion of each class
print("The portion for not being fraud:", (df.Class.value_counts()[0] / len(df)) * 100, "\nThe portion for being fraud:", (df.Class.value_counts()[1] / len(df)) * 100)
plt.pie([df[df["Class"] == 1].shape[0], df[df["Class"] == 0].shape[0]], labels=["Fraud", "Not Fraud"], colors=["yellow", "red"])
plt.show


# In[55]:


import pandas as pd
import matplotlib.pyplot as plt

def top_10(class_num):
    # Get the top 10 amounts and their counts for the given class
    top_amounts = df[df["Class"] == class_num]["Amount"].value_counts().head(10)

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    top_amounts.plot(kind='bar', color='red')
    plt.title(f'Top 10 Amounts for Class {class_num} Transactions')
    plt.xlabel('Amount')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

top_10(1)
print("\n\n\n")
top_10(0)


# ### train and test data

# In[56]:


# seperate independent and dependent variables

y=df['Class']
X=df.drop(['Class'], axis=1)


# In[57]:


X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2)


# In[58]:


model=DecisionTreeClassifier()


# In[59]:


model.fit(X_train,y_train)
y_hat=model.predict(X_test)
accuracy_score(y_test,y_hat)


# In[60]:


print(classification_report(y_test,y_hat))


# In[61]:


print(confusion_matrix(y_test,y_hat))


# In[62]:


from sklearn.linear_model import LogisticRegression


# In[63]:


model=LogisticRegression()


# In[64]:


y=df['Class']
X=df.drop(['Class'], axis=1)


# In[65]:


X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2)


# In[66]:


pd.crosstab(y_test,y_hat)


# In[67]:


model.fit(X_train,y_train)
y_hat=model.predict(X_test)
accuracy_score(y_test,y_hat)


# ### Both have same accuracy !!!

# In[ ]:





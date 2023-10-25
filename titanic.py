#!/usr/bin/env python
# coding: utf-8

# ## Titanic survival prediction-Codsoft task 1

# In[160]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, classification_report


# In[127]:


titanic =pd.read_csv("C:\\Users\\prasanna\\OneDrive\\Desktop\\5thsem\\datasets(codsoft)\\tested.csv")


# In[128]:


titanic.head(6)


# In[129]:


titanic.isnull().sum()


# ### data visualization

# In[130]:


# visualizing survived and non survived passengers
plt.figure(figsize=(8, 6))
colors=['red','green','blue']
sns.countplot(data=titanic, x='Survived', palette="Set3")
plt.title("survival prediction")
plt.xlabel("Survival status")
plt.xticks([0, 1], ["Not Survived", "Survived"])
plt.ylabel("Census")
plt.show()



# In[131]:


#age group that has survived
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='Age', hue='Survived', bins=10, kde=True, palette="Set1")
plt.title("Age based Survival")
plt.xlabel("Age")
plt.ylabel("Census")
plt.legend(title='Survival Status', labels=['Not Survived', 'Survived'])
plt.show()


# In[132]:


#survival based on gender 
plt.figure(figsize=(8, 5))
sns.countplot(data=titanic, x='Sex', palette="Set1")
plt.title("gender based survival")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()


# In[133]:


#class distribution and survival
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Pclass', hue='Survived', palette="Set3")
plt.title("Survival based on P-Class")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()


# ### data cleaning

# In[134]:


#reducing the number of records that are non=important
titanic["Kin"]=titanic["SibSp"] + titanic["Parch"]
titanic.head(5)


# In[135]:


# Dropping non essential coclumn as it has too many null values and not essential for survival prediction
data=titanic.drop(["SibSp","Parch"], axis=1, inplace=False)
data.head(5)


# ### analysing data for training

# In[136]:


titanic.describe()


# In[137]:


titanic.groupby('Sex')[['Survived', 'Pclass', 'Age', 'Kin', 'Fare']].mean()


# In[138]:


titanic.Survived.value_counts()


# In[139]:


#splitting data into train and test
X = titanic.drop('Survived', axis = 1)
y = titanic['Survived']


# In[140]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
train_data = pd.concat([X_train,y_train], axis = 1)
test_data = pd.concat([X_test,y_test], axis = 1)


# In[141]:


train_data


# In[146]:


import statsmodels.api as sm


# In[149]:


formula = 'Survived ~ Pclass + Sex+ Age + SibSp + Parch + Fare + Embarked + Kin'
log_reg = sm.formula.logit(formula,data = train_data).fit()
log_reg.summary()


# In[152]:


pred_test = log_reg.predict(X_test)
test_data['test_pred_class'] = [1 if x >= 0.5 else 0 for x in pred_test]
pred_train = log_reg.predict(X_train)
train_data['train_pred_class'] = [1 if x >= 0.5 else 0 for x in pred_train]


# In[155]:


test_data_cm = confusion_matrix(test_data['Survived'], test_data['test_pred_class'])
train_data_cm = confusion_matrix(train_data['Survived'], train_data['train_pred_class'])
print('Train Data Confusion Matrix--> \n',train_data_cm )
print('Test Data Confusion Matrix--> \n',test_data_cm )


# In[156]:


## Train and test accuracy report

print('Train Data Accuracy Report--> \n',classification_report(train_data['Survived'], train_data['train_pred_class']))
print('Test Data Accuracy Report--> \n',classification_report(test_data['Survived'], test_data['test_pred_class']))


# In[161]:


# plot ROC Curve


fpr, tpr, thresholds = roc_curve(test_data['Survived'], test_data['test_pred_class'], pos_label = 1)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


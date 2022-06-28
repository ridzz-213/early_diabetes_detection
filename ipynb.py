#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd


# In[45]:


df = pd.read_csv(r"C:\Users\ridhi\Downloads\diabetes_data_upload.csv")
df


# In[46]:


#replace the values in dataframe
df=df.replace("No",0)
df=df.replace("Yes",1)

df=df.replace("Negative",0)
df=df.replace("Positive",1)

#consider gender column as ismale
df=df.replace("Male",1)
df=df.replace("Female",0)
df


# In[47]:


#check for missing values
df.isnull().sum()


# In[48]:


#check for dtypes of columns(int/float)
df.dtypes


# In[54]:


#change column name
replace={"gender":"isMale"}
df.rename(columns=replace)


# In[56]:


#make everything lowercase in columns
df.columns
df.columns=df.columns.str.lower()
df


# In[57]:


#export  dataframe to csv
df.to_csv(r"C:\Users\ridhi\Downloads\diabetes_data_upload.csv",index=None)
pd.read_csv(r"C:\Users\ridhi\Downloads\diabetes_data_upload.csv")


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt

#import statistics lib
from scipy.stats import chi2_contingency
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.weightstats import ztest


# In[36]:


#read data from csv to dataframe
df=pd.read_csv(r"C:\Users\ridhi\Downloads\diabetes_data_upload.csv")
df


# In[37]:


#examine age w histogram
plt.hist(df['age'])


# In[38]:


df['age'].mean()


# In[40]:


df['age'].median()


# In[61]:


#countplot for ismale(ratio of female to male)
sns.countplot(df['gender'])
plt.title('ismale')
sns.despine()


# In[63]:


columns=df.columns[1:]
columns


# In[64]:


#iteratively plot countplot
for column in columns:
    sns.countplot(df[column])
    plt.title(column)
    sns.despine()
    plt.show()


# In[ ]:


###Questions:
get_ipython().set_next_input('1.is obesity related to diabetes status');get_ipython().run_line_magic('pinfo', 'status')
get_ipython().set_next_input('2.is age related to diabetes status');get_ipython().run_line_magic('pinfo', 'status')


# In[67]:


obesity_diabetes=pd.crosstab(df['class'],df['obesity'])
obesity_diabetes


# In[68]:


chi2_contingency(obesity_diabetes)


# In[ ]:


#try:
1.polyuria vs class
2.ismale vs polyria


# In[70]:


#age is integer while class is categorical so we use boxplot
sns.boxplot(df['class'],df['age'])


# In[73]:


no_diabetes=df[df['class']==0]
no_diabetes['age'].median()


# In[74]:


diabetes=df[df['class']==1]
diabetes['age'].median()


# In[75]:


qqplot(df['age'],fit=True,line="s")
plt.show()


# In[77]:


#conduct z test of difference
ztest(diabetes['age'],no_diabetes['age'])


# In[78]:


#get a correlation plot
df.corr()


# In[79]:


sns.heatmap(df.corr())


# In[80]:


#import ML lib
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,classification_report


# In[83]:


#prepare independent and dependent variables
X=df.drop('class',axis=1)
y=df['class']
X
y


# In[84]:


#split data into train,test 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y)


# In[85]:


#begin our model training
#start with DummyClassifier to establish baseline
dummy=DummyClassifier()
dummy.fit(X_train,y_train)
dummy_pred=dummy.predict(X_test)


# In[86]:


#assess DummyClasisier model
confusion_matrix(y_test,dummy_pred)


# In[87]:


#use a classification report
print(classification_report(y_test,dummy_pred))


# In[89]:


#start w logisticregression
logr=LogisticRegression(max_iter=1000)
logr.fit(X_train,y_train)
logr_pred=logr.predict(X_test)


# In[90]:


confusion_matrix(y_test,logr_pred)


# In[91]:


#use a classification report
print(classification_report(y_test,logr_pred))


# In[99]:


#try decisiontree
tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)
tree_pred=tree.predict(X_test)


# In[100]:


confusion_matrix(y_test,tree_pred)


# In[101]:


print(classification_report(y_test,tree_pred))


# In[102]:


#try randomforest
forest=RandomForestClassifier()
forest.fit(X_train,y_train)
forest_pred=forest.predict(X_test)


# In[103]:


confusion_matrix(y_test,forest_pred)


# In[104]:


print(classification_report(y_test,forest_pred))


# In[105]:


forest.feature_importances_


# In[107]:


pd.DataFrame({'feature':X.columns,
             'importance':forest.feature_importances_}).sort_values('importance',ascending=False)


# In[ ]:





#to change working directory
import os

#to work with dataframes
import pandas as pd

#to perform numerical operations  
import numpy as np

#to visualise data 
import matplotlib.pyplot as plt

#to visualise data 
import seaborn as sns 

#to partition data 
from sklearn.model_selection import train_test_split

#importing library for Logistic Regression
from sklearn.linear_model import LogisticRegression

#importing performance metrics-accuracy score and confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

os.chdir("C:\\Users\chakr\Desktop")

data_income = pd.read_csv("income.csv") #reading data 

data = data_income.copy() #creating copy of data

#Exploratory data analysis
#1.Getting to know the data 
#2.Data preprocessing 
#3.cross tables and data visualization

print(data.info()) #to check variables datatypes 

data.isnull().sum() #checking for missing values
 
#summary of numerical variables 
summary_num = data.describe() 
print(summary_num)

#summary of categorical variables
summary_cate = data.describe(include="O")
print(summary_cate)

#frequency of each categories 
data['JobType'].value_counts() 
data['occupation'].value_counts()

#checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

#there exists ' ?' instead of nan
data=pd.read_csv('income.csv',na_values=[" ?"])

#data preprocessing
data.isnull().sum()

#axis=1, to consider atleast one column value is missing 
missing = data[data.isnull().any(axis=1)]
#1.Missing values in JobType = 1809
#2.Missing values in occupation = 1816
#3.There are 1809 rows where two specific columns in occupation and Jobtype have missing values
#4.(1816-1809)=7,you still havne occupation unfilled for these 7 rows. Because JobType is Never Worked

data2 = data.dropna(axis=0)

#relationship between independent variables
correlation = data2.corr()

#crosstables and data visualization 
#extracting column names
data2.columns

#Gender propotion table
gender = pd.crosstab(index=data2["gender"],columns='count',normalize=True)
print(gender)

#Gender vs Salary Status
gender_salstat = pd.crosstab(index=data2["gender"],columns=data2["SalStat"],margins=True,normalize='index')
print(gender_salstat)

#frequency distribution of Salary Status
SalStat = sns.countplot(data2['SalStat'])
#75% of people's Salary Status is <=50,000
#25% of people's Salary Status is >50,000

#Histogram of Age
sns.distplot(data2['age'],bins=10,kde=False)
#people with Age 20-45 are high in frequency

#Box-plot : Age vs Salary Status 
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()
#people with 35-50 are more likely to earn >50,000
#people with 25-35 age are more likely to earn <=50,000

#Bar-plot : JobType vs Salary Status 
sns.countplot(y="JobType",data=data2,hue="SalStat")

#crosstab : JobType vs Salary Status
pd.crosstab(index=data2["JobType"],columns=data2["SalStat"],dropna=True,margins=True,normalize='index')
#from the table it is visible that 56% of self employed people earn more than 50,000 USD per year 
#hence,an important variable in avoiding the ,misuse of subsidies

#Bar-plot : Education vs Salary Status
sns.countplot(y="EdType",data=data2,hue="SalStat")

#crosstab : Education vs Salary Status
pd.crosstab(index=data2["EdType"],columns=data2["SalStat"],normalize='index',margins=True,dropna=True)
#from the table we can see that people who have done Doctorate,Masters,Prof-school are more likely to earn above 50,000 USD per year compared with others. Hence an influencing variable in avoiding misuse of subsidies 

#Bar-plot : Occupation vs Salary Status
sns.countplot(y="occupation",data=data2,hue="SalStat")

#crosstab : Occupation vs Salary Status
pd.crosstab(index=data2["occupation"],columns=data2["SalStat"],normalize='index',margins=True,dropna=True)

#Histogram : Capital Gain
plt.hist(data2["capitalgain"],color="green")
#92%(27611) of capital gain is 0

#Histogram : Capital Loss
plt.hist(data2["capitalloss"],color="red")
#95%(28721) of the capital loss is 0

#Box and Whiskers plot : Salary Status vs Hours per week
sns.boxplot(x=data2["SalStat"],y=data2["hoursperweek"])
#those who spend 40-50 hours per week make more than 50,000 USD per year


#Logistic Regression : To predict the probability of categorical variable

#Reindexing the Salary Status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2,drop_first=True)

#Storing column names 
columns_list=list(new_data.columns)
print(columns_list)

#seperating the input names from data 
features=list(set(columns_list)-set(['SalStat']))
print(features)

#storing the output values in y
y=new_data["SalStat"].values
print(y)

#Storing the values from input features 
x=new_data[features].values
print(x)

#splitting the data into train and test 
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

#Make an instance of the model             
logistic=LogisticRegression(max_iter=10000)

#Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#prediction from test data 
prediction=logistic.predict(test_x)
print(prediction)

#confusion matrix
cfm=confusion_matrix(test_y,prediction)
print(cfm)

#calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

#printing the misclassified values from prediction
print('Misclassified samples : %d ' %(test_y!=prediction).sum())

#Logistic Regression - removing insignificant variables 

#Reindexing the Salary Status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)
new_data=pd.get_dummies(data2,drop_first=True)

#storing the column names
columns_list=list(new_data.columns)
print(columns_list)

#seperating the input names from data 
features=list(set(columns_list)-set(['SalStat']))
print(features)

#storing the output values in y
y=new_data["SalStat"].values
print(y)

#storing the values from input features
x=new_data[features].values
print(x)


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
logistic=LogisticRegression(max_iter=10000)
logistic.fit(train_x,train_y)
prediction=logistic.predict(test_x)
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
print('Misclassified samples : %d' %(test_y!=prediction).sum())


#importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

#storing the K nearest neighbors classifier 
KNN_Classifier=KNeighborsClassifier(n_neighbors=5)

#fitting the values for x and y
KNN_Classifier.fit(train_x,train_y)

#predicting the test values with model
prediction=KNN_Classifier.predict(test_x)

#performance metric check
cfm=confusion_matrix(test_y,prediction)
print("\t","Predicted values")
print("Original values","\n",cfm)

#calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
print('Misclassified samples : %d' %(test_y!=prediction).sum())

#Effect of K value on classifier
Misclassified_sample=[]

#calculating error for K values between 1 to 20

for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    predict_i=knn.predict(test_x)
    Misclassified_sample.append((test_y!=predict_i).sum())
    print(Misclassified_sample)

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import choices

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#Count number of NaN values in a column
for col in train_data.columns:
    print(col," ",pd.isnull(train_data[col]).sum())

#Socio-Economic Class of passenger(Upper,Middle,Lower)
sns.barplot(x='Pclass',y='Survived',data=train_data)
#plt.show()
print(train_data['Pclass'].value_counts())

#Number of parents and children of the passenger aboard
sns.barplot(x='Parch',y='Survived',data=train_data)
#plt.show()
print(train_data['Parch'].value_counts())

#Number of siblings and spouses of the passenger aboard
sns.barplot(x='SibSp',y='Survived',data=train_data)
#plt.show()
print(train_data['SibSp'].value_counts())

#Port Of Embarkation("C","Q","S")
sns.barplot(x='Embarked',y='Survived',data=train_data)
#plt.show()
print(train_data['Embarked'].value_counts())

#Turn Age into category
def process_age(data_df,cut_pts,label_names):
    data_df["Age"].fillna(-0.5,inplace=True)
    data_df['Age_Group'] = pd.cut(data_df["Age"], cut_pts, labels=label_names)
    return data_df

cut_pts = [-1,0,5,12,18,24,35,60,110]
label_names = ['Unknown', 'Infant', 'Child', 'Teenager', 'Young Adult', 'Middle Aged', 'Adult', 'Senior']

train_data = process_age(train_data,cut_pts,label_names)
test_data = process_age(test_data,cut_pts,label_names)

sns.barplot(x="Age_Group",y="Survived",data=train_data)
#plt.show()

#Cabin and Ticket are base less,and leds to false prediction,so drop both of them
test_data.drop(['Cabin','Ticket','Name'],axis=1,inplace=True)
train_data.drop(['Cabin','Ticket','Name'],axis=1,inplace=True)

print(pd.isnull(train_data).sum())
print()
print(pd.isnull(test_data).sum())

#filling port of embarkment("Embarked") with "S"
common_value="S"
train_data['Embarked'].fillna(common_value,inplace=True)

#fixing age by random generating age based on train data
age_cnt = train_data['Age_Group'].value_counts()
age_cnt.drop(['Unknown'],axis=0,inplace=True)
pb = age_cnt/age_cnt.sum()

for x in range(len(test_data['Age_Group'])):
    if(test_data['Age_Group'][x]=='Unknown'):
        test_data["Age_Group"][x] = choices(pb.index.values,pb.values)

for y in range(len(train_data['Age_Group'])):
    if(train_data['Age_Group'][y] == 'Unknown'):
        train_data["Age_Group"][y] = choices(pb.index.values,pb.values)

train_data.drop(['Age'],axis=1,inplace=True)
test_data.drop(['Age'],axis=1,inplace=True)

#converting string to numeric values
def get_dummies(attributes):
    sv = train_data[attributes][train_data['Survived'] == 1].value_counts().sort_index()
    total = train_data[attributes].value_counts().sort_index()

    tt_ind = test_data[attributes].value_counts().index
    for i in tt_ind:
        if i not in sv.index:
            sv[x] = 0
    for i in total.index:
        if i not in sv.index:
            sv[x] = 0

    pb_mapping = (sv/(total)).to_dict()

    for ind in pb_mapping:
        if math.isnan(pb_mapping[ind]):
            pb_mapping[ind]=0

    train_data[attributes] = train_data[attributes].map(pb_mapping)
    test_data[attributes] = test_data[attributes].map(pb_mapping)

get_dummies('Pclass')
get_dummies('Sex')
get_dummies('Age_Group')
get_dummies('Embarked')
get_dummies('SibSp')
get_dummies('Parch')

#Replacing single missing fare with mean
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

#Standard Scaling
#Scales the data so that it has mean 0 and variance 1
#Removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_data['Fare'] = sc.fit_transform(train_data[['Fare']])
test_data['Fare'] = sc.fit_transform(test_data[['Fare']])

from sklearn.model_selection import train_test_split as tts

x = train_data.drop(['PassengerId','Survived'],axis=1)
y = train_data['Survived']

x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2,random_state=0)

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.metrics import accuracy_score as acc

gaussian = gnb()
gaussian.fit(x_train,y_train)
y_pred = gaussian.predict(x_test)
acc_gn = round(acc(y_pred,y_test)*100,2)
print("Accuracy Using Naive Bayes ",acc_gn)

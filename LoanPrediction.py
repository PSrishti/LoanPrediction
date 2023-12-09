# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix , classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#Data Gathering

df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')

#Data PreProcessing

def handleNullValues(df_train):
    NullColumns = df_train.isna().sum()/len(df_train)
    
    CategColumns = ['Gender','Married','Dependents','Education','Self_Employed','Loan_Amount_Term','Credit_History','Property_Area']
    ContColumns = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
    
    for col in CategColumns:
        df_train[col].fillna(value=df_train[col].mode().iloc[0], inplace=True)
    for col in ContColumns:
        df_train[col].fillna(value=df_train[col].mean(), inplace=True)

def encodingCategColumns(df_train):
    CategColumns = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
     
    New_df=pd.get_dummies(df_train, columns = CategColumns)
    label_encoder = LabelEncoder()
    New_df['Loan_Amount_Term']= label_encoder.fit_transform(New_df['Loan_Amount_Term'])
    df_train.Loan_Amount_Term.value_counts()
    New_df.Loan_Amount_Term.value_counts()                        
    
    #Label Encode target column
    New_df['Loan_Status']= label_encoder.fit_transform(New_df['Loan_Status'])
    return New_df

handleNullValues(df_train)
New_df = encodingCategColumns(df_train)

#Reviewing statistical parameters
New_df.describe()
New_df.corr()    
    
#Splitting the train and test data
x = New_df.iloc[:, ~New_df.columns.isin(['Loan_ID','Loan_Status'])].values            
y = New_df.iloc[:, 5].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=0)

#Building classification models


#1 : Logistic Regression classifier
lrclf = LogisticRegression(random_state=0)
#Implementing cross validation
cv = ShuffleSplit(n_splits=5, test_size=0.3, n_splits = 5, random_state=0)
shufflescore = cross_val_score(lrclf, x_train, y_train, cv=cv)
scores = cross_val_score(lrclf, x_train, y_train, cv=KFold(n_splits=5))
score = cross_val_score(lrclf, x_train, y_train, cv=StratifiedKFold(n_splits=5))
rmse_score = np.sqrt(-score)
#Model fitting
lrclf.fit(x_train[:3], y_train[:3])
lrclf.predict(x_test)
lrclf.score(x_test,y_test)    
#Hyperparameter Tuning
clf_GS = GridSearchCV(lrclf, cv=5, param_grid={'C' : [1.5,1.75,2] , 'max_iter' : [800,1000,1200], 'class_weight' : [None,'balanced',0.25], 'tol' : [0.00001,0.0001,0.01], 'penalty' : ["l1","l2"]})
clf_GS.fit(x_train, y_train)
print("tuned hpyerparameters :(best parameters) ",clf_GS.best_params_)
print("accuracy :",clf_GS.best_score_)



#2 : Random Forest classifier
rfclf = RandomForestClassifier(max_depth=2, random_state=0)
rfclf.fit(x_train, y_train)    
rfclf.predict(x_test)
rfclf.score(x_test,y_test)  

#3 : SVM - Linear/RBF
lsvclf = make_pipeline(StandardScaler(), svm.LinearSVC(dual="auto", random_state=0, tol=1e-5))
lsvclf.fit(x_train, y_train)
lsvclf.predict(x_test)
lsvclf.score(x_test,y_test)  

rbfsvclf = svm.SVC(kernel = 'rbf')
rbfsvclf.fit(x_train, y_train)
rbfsvclf.predict(x_test)
rbfsvclf.score(x_test,y_test) 
    
#4 : KNN
knnclf = KNeighborsClassifier(n_neighbors=50)
knnclf.fit(x_train, y_train)
knnclf.predict(x_test)
knnclf.score(x_test,y_test)  

#5 : Decision Tree Classifier
dtclf = DecisionTreeClassifier(random_state=0)
dtclf.fit(x_train, y_train)
dtclf.predict(x_test)
dtclf.score(x_test,y_test)  

#6 : Ada-Boost
adbclf = AdaBoostClassifier(n_estimators=100, random_state=0)
adbclf.fit(x_train, y_train)
adbclf.predict(x_test)
adbclf.score(x_test,y_test)  

#7 : XgBoost
xgclf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
xgclf.fit(x_train, y_train)
xgclf.predict(x_test)
xgclf.score(x_test,y_test)  

#8 : Naive Bayes
nbclf = GaussianNB()
nbclf.fit(x_train, y_train)
nbclf.predict(x_test)
nbclf.score(x_test,y_test)  

#9 : Gaussian Process classifier  
kernel = 1.0 * RBF(1.0)
gpclf = GaussianProcessClassifier(kernel=kernel, random_state=0) 
gpclf.fit(x_train, y_train)
gpclf.predict(x_test)
gpclf.score(x_test,y_test)      

#10 : Voting classifier
vclf = VotingClassifier(estimators=[('lr', lrclf), ('rf', rfclf), ('adb', adbclf), ('xgb', xgclf)], voting='hard')
vclf.fit(x_train, y_train)
vclf.predict(x_test)
vclf.score(x_test,y_test) 













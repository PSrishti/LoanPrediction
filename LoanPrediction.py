import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix , classification_report, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns

#Data Gathering

df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')

#Data PreProcessing

def handleNullValues(df_train):
    NullColumns = df_train.isna().sum()/len(df_train)
    
    CategColumns = df_train.select_dtypes(include=["object"]).columns.tolist()
    ContColumns = list(set(df_train.columns) - set(CategColumns))
    
    for col in CategColumns:
        df_train[col].fillna(value=df_train[col].mode().iloc[0], inplace=True)
    for col in ContColumns:
        df_train[col].fillna(value=df_train[col].mean(), inplace=True)

def encodingCategColumns(x,y=None):
    CategColumns = x.select_dtypes(include=["object"]).columns.tolist() # Gets the list of all columns with data type 'object' - means categorical data
    
    #xnew=pd.get_dummies(x, columns = CategColumns)
    transformer = ColumnTransformer(
                transformers=[("ohe", OneHotEncoder(), CategColumns)],
                remainder="passthrough",
            )
    xnew = transformer.fit_transform(x)
    
    #Label Encode target column for train data
    label_encoder = LabelEncoder()
    if y is not None and y.dtype == 'O': #For test data, y is not present so takes default value none, and for categorical y - y.dtype = "O"
        ynew = label_encoder.fit_transform(y)
        return xnew, ynew
    else:
        return xnew

#Handling null values of data
handleNullValues(df_train)
handleNullValues(df_test)

#Dropping column Loan_ID since it is not contributing to the target anyway
x = df_train.iloc[:, ~df_train.columns.isin(['Loan_ID','Loan_Status'])]          
y = df_train.iloc[:, -1]

x,y = encodingCategColumns(x,y)
xdf_test = encodingCategColumns(df_test.iloc[:,1:]) #Dropping Loan_ID from test data too

#Scaling the data
trans = MinMaxScaler()
x = trans.fit_transform(x)
xdf_test = trans.fit_transform(xdf_test)

#Splitting train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=0)

#Reviewing statistical parameters
#New_df.describe()
#New_df.corr()    
         

'''.................Building classification models....................'''


'''1 : Logistic Regression classifier'''

lrclf = LogisticRegression(random_state=0)
#Hyperparameter Tuning
clf_GS = GridSearchCV(lrclf, cv=5, param_grid={'solver' : ['liblinear'] , 'C' : [0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1] , 'max_iter' : [100,500,800], 'class_weight' : ['balanced',None], 'tol' : [0.00001,0.000001], 'penalty' : ["l1","l2"]})
clf_GS.fit(x_train, y_train)
print("tuned hpyerparameters :(best parameters) ",clf_GS.best_params_)
print("accuracy :",clf_GS.best_score_)
#tuned hpyerparameters :(best parameters)  {'C': 0.001, 'class_weight': 'balanced', 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear', 'tol': 1e-05}
#accuracy : 0.8021739130434783

#Create StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
pred_test_full =0
cv_score =[]
i=1
for train_index,test_index in stratified_kfold.split(x_train, y_train):
    #print(train_index, test_index)
    print('{} of KFold {}'.format(i,stratified_kfold.n_splits))
    xtr,xvl = x_train[train_index],x_train[test_index]
    ytr,yvl = y_train[train_index],y_train[test_index]
    
    #model
    lrclf = LogisticRegression(random_state=0, C=0.001, solver="liblinear", class_weight="balanced", tol= 0.0001, penalty="l2")
    lrclf.fit(xtr,ytr)
    score = roc_auc_score(yvl,lrclf.predict(xvl))
    print('ROC AUC score:',score)
    cv_score.append(score)    
    pred_test = lrclf.predict_proba(x_test)[:,1]
    pred_test_full +=pred_test
    i+=1
    
lrclf.predict(x_test)
lrclf.score(x_test,y_test)    #0.77

'''2 : SGD classifier'''

#Hyperparameter Tuning
hyperparams = dict(loss = ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
penalty = ['l1', 'l2', 'elasticnet'],
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive'],
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
eta0 = [1, 10, 100])
#Using GridSearchCV
sgdclf = GridSearchCV(SGDClassifier(),hyperparams,scoring="accuracy",cv=5,n_jobs=-1)
sgdclf.fit(x_train, y_train)
print("tuned hpyerparameters :(best parameters) ",sgdclf.best_params_)
print("accuracy :",sgdclf.best_score_)
#tuned hpyerparameters :(best parameters)  {'alpha': 0.0001, 'class_weight': {1: 0.6, 0: 0.4}, 'eta0': 10, 'learning_rate': 'adaptive', 'loss': 'log_loss', 'penalty': 'l2'}, accuracy : 0.80
#Using RandomizedSearchCV
random = RandomizedSearchCV(estimator=SGDClassifier(), param_distributions=hyperparams, scoring='roc_auc', verbose=1, n_jobs=-1, n_iter=1000)
random_result = random.fit(x_train, y_train)
print('Best Score: ', random_result.best_score_)
print('Best Params: ', random_result.best_params_)
#Best Params:  {'penalty': 'elasticnet', 'loss': 'modified_huber', 'learning_rate': 'adaptive', 'eta0': 1, 'class_weight': {1: 0.6, 0: 0.4}, 'alpha': 0.1}, best score : 0.75

#implementing k-fold cross validation with k=5 for the best params 
sgd_classifier = SGDClassifier(alpha=0.0001, class_weight={1: 0.6, 0: 0.4}, eta0=10, learning_rate='adaptive', loss='log_loss', penalty='l2',max_iter=1000, random_state=42)

# Initialize a list to store accuracy scores for each fold
accuracy_scores = []

# Perform stratified k-fold cross-validation
for train_index, test_index in stratified_kfold.split(x_train, y_train):
    X_tr, X_val = x_train[train_index], x_train[test_index]
    y_tr, y_val = y_train[train_index], y_train[test_index]

    # Fit the model on the training data
    sgd_classifier.fit(X_tr, y_tr)

    # Make predictions on the test data
    y_pred = sgd_classifier.predict(X_val)

    # Calculate accuracy and store it
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_scores.append(accuracy)

# Print the accuracy for each fold
for fold, accuracy in enumerate(accuracy_scores, start=1):
    print(f"Fold {fold}: Accuracy = {accuracy:.2f}")

# Print the average accuracy across all folds
average_accuracy = np.mean(accuracy_scores)
print(f"\nAverage Accuracy: {average_accuracy:.2f}")
#Average accuracy : 0.80


'''3 : SVM - Linear/RBF'''

lsvclf = svm.LinearSVC(dual="auto", random_state=0, tol=1e-5)
lsvclf.fit(x_train, y_train)
lsvclf.predict(x_test)
lsvclf.score(x_test,y_test)  #0.837

rbfsvclf = svm.SVC(kernel = 'rbf')
rbfsvclf.fit(x_train, y_train)
rbfsvclf.predict(x_test)
rbfsvclf.score(x_test,y_test)  #0.831

# Create SVC model
svc_model = svm.SVC()
#Hyperparametetr tuning
# Define hyperparameters for tuning including 'kernel'
param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf'],
              'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(svc_model, param_grid, cv=stratified_kfold)
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
# Train the model with the best hyperparameters
best_svc_model = svm.SVC(**best_params)
best_svc_model.fit(x_train, y_train)
# Make predictions on the test set
y_pred = best_svc_model.predict(x_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") #0.83
# Print the best hyperparameters
print("Best Hyperparameters:", best_params)
#{'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
    
'''4 : KNN'''

knnclf = KNeighborsClassifier(n_neighbors=50)
knnclf.fit(x_train, y_train)
knnclf.predict(x_test)
knnclf.score(x_test,y_test)  #0.74

# Create KNN model
knn_model = KNeighborsClassifier()

# Define hyperparameters for tuning
param_grid = {'n_neighbors': [3, 5, 7, 9, 20, 30],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(knn_model, param_grid, cv=stratified_kfold)
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
# Print the best hyperparameters
print("Best Hyperparameters:", best_params)
#Best Hyperparameters: {'algorithm': 'auto', 'n_neighbors': 9, 'weights': 'distance'}
# Train the model with the best hyperparameters
best_knn_model = KNeighborsClassifier(**best_params)
best_knn_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = best_knn_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") #0.81


'''5 : Decision Tree Classifier'''

dtclf = DecisionTreeClassifier(random_state=0)
dtclf.fit(x_train, y_train)
dtclf.predict(x_test)
dtclf.score(x_test,y_test) #0.747

# Create Decision Tree model
dt_model = DecisionTreeClassifier()

# Define hyperparameters for tuning
param_grid = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(dt_model, param_grid, cv=stratified_kfold)
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
# Print the best hyperparameters
print("Best Hyperparameters:", best_params)
#Best Hyperparameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'splitter': 'random'}
# Train the model with the best hyperparameters
best_dt_model = DecisionTreeClassifier(**best_params)
best_dt_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = best_dt_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") #0.81


'''6 : Random Forest classifier'''

rfclf = RandomForestClassifier(max_depth=2, random_state=0)
rfclf.fit(x_train, y_train)    
rfclf.predict(x_test)
rfclf.score(x_test,y_test)  #0.83

# Create Random Forest model
rf_model = RandomForestClassifier()

# Define hyperparameters for tuning
param_grid = {'n_estimators': [50, 100, 150],
              'criterion': ['gini', 'entropy'],
              'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': [None, 'sqrt', 'log2']}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(rf_model, param_grid, cv=stratified_kfold)
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
# Print the best hyperparameters
print("Best Hyperparameters:", best_params)
#Best Hyperparameters: {'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 100}
# Train the model with the best hyperparameters
best_rf_model = RandomForestClassifier(**best_params)
best_rf_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = best_rf_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  #0.81


'''7 : Ada-Boost'''

adbclf = AdaBoostClassifier(n_estimators=100, random_state=0)
adbclf.fit(x_train, y_train)
adbclf.predict(x_test)
adbclf.score(x_test,y_test)  #0.845

# Create AdaBoost model with DecisionTreeClassifier as base estimator
base_estimator = DecisionTreeClassifier(random_state=42)
adaboost_model = AdaBoostClassifier(estimator=base_estimator, random_state=42)

# Define hyperparameters for tuning
param_grid = {
    'base_estimator__max_depth': [1, 2, 3],  # Maximum depth of base estimator
    'n_estimators': [50, 100, 150],          # Number of weak learners
    'learning_rate': [0.01, 0.1, 0.5]        # Learning rate
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(adaboost_model, param_grid, cv=stratified_kfold)
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
#Best Hyperparameters: {'base_estimator__max_depth': 1, 'learning_rate': 0.01, 'n_estimators': 50}
# Train the model with the best hyperparameters
#best_adaboost_model = AdaBoostClassifier(estimator=base_estimator, **best_params)
best_adaboost_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=best_params['base_estimator__max_depth']),
                                         n_estimators=best_params['n_estimators'],
                                         learning_rate=best_params['learning_rate'],
                                         random_state=42)
best_adaboost_model.fit(x_train, y_train)

# Make predictions on the test dataset 
y_pred = best_adaboost_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") #0.81

# Create AdaBoost model with SVC as base estimator
base_estimator = svm.SVC(probability=True, kernel='linear', random_state=42)
adaboost_model = AdaBoostClassifier(estimator=base_estimator, random_state=42)

# Define hyperparameters for tuning
param_grid = {
    'base_estimator__C': [0.1, 1, 10],          # Regularization parameter for SVC
    'n_estimators': [50, 100, 150],                # Number of weak learners
    'learning_rate': [0.01, 0.1, 0.5]              # Learning rate
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(adaboost_model, param_grid, cv=stratified_kfold)
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
#Best Hyperparameters: {'base_estimator__C': 0.1, 'learning_rate': 0.01, 'n_estimators': 50}
# Train the model with the best hyperparameters
best_adaboost_model = AdaBoostClassifier(estimator=svm.SVC(C=best_params['base_estimator__C'],probability=True, kernel='linear', random_state=42),
                                         n_estimators=best_params['n_estimators'],
                                         learning_rate=best_params['learning_rate'],
                                         random_state=42)
best_adaboost_model.fit(x_train, y_train)

# Make predictions on the entire dataset (this is just for demonstration purposes)
y_pred = best_adaboost_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") #0.81


'''8 : XgBoost'''

xgclf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
xgclf.fit(x_train, y_train)
xgclf.predict(x_test)
xgclf.score(x_test,y_test)  #0.83

# Define XGBoost parameters (some of these will be tuned)
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.2],
    'reg_lambda': [1, 1.1, 1.2],
    'scale_pos_weight': [1, 2, 3],  # Adjust based on class imbalance
}

# Create XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, n_jobs=-1)

# Perform GridSearchCV for hyperparameter tuning with early stopping
grid_search = GridSearchCV(xgb_model, param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=4)
grid_search.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=True)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
#Best Hyperparameters: {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 3, 'n_estimators': 50, 'reg_alpha': 0.2, 'reg_lambda': 1, 'scale_pos_weight': 2, 'subsample': 0.8}
# Train the model with the best hyperparameters
best_xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, **best_params, n_jobs=-1)
best_xgb_model.fit(x_train, y_train, eval_set=[(x_train, y_train)], early_stopping_rounds=10, verbose=True)

# Make predictions on the entire dataset (this is just for demonstration purposes)
y_pred = best_xgb_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") #0.83


'''9 : Naive Bayes'''

nbclf = GaussianNB()
nbclf.fit(x_train, y_train)
nbclf.predict(x_test)
nbclf.score(x_test,y_test)  #0.825

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'var_smoothing': np.logspace(0, -9, num=10)  # Adjust the range as needed
}

# Create Gaussian Naive Bayes model
nb_model = GaussianNB()

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(nb_model, param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
#Best Hyperparameters: {'var_smoothing': 0.1}

# Train the model with the best hyperparameters
best_nb_model = GaussianNB(var_smoothing=best_params['var_smoothing'])
best_nb_model.fit(x_train, y_train)

# Make predictions on the entire dataset (this is just for demonstration purposes)
y_pred = best_nb_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") #0.83


'''10 : Gaussian Process classifier'''
  
kernel = 1.0 * RBF(1.0)
gpclf = GaussianProcessClassifier(kernel=kernel, random_state=0) 
gpclf.fit(x_train, y_train)
gpclf.predict(x_test)
gpclf.score(x_test,y_test)      #0.8311

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'kernel': [1.0 * RBF(length_scale=1.0), 2.0 * RBF(length_scale=1.0), 1.0 * RBF(length_scale=2.0)]  # Adjust the kernel choices
}

# Create Gaussian Process Classifier model
gp_model = GaussianProcessClassifier()

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(gp_model, param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
#Best Hyperparameters: {'kernel': 1**2 * RBF(length_scale=1)}

# Train the model with the best hyperparameters
best_gp_model = GaussianProcessClassifier(kernel=best_params['kernel'])
best_gp_model.fit(x_train, y_train)

# Make predictions on the entire dataset (this is just for demonstration purposes)
y_pred = best_gp_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") #0.83


'''11 : Voting classifier'''

vclf = VotingClassifier(estimators=[('lr', lrclf), ('rf', rfclf), ('adb', adbclf), ('xgb', xgclf)], voting='hard')
vclf.fit(x_train, y_train)
vclf.predict(x_test)
vclf.score(x_test,y_test) #0.844

# Initialize lists to store evaluation metrics across folds
accuracy_scores = []
classification_reports = []
confusion_matrices = []

# Iterate over folds
for train_index, test_index in stratified_kfold.split(x_train, y_train):
    # Split the data into training and testing sets for the current fold
    x_tr, x_val = x_train[train_index], x_train[test_index]
    y_tr, y_val = y_train[train_index], y_train[test_index]

    # Address class imbalance by computing sample weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_tr)

    # Train the ensemble on the training data with sample weights
    vclf.fit(x_tr, y_tr, sample_weight=sample_weights)

    # Make predictions on the test data
    predictions = vclf.predict(x_val)

    # Calculate and store the accuracy score for the current fold
    accuracy_scores.append(accuracy_score(y_val, predictions))

    # Print additional metrics for the current fold
    classification_reports.append(classification_report(y_val, predictions))
    confusion_matrices.append(confusion_matrix(y_val, predictions))

# Print average accuracy score across folds
print(f"Average Accuracy: {sum(accuracy_scores) / 5:.2f}")

# Print average classification report across folds
print("Average Classification Report:")
print("\n".join(classification_reports))

# Print average confusion matrix across folds
print("Average Confusion Matrix:")
print(sum(confusion_matrices) // 5)
















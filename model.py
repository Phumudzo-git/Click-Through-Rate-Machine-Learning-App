import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
#%matplotlib inline
from collections import Counter
from collections import defaultdict
from sklearn import tree
import pickle

file = "./train"
data = pd.read_csv(file)

data

data.columns

data.dtypes

data["click"].value_counts()
data['site_category'].value_counts()
data['site_domain'].value_counts()
data["site_category"].value_counts()
data["app_id"].value_counts()
data["app_domain"].value_counts()
data["app_category"].value_counts()
data["device_model"].value_counts()
data["device_id"].value_counts()
data["device_ip"].value_counts()
data.describe()

d_device = pd.concat([data['device_type'], data["click"]],axis=1)

sns.countplot(x='device_type', data=d_device)
d_device.loc[:,'device_type'].value_counts()

device_type_table = pd.crosstab(data['device_type'], data["click"])
device_type_table

device_type_table.plot(kind='bar',figsize=(5,5),stacked=False)

data_correlation = data.corr(method='pearson')
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data_correlation, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})

data.isnull().sum().sort_values(ascending=False)

print(data.shape)
data = data.drop_duplicates(keep = 'first')
print(data.shape)

#Removing null entries
data.dropna()
print(data.shape)

# Function to map categorical variables (dtypes objects) into numerical representation
def map_function(column):
    map_dictionary = defaultdict(int)
    for i, j in enumerate(data[column].unique()):
        map_dictionary[j]=i+1
    return(map_dictionary)

data["device_model"]=data["device_model"].map(map_function("device_model"))
data["site_domain"]=data["site_domain"].map(map_function("site_domain"))
data["app_id"]=data["app_id"].map(map_function("app_id"))
data["app_domain"]=data["app_domain"].map(map_function("app_domain"))
data["app_category"]=data["app_category"].map(map_function("app_category"))

print(data)

data["app_domain"].value_counts()

# Droping the id, site_id, site_category, device_id columns since they contain categorical variables with too many categories.
# Using pd.get_dummies to categorize them will require more computation power and will also create too many columns.

data = data.dropna()
X = data[['ui_component_position', 'device_type','device_model','site_domain',"app_id","app_domain", "app_category",
       'device_conn_type', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']]
y = data.click.values

print(X)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Oversampling the under representated class by bringing the ratio of classes to 1:2

print(Counter(y))

oversample = RandomOverSampler(sampling_strategy=0.5)
#undersample = RandomUnderSampler(sampling_strategy='majority')
X_over, y_over = oversample.fit_resample(X, y)
print(Counter(y_over))

import sklearn.model_selection

# Split data into 80% training and 20% testing

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X_over, y_over, test_size=0.2, random_state=5)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

### We will first use the simple decision tree classifier model to model the data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

y_predict = dtc.predict(x_test)

# Use the accuracy to measure the performance of the model
accuracy_score(y_test, y_predict)

"""
The accuracy score is about 77%, which is a good score. However the accuracy metric is not a good measure for 
the model performance, since there is class imbalance. The model can be bias and mostly predict the over 
represented class and the accuracy score of the model will still be high. A good metric to measure the 
performance of the model in this instance is the AUC (Area under the Curve) metric and also the 
AUC-ROC (Area Under the ROC Curve). A good classifier model should have an AUC-score which is close
to 1, and bad classifier model which randomly classifies (poor classifier) should have an AUC-score 
close to 0.5. Also the confusion matrix and f1-score are good metrics to use, since they can both show
you how well the model is able to predict the under represented class. An f1-score close to 1 also
indicates a good classifier model, this shows that the model is able to fairly balance the prediction
of both the binary classes (under represented and over represented classes)

"""

from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict)
auc_1 = auc(false_positive_rate, true_positive_rate)
print("Area Under the Curve",auc_1)

y_prob = dtc.predict_proba(x_test)[:,1]
roc_auc_score_1 = roc_auc_score(y_test, y_prob)
print("Area Under the ROC-Curve", roc_auc_score_1)

pd.DataFrame(
   confusion_matrix(y_test, y_predict),
   columns = ['Predicted not clicked', "predicted_clicked"],
   index = ['Actual not clicked', "Actual clicked"]
)

print(classification_report(y_test, y_predict))

#tree.plot_tree(decision_tree_best)
#plt.show()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[5,6,7,8,9, 10,11,12,13,14,15, 20, 50, 100, None], 
              "criterion":["gini", "entropy"]}

dtc = DecisionTreeClassifier()
grid_search = GridSearchCV(dtc, parameters)
grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

decision_tree_best = grid_search.best_estimator_

y_predict = decision_tree_best.predict(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict)

grid_auc_socre = auc(false_positive_rate, true_positive_rate)
print("Area Under the Curve",grid_auc_socre)

y_predict_proba = decision_tree_best.predict_proba(x_test)[:,1]
grid_auc = roc_auc_score(y_test, y_predict_proba)
print("Area Under the ROC Curve",grid_auc)

print("accuracy",accuracy_score(y_test, y_predict))

print(classification_report(y_test, y_predict))

text_model_architecture = tree.export_text(decision_tree_best)
print(text_model_architecture)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=1)
rfc.fit(x_train,y_train)

y_predict = rfc.predict(x_test)
print("accuracy score:",accuracy_score(y_test, y_predict))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict)
rfc_roc_auc = auc(false_positive_rate, true_positive_rate)
print("Area Under the Curve:",rfc_roc_auc)

pd.DataFrame(
   confusion_matrix(y_test, y_predict),
   columns = ['Predicted not clicked', "predicted_clicked"],
   index = ['Actual not clicked', "Actual clicked"]
)

print(classification_report(y_test, y_predict))


n_estimators = [100, 300, 500, 800, 1000]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

parameters = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

rfc = RandomForestClassifier()
grid_search = GridSearchCV(rfc, parameters, verbose = 1, 
                      n_jobs = -1)
grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

random_forest_best = grid_search.best_estimator_

y_predict = random_forest_best.predict(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict)

rfc_auc_roc = auc(false_positive_rate, true_positive_rate)
print("Area Under the Curve:",rfc_auc_roc)

y_predict_proba = decision_tree_best.predict_proba(x_test)[:,1]
grid_auc = roc_auc_score(y_test, y_predict_proba)
print("Area Under the ROC curve:",grid_auc)

print("accuracy score:",accuracy_score(y_test, y_predict))

print(classification_report(y_test, y_predict))


#Bayesian optimization for decision tree classifier

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score



def objective_function(params):
    X_ = x_train[:]
    
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = preprocessing.scale(X_)
            del params['scale']
    model = DecisionTreeClassifier(**params)
    return(cross_val_score(model, X_, y_train).mean())

serach_space_for_model = {'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
#    'scale': hp.choice('scale', [0, 1]),
#    'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    global best
    
#    clf = DecisionTreeClassifier(*params)
#    acc = cross_val_score(clf, x_train, y_train).mean()
    acc = objective_function(params)
    if acc > best:
         best = acc
    return({'loss': -acc, 'status': STATUS_OK})

best = 0
trials = Trials()
best = fmin(f, search_space_for_model, algo=tpe.suggest, max_evals=100, trials=trials)
print('best:', best)

print(best)

#Fitting the model using best parameters
best_dtr = DecisionTreeClassifier(max_depth=int(best['max_depth']), 
                                  criterion="entropy", 
                                  max_features=int(best["max_features"]))

best_dtr.fit(x_train,y_train)

y_predict = best_dtr.predict(x_test)
print("accuracy-score", accuracy_score(y_test, y_predict))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predict)
dtr_auc_4 = auc(false_positive_rate, true_positive_rate)
print("Area Under the Curve", dtr_auc_4)

y_prob = best_dtr.predict_proba(x_test)[:,1]
dtr_roc_auc_score_4 = roc_auc_score(y_test, y_prob)
print("Area Under the ROC-Curve", dtr_roc_auc_score_4)

pd.DataFrame(
   confusion_matrix(y_test, y_predict),
   columns = ['Predicted not clicked', "predicted_clicked"],
   index = ['Actual not clicked', "Actual clicked"]
)

print(classification_report(y_test, y_predict))


## Save the random forest classifier model
filename = 'random_forest.sav'
pickle.dump(rfc, open(filename, 'wb'))
rfc_model = pickle.load(open(filename, 'rb'))





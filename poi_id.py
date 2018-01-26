#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import helper_functions
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# this feature was not used as it is nearly 100% correlated with total_stock_value
decision_tree_features_list = ['poi', 'salary',
                                 'bonus',
                                 'expenses',
                                 'long_term_incentive',
                                 'restricted_stock_deferred',
                                 'from_messages',
                                 'total_stock_value',
                                 'restricted_stock',
                                 'shared_receipt_with_poi',
                                 'to_messages']


features_not_used = ['exercised_stock_options',
                     'shared_receipt_with_poi',
                     'to_messages',
                     ]

features_list = ['poi', 'salary', 'director_fees','bonus', 'expenses',
                    'loan_advances', 'long_term_incentive', 'other',
                    'total_payments',
                    'deferral_payments', 'deferred_income',
                     'total_stock_value',
                    'restricted_stock','restricted_stock_deferred',
                    'from_messages',
                     'from_poi_to_this_person', 'from_this_person_to_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
## during the investigation, a "total" row was found and is removed here.
del data_dict['TOTAL']

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

###############################################################################
# fed up with dictionary i continue with pandas
data_pd = pd.DataFrame(data_dict)
data_pd = data_pd.transpose()
data_pd.fillna(value=0, inplace=True)
data_pd.replace(to_replace='NaN', value=0, inplace=True)
original_pd = data_pd.copy()
original_pd['deferred_income_x_exercised_stock_options'] = data_pd.apply(
        lambda row:(row['deferred_income']*row['exercised_stock_options']),
        axis = 1)
original_pd['deferred_income_x_total_stock_value'] = data_pd.apply(
        lambda row:(row['deferred_income']*row['total_stock_value']),
        axis = 1)
original_pd['exercised_stock_options_x_deferral_payments'] = data_pd.apply(
        lambda row:(row['exercised_stock_options']*row['deferral_payments']),
        axis = 1)
original_pd.fillna(value=0, inplace=True)
new_feature_names = ['deferred_income_x_exercised_stock_options',
                     'deferred_income_x_total_stock_value',
                     'exercised_stock_options_x_deferral_payments']

# comment these two lines for experiments without three additional features
features_list = features_list + new_feature_names
data_dict = original_pd.transpose().to_dict()
###############################################################################


my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#  I prefer the data in numpy for the speed and named "X, Y" for compatibilty
# with the sklearn examples
X = np.array(features)
Y = np.array(labels)

# due to the massive number of warnings during training, i turn them off.
# The warnings are about 0 predicted samples - a problem that is corrected
# by choosing only the most performant algorithms in the end.
import warnings
warnings.filterwarnings("ignore")

# define a splitting strategy that is in line with the tester code.
# It will be reused in each GridSearch.
from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(Y, 1000, random_state=42)
scoring = 'f1' # score with f1 to have both, precision and recall
best_k = len(features_list)-1 # use all features
X_in = X # separating original and worked-on data for K-Best and PCA feature selection


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# final parameters after lots of steps of refinment

# AdaBoost first.
# class_weight, criterion and splitter each were found by gridsearch.
# n_estimators didn't converge - there was a different result each time.
# Along with the fact that it is delivering no good results, this is a hint
# that this algorithm isn't best suited for the problem.
'''
dtc = DecisionTreeClassifier(class_weight='balanced',
                             criterion = 'entropy',
                             splitter = 'random')
clf = AdaBoostClassifier(base_estimator=dtc)
param_grid = {'n_estimators' : [ 25, 30, 35, 100, 150]}
grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring)
grid = grid.fit(X_in, Y)

clf = grid.best_estimator_
print(clf)
'''

# SVC next
# C, gamma and kernel were all found by GridSearch. class_weight was set;
# though it appears in the param grid because i was experimenting with manual
# weight settings. After it always converged on "balanced" i removed those
# manual weight settings.
# Parameters inside pipelines can be set using __ separated parameter names:
param_grid = {'svc__C' : [600, 620],
    'svc__gamma': [ 0.005, 0.01,],
    'svc__kernel':['sigmoid'],
    'svc__class_weight':['balanced']}

# make a scaler for a pipeline.
svc_scaler = preprocessing.StandardScaler()
pipe = Pipeline(steps=[('svc_scaling', svc_scaler),
                      ('svc', SVC())])
grid = GridSearchCV(pipe,
                   param_grid,
                   scoring = scoring,
                   cv=cv,
                   refit=True)
grid = grid.fit(X, Y)
estimation = ['SVC:'+scoring+'= ', grid.best_score_]
print(estimation)

# save the best estimator
clf = grid.best_estimator_







#RandomForest
'''
features = list(range(2, len(features_list)-1))
param_grid = {'n_estimators':[5,10,15],
              'min_samples_leaf':[1, 3],
              'class_weight':['balanced'],
              'criterion':['gini', 'entropy'],
              'max_features':features}
rfc =  RandomForestClassifier()
grid = GridSearchCV(rfc,
                   param_grid,
                   scoring = scoring,
                   cv=cv, refit=True)
grid = grid.fit(X_in, Y)
rfc = grid.best_estimator_
print(rfc)
clf = rfc
'''








# finally, DecisionTreeClassifier
'''
dtc = DecisionTreeClassifier()
features = list(range(2, len(features_list)-1))

param_grid = {'criterion':['gini', 'entropy'],
              'splitter':['best', 'random'],
              'min_samples_split':[2,3,4,5],
              'max_features':features}
grid = GridSearchCV(DecisionTreeClassifier(),
                   param_grid,
                   scoring = scoring,
                   cv=cv,
                   refit=True)
grid = grid.fit(X, Y)
estimation = ['Decision Tree', grid.best_score_]
print(estimation)
print(grid.best_estimator_)
clf = grid.best_estimator_
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
import tester
tester.execute()

# reload function needed only in case of changes to the helper_functions module
def reload():
    import importlib
    importlib.reload(helper_functions)

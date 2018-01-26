# -*- coding: utf-8 -*-
"""
Special copy of the poi_id code for mass-testing with different numbers of
features and different algorithms
"""

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

features_list_all = ['poi', 'salary', 'bonus', 'director_fees',  'expenses',
                 'loan_advances', 'long_term_incentive', 'other',
                 'total_payments',
                 'deferral_payments', 'deferred_income',
                 'exercised_stock_options', 'total_stock_value',
                 'restricted_stock', 'restricted_stock_deferred',
                 'shared_receipt_with_poi', 'to_messages', 'from_messages',
                 'from_poi_to_this_person', 'from_this_person_to_poi']

features_list_reduced = ['poi', 'salary', 'bonus', 'expenses',
                 'loan_advances', 'long_term_incentive', 'other',
                 'total_payments',
                 'total_stock_value',
                 'restricted_stock',
                 'shared_receipt_with_poi', 'to_messages',
                 'from_poi_to_this_person', 'from_this_person_to_poi']

features_list_used = ['poi', 'salary', 'bonus', 'expenses',
                 'loan_advances', 'long_term_incentive', 'other',
                 'restricted_stock_deferred',
                'deferred_income',
                'director_fees',
                'deferral_payments',
                'from_messages',
                 'total_payments',
                 'total_stock_value',
                 'restricted_stock',
                 'shared_receipt_with_poi', 'to_messages',
                 'from_poi_to_this_person', 'from_this_person_to_poi']

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


features_list = features_list_used
#features_list = decision_tree_features_list



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
## during the investigation, a "total" row was found and is removed here.
del data_dict['TOTAL']

### Task 3: Create new feature(s)

# Three additional features, sometimes used
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

features_list = features_list + new_feature_names
data_dict = original_pd.transpose().to_dict()


my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#  I prefer the data in numpy for the speed and named "X, Y" for compatibilty
# with the sklearn examples
X_original = np.array(features)
Y_original = np.array(labels)

##########################################################################
# see Featuring II in the project report.
# This was a short investigation if i can generate additional information with
# artificially generated additional "dumb" features: substractions and
# multiplications of every feature with every feature.
# they are stored in a "lots_X" array that can be used in the algorithms.

'''
lots_data_dict = helper_functions.more_features(data_dict)
features_list_all = lots_data_dict.columns.tolist()
features_list_all.remove('poi')
features_list_all.remove('email_address')
features_list_all = ['poi'] + features_list_all  # to place poi at the right spot
lots_data_dict = lots_data_dict.transpose().to_dict()
lots_data = featureFormat(lots_data_dict, features_list_all, sort_keys = True)
lots_Y, lots_X = targetFeatureSplit(lots_data)
lots_X = np.array(lots_X)

# find the best artificial features
kb = SelectKBest(k=5)
kb.fit(lots_X, lots_Y)
for i, choice in enumerate(kb.get_support()):
    if choice:
        print(features_list_all[i])
'''
# Most of this set of features finally is not used, because the results were getting
# far worse with those additional features.
###############################################################################

# due to the massive number of warnings during training, i turn them off.
# The warnings are about 0 predicted samples - a problem that is corrected
# by choosing only the most performant algorithms in the end.
import warnings
warnings.filterwarnings("ignore")


clf_estimation = pd.DataFrame(columns=["algo", "selection", "nb_features", "F1-result"])

X_in = X_original
Y = Y_original

# define a splitting strategy that is in line with the tester code.
# It will be reused in each GridSearch.
from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(Y, 1000, random_state=42)

scoring = 'f1'

#for i in [3, 5, 8, 15, 18, len(features_list)-1]:
'''
pca_scaler = preprocessing.StandardScaler().fit(X_in)
pca_raw = PCA(n_components=i)
pca = Pipeline(steps=[('scaling', pca_scaler),('pca', pca_raw)])

kb = SelectKBest(k=i)
for j in range(2):
    if j==0:
        X = pca.fit_transform(X_in)
        selection_text = "PCA"
    else:
        X = kb.fit_transform(X_in, Y)
        selection_text = "SelectKBest"
'''

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
'''
dtc = DecisionTreeClassifier(class_weight='balanced',
                             criterion = 'entropy',
                             splitter = 'random')
clf = AdaBoostClassifier(base_estimator=dtc)
param_grid = {'n_estimators' : [ 25, 30, 35, 100, 150]}
grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring)
grid = grid.fit(X_in, Y)

#estimation = ['AdaBoost', selection_text, i, clf.best_score_]
#print(estimation)
#clf_estimation.loc[len(clf_estimation)] = estimation
ada = grid.best_estimator_
dump_classifier_and_data(ada, my_dataset, features_list)
#print(clf.best_estimator_)
# clf = clf.best_estimator_

# choosing svc parameters from a wide and unprecise grid
# original parameter grid
'''
# first param grid
'''
param_grid = {'C' : [1, 2, 5,
       1e1, 2e1, 5e1,
       1e2, 2e2, 5e2,
       1e3, 2e3, 5e3, 5e4, 5e5, 5e6 ],
    'gamma': [0.0001, 0.001, 0.005, 0.01, 0.02, 0.1, 0.8],
    'kernel':['rbf', 'sigmoid']}
'''
# parameter grid after several steps of refinment
# note, the "class weight" appears...
'''
param_grid = {'C' : [ 550, 600, 650, 700, ],
    'gamma': [ 0.005, 0.01,],
    'kernel':['sigmoid'],
    'class_weight':[None,
                    'balanced',
                    {True:1, False:1},
                    {True:1, False:0.9},
                    {True:1, False:0.8},
                    {True:1, False:0.7},
                    {True:1, False:0.6},
                    {True:1, False:0.5},
                    {True:1, False:0.4},
                    {True:1, False:0.3},
                    {True:1, False:0.2},
                    {True:1, False:0.15},
                    {True:1, False:0.1},
                    {True:1, False:0.05}]}
'''
# final parameter grid
param_grid = {'svc__C' : [ 600, 610 ,620],
    'svc__gamma': [ 0.001, 0.005, 0.01, 0.05],
    'svc__kernel':['sigmoid', 'rbf'],
    'svc__class_weight':['balanced',{True:1, False:0.05}]}

# save the scaler for a pipeline.
pca_scaler = preprocessing.StandardScaler()
pca_raw = PCA(n_components=10)
svc_scaler = preprocessing.StandardScaler()

#pipe = Pipeline(steps=[('pca_scaling', pca_scaler),
#                      ('pca', pca_raw),
#                      ('svc_scaling', svc_scaler),
#                      ('svc', SVC())])

# without PCA it works better
pipe = Pipeline(steps=[('svc_scaling', svc_scaler),
                      ('svc', SVC())])
grid = GridSearchCV(pipe,
                   param_grid,
                   scoring = scoring,
                   cv=cv,
                   refit=True)
grid = grid.fit(X_in, Y)
svc = grid.best_estimator_
print(svc) # see params chosen
# save the best estimator
dump_classifier_and_data(svc, my_dataset, features_list)

'''
features = list(range(2, len(features_list)-1))
param_grid = {'n_estimators':[5,10,15],
              'min_samples_leaf':[1, 3],
              'class_weight':['balanced'],
              'criterion':['gini', 'entropy'],
              'max_features':features}
rfc =  RandomForestClassifier()
rfc = GridSearchCV(rfc,
                   param_grid,
                   scoring = scoring,
                   cv=cv, refit=True)
rfc = rfc.fit(X_in, Y)

#estimation = ['Random Forest Classifier', selection_text, i, rfc.best_score_]
#print(estimation)
#clf_estimation.loc[len(clf_estimation)] = estimation
print(rfc.best_estimator_)
clf = rfc.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list)




dtc = DecisionTreeClassifier()
features = list(range(2, len(features_list)-1))

param_grid = {'class_weight':['balanced'],
              'criterion':['gini', 'entropy'],
              'splitter':['best', 'random'],
              'min_samples_split':[2,3,4,5],
              'max_features':features}
dtc = GridSearchCV(dtc,
                   param_grid,
                   scoring = scoring,
                   cv=cv,
                   refit=True)
dtc = dtc.fit(X_in, Y)
#estimation = ['Decision Tree', selection_text, i, dtc.best_score_]
#print(estimation)
#clf_estimation.loc[len(clf_estimation)] = estimation
print(dtc.best_estimator_)
#clf = dtc.best_estimator_

# get important features - may be interesting to test with the other estimators!
from itertools import compress
list(compress(features_list, dtc.best_estimator_.feature_importances_ > 0))
###########################
dump_classifier_and_data(dtc.best_estimator_, my_dataset, features_list)
'''
import tester
tester.execute()

#clf_estimation.to_excel('Random_forest_estimation.xls')

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

#dump_classifier_and_data(clf, my_dataset, features_list)


def reload():
    import importlib
    importlib.reload(helper_functions)

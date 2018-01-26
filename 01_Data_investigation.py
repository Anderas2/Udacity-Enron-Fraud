# -*- coding: utf-8 -*-
"""
This is a partial copy from the poi_id code. It contains mainly display code
for the data investigationd done during different project phases.

@author: tha2w1
"""
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import helper_functions
import pandas as pd
import numpy as np

features_list = ['poi', 'salary', 'bonus', 'director_fees', 'expenses',
                    'loan_advances', 'long_term_incentive', 'other',
                    'total_payments',
                    'deferral_payments', 'deferred_income','exercised_stock_options',
                     'total_stock_value',
                    'restricted_stock', 'restricted_stock_deferred',
                    'shared_receipt_with_poi', 'to_messages', 'from_messages',
                    'from_poi_to_this_person', 'from_this_person_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
X = np.array(features)
helper_functions.print2d(X, 0, 1)

# delete total line to improve correlations plus to remove outlier
del data_dict['TOTAL']
data = featureFormat(my_dataset, features_list, sort_keys = True)
helper_functions.heat(np.corrcoef(data, rowvar=False), titles=features_list)

# was used to identify the least poi correlated features
print("the feature that is the least correlated with POI")
features_list[abs(np.corrcoef(data, rowvar=False)[0]).argmin()]
data_df = pd.DataFrame(data_dict)
data_df = data_df.transpose()

print('total number of rows:', data_df.count())
print(data_df[data_df=="NaN"].count(axis=0))

print ('total number of POI:', data_df.poi.sum())



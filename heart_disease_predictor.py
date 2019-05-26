import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.load_data import load_hear_disease_data

# As usual we have our features and we have our targets
# This is going to be a supervised machine learning problem.
# First we wil train our model based on training data
# Then we will test our model


# Let's load data first


# features_df = pd.read_csv(os.path.relpath('datasets/train_values.csv'))
# labels_df = pd.read_csv(os.path.relpath('datasets/train_labels.csv'))
#
# training_df = pd.merge(left=features_df, right=labels_df, on='patient_id', how='inner')
#
# features = ['slope_of_peak_exercise_st_segment', 'resting_blood_pressure',
#             'chest_pain_type', 'num_major_vessels',
#             'fasting_blood_sugar_gt_120_mg_per_dl', 'resting_ekg_results',
#             'serum_cholesterol_mg_per_dl', 'oldpeak_eq_st_depression', 'sex', 'age',
#             'max_heart_rate_achieved', 'exercise_induced_angina']
#
# label_col = ['heart_disease_present']
#
# X = training_df[features].as_matrix()
# y = training_df[label_col].as_matrix()
# std_scaler = StandardScaler()
# X = std_scaler.fit_transform(X)
# y = y.flatten()

ret = load_hear_disease_data(os.path.relpath('datasets/train_values.csv'), os.path.relpath('datasets/train_labels.csv'))
X, y = ret.samples, ret.targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

print("Training set score:{}".format(model.score(X_train, y_train)))
print("Testing set score:{}".format(model.score(X_test, y_test)))

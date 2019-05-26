import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from models.load_data import load_hear_disease_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# As usual we have our features and we have our targets
# This is going to be a supervised machine learning problem.
# First we wil train our model based on training data
# Then we will test our model


# Let's load data first

ret = load_hear_disease_data(os.path.relpath('datasets/train_values.csv'), os.path.relpath('datasets/train_labels.csv'))
X, y = ret.samples, ret.targets
feature_labels = ret.feature_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
print("Training set score:{}".format(svm_model.score(X_train, y_train)))
print("Testing set score:{}".format(svm_model.score(X_test, y_test)))

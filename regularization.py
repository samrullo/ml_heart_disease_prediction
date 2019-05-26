import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.load_data import load_hear_disease_data
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

ret = load_hear_disease_data(os.path.relpath('datasets/train_values.csv'), os.path.relpath('datasets/train_labels.csv'))
X, y = ret.samples, ret.targets
feature_labels = ret.feature_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression()
searcher = GridSearchCV(lr_model, {'C': [0.01, 0.1, 1, 10]})

searcher.fit(X_train, y_train)
print("Best searcher parameters:{}".format(searcher.best_params_))
print("Best score:{}".format(searcher.best_score_))

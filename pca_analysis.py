import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.load_data import load_hear_disease_data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ret = load_hear_disease_data(os.path.relpath('datasets/train_values.csv'), os.path.relpath('datasets/train_labels.csv'))
X, y = ret.samples, ret.targets
feature_labels = ret.feature_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = PCA(n_components=3)

X_train_transformed = model.fit_transform(X_train, y_train)
X_test_transformed = model.transform(X_test)

plt.bar(range(model.n_components_), model.explained_variance_)
plt.title("Explained variances of PCA model")
plt.xlabel("features")
plt.xticks(rotation=60)
plt.ylabel("explained variance")
plt.show()

log_model = LogisticRegression()
log_model.fit(X_train_transformed, y_train)
print("Training score using 3 components only:{}".format(log_model.score(X_train_transformed, y_train)))
print("Test score using 3 components:{}".format(log_model.score(X_test_transformed, y_test)))

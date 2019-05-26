import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.simple_numpy_based import NumpyBinaryClassifier
import logging

logging.basicConfig()
_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)
# As usual we have our features and we have our targets
# This is going to be a supervised machine learning problem.
# First we wil train our model based on training data
# Then we will test our model


# Let's load data first

features_df = pd.read_csv(os.path.relpath('datasets/train_values.csv'))
labels_df = pd.read_csv(os.path.relpath('datasets/train_labels.csv'))

training_df = pd.merge(left=features_df, right=labels_df, on='patient_id', how='inner')

features = ['slope_of_peak_exercise_st_segment', 'resting_blood_pressure',
            'chest_pain_type', 'num_major_vessels',
            'fasting_blood_sugar_gt_120_mg_per_dl', 'resting_ekg_results',
            'serum_cholesterol_mg_per_dl', 'oldpeak_eq_st_depression', 'sex', 'age',
            'max_heart_rate_achieved', 'exercise_induced_angina']

label_col = ['heart_disease_present']

std_scaler = StandardScaler()

X = training_df[features].as_matrix()
y = training_df[label_col].as_matrix()
y = y.flatten()
X = std_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

_logger.info("Finished loading samples into numpy arrays, X_train shape is {}, y_train shape is {}".format(X_train.shape, y_train.shape))


def log_loss(targets, predictions):
    return -(targets.T.dot(np.log(predictions)) + (1 - targets).T.dot(np.log(1 - predictions)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# W = np.array(np.random.randn(X_train.shape[1]))
# b = np.random.random()
# _logger.info("Initialized weights with shape {} and bias {}".format(W.shape, b))
#
# loss_df = pd.DataFrame(columns=['log_loss', 'w', 'b', 'gradient_w', 'gradient_b'])
# learning_rate = 0.0001
# predictions = sigmoid(X_std_scaled.dot(W) + b)
# for epoch in range(10000):
#     _logger.info("{}th epoch started...".format(epoch))
#     gradient_w = X_std_scaled.T.dot(predictions - y_train)
#     gradient_b = np.sum(predictions - y_train)
#     _logger.info("Computed gradient of W:{}, gradient of bias:{}".format(gradient_w, gradient_b))
#     W = W - learning_rate * gradient_w
#     b = b - learning_rate * gradient_b
#     predictions = sigmoid(X_std_scaled.dot(W) + b)
#     loss_df.loc[epoch, 'log_loss'] = log_loss(y_train, predictions)
#     loss_df.loc[epoch, 'w'] = W
#     loss_df.loc[epoch, 'b'] = b
#     loss_df.loc[epoch, 'gradient_w'] = gradient_w
#     loss_df.loc[epoch, 'gradient_b'] = gradient_b

model = NumpyBinaryClassifier()
model.fit(X_train, y_train)
print("Training set score:{}".format(model.score(model.predictions, model.targets)))
model.draw_log_loss()

test_predictions = model.predict(X_test)
print("Testing set score:{}".format(model.score(test_predictions, y_test)))

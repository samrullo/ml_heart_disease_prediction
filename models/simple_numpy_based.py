import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig()
logging.getLogger(__file__)

class NumpyBinaryClassifier:
    def __init__(self, learning_rate=0.0001, numb_of_epochs=10000):
        self.learning_rate = learning_rate
        self.numb_of_epochs = numb_of_epochs
        self.loss_df = pd.DataFrame(columns=['log_loss', 'w', 'b', 'gradient_w', 'gradient_b'])
        return

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def log_loss(self, predictions, targets):
        return -(targets.dot(np.log(predictions)) + (1 - targets).dot(np.log(1 - predictions)))

    def predict(self, samples):
        return self.sigmoid(samples.dot(self.W) + self.b)

    def gradient_of_w(self, samples, predictions, targets):
        return samples.T.dot(predictions - targets)

    def gradient_of_b(self, predictions, targets):
        return np.sum(predictions - targets)

    def fit(self, samples: np.ndarray, targets: np.ndarray):
        logging.info("Will start doing gradient descent {} epochs based on {} samples".format(self.numb_of_epochs, len(samples)))
        self.samples = samples
        self.targets = targets
        self.sample_size = len(self.samples)
        self.numb_of_features = self.samples.shape[1]
        self.W = np.random.randn(self.numb_of_features)
        self.b = np.random.random()
        self.predictions = self.predict(self.samples)
        for epoch in range(self.numb_of_epochs):
            logging.info("{}th epoch started...".format(epoch))
            gradient_w = self.gradient_of_w(self.samples, self.predictions, self.targets)
            gradient_b = self.gradient_of_b(self.predictions, self.targets)
            logging.info("Computed gradient of W:{}, gradient of bias:{}".format(gradient_w, gradient_b))
            self.W = self.W - self.learning_rate * gradient_w
            self.b = self.b - self.learning_rate * gradient_b
            self.predictions = self.predict(self.samples)
            self.loss_df.loc[epoch, 'log_loss'] = self.log_loss(self.predictions, self.targets)
            self.loss_df.loc[epoch, 'w'] = self.W
            self.loss_df.loc[epoch, 'b'] = self.b
            self.loss_df.loc[epoch, 'gradient_w'] = gradient_w
            self.loss_df.loc[epoch, 'gradient_b'] = gradient_b
        return

    def score(self, predictions, targets):
        return np.mean(np.round(predictions) == targets)

    def draw_log_loss(self):
        plt.xlabel("Epochs")
        plt.ylabel("Log loss")
        plt.title("How Log loss changes through gradient descent")
        self.loss_df.log_loss.plot()
        plt.show()

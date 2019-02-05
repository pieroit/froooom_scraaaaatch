from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from core import  PieroitMLAlgo

class PieroitLinearRegression(PieroitMLAlgo):

    def __init__(self, learning_rate=0.0001, num_epochs=1000):

        self.learning_rate = learning_rate
        self.num_epochs    = num_epochs
        self.scaler        = StandardScaler()

    def fit(self, X, y):

        # normalize data
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        # prepare params
        numWeights   = len(X[0])
        self.weights = [0] * (numWeights + 1) # one is the bias

        # iterate data many times
        for epoch in range(self.num_epochs):

            mse = 0.0

            # get prediction and measure error
            for m in range(len(X)):
                x          = X[m]
                prediction = self.predictOne(x)
                error      = y[m] - prediction

                # adjust weights
                for i, x_i in enumerate(x):
                    self.weights[i] += (self.learning_rate * error * x_i)
                self.weights[-1] += self.learning_rate * error  # adjust bias

                mse += error * error

            print 'Epoch', epoch, 'error', mse/len(X)


    def predictOne(self, x):

        prediction = 0.0
        for i, xi in enumerate(x):
            prediction += x[i] * self.weights[i]
        prediction += self.weights[-1]  # bias term

        return prediction


if __name__ == '__main__':

    boston_dataset = load_boston()

    features = boston_dataset['data']
    targets  = boston_dataset['target']

    lr = PieroitLinearRegression(learning_rate=0.0001, num_epochs=5000)
    lr.fit(features, targets)
    predictions = lr.predict( features )

    sklr = LinearRegression()
    sklr.fit(features, targets)
    skpredictions = sklr.predict( features )



    print '***Error Scikit***', mean_squared_error(skpredictions, targets)
    print '***Error Piero***', mean_squared_error(predictions, targets)

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from core import  PieroitMLAlgo

np.set_printoptions(suppress=True)

class PieroitLogisticRegression(PieroitMLAlgo):

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

                #print 'error is', error

                # adjust weights
                for i, x_i in enumerate(x):
                    #print '\tweight', i, 'correction',   self.learning_rate * error * x_i
                    self.weights[i] += (self.learning_rate * error * x_i)
                self.weights[-1] += self.learning_rate * error  # adjust bias

                mse += error * error

            print('Epoch', epoch, 'error', mse/len(X))


    def predictOne(self, x):

        prediction = 0.0
        for i, xi in enumerate(x):
            prediction += x[i] * self.weights[i]
        prediction += self.weights[-1]  # bias term

        prediction = 1/( 1 + ( np.e**(-prediction)) )

        return prediction


if __name__ == '__main__':

    boston_dataset = load_breast_cancer()

    features = boston_dataset['data']
    targets  = boston_dataset['target']

    lr = PieroitLogisticRegression(learning_rate=0.0001, num_epochs=1000)
    lr.fit(features, targets)
    predictions = lr.predict( features )
    # turn probabilities into classifications
    for i, p in enumerate(predictions):
        if p > 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0

    sklr = LogisticRegression()
    sklr.fit(features, targets)
    skpredictions = sklr.predict( features )

    print('***Error Scikit***', classification_report(skpredictions, targets))
    print('***Error Piero***', classification_report(predictions, targets))

    print(sklr.coef_, sklr.intercept_)
    print(lr.weights)

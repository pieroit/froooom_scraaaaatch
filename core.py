class PieroitMLAlgo():

    def predict(self, X):

        X = self.scaler.transform(X)

        y = []
        for x in X:
            y.append( self.predictOne(x) )
        return y
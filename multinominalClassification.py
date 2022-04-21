import pandas as pd

from libraries import *

class MultinominalLogisticRegression:
    def __init__(self, X, y):
        self.X = X.values
        self.targets = y.unique()
        self.y = list(y)
        self.LCRFClass = LRCF(self.X, self.y)

    def prepareTargetY(self, target, y):
        newY = [1 if target == i else 0 for i in y]
        return newY

    def train(self, batchSize = 100, epochs = 1000, learningRate = 0.01):
        WBLDict = dict()

        for i in self.targets:
            targetY = self.prepareTargetY(i, self.y)
            WBLDict[i] = LRCF(self.X, targetY).train()

        return WBLDict

    def predictZ(self, X, weight, bias):
        X = self.LCRFClass.normalize(X)
        preds = self.LCRFClass.sigmoid(np.dot(X, weight) + bias)
        return preds

    def predict(self, X, WBLDict):
        predictions = []
        for i in range(X.iloc[:,0].__len__()):
            predValues = [self.predictZ(X.iloc[i, :], WBLDict[j][0], WBLDict[j][1]) for j in self.targets]
            predictions.append(self.targets[predValues.index(max(predValues))])
        return predictions


def main():
    data = pd.read_csv("data.csv", index_col=0)
    X = data.iloc[:, 0:data.iloc[0, :].__len__() - 1]
    y = data.loc[:, "tag"]

    model = MultinominalLogisticRegression(X,y)
    #WBLDict = model.train()
    #pd.DataFrame(WBLDict).to_csv("WBLDict.csv")
    WBLDict = pd.read_csv("WBLDict.csv", index_col=0)
    print(WBLDict.values)
    #print(model.predict(X, WBLDict))

if __name__ == "__main__":
    main()
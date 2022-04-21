import numpy as np
import pandas as pd

from libraries import *

class LRCF:
    def __init__(self, X = pd.DataFrame, y = pd.Series):
        self.X = pd.DataFrame(X)
        self.X = self.X.values
        self.y = list(y)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.normalizedX = self.normalize(self.X)

    def costFunction(self, yHat):
        loss = -np.mean(self.y * (np.log(yHat)) - (1 - self.y) * np.log(1 - yHat))
        return loss

    def train(self, batchSize = 100, epochs = 1000, learningRate = 0.01):
        sampleLength, featureLength = self.X.shape
        weight = np.zeros((featureLength,1))
        bias = 0

        self.y = self.y.reshape(sampleLength, 1)
        X = self.normalizedX

        losses = []

        for epoch in range(epochs):
            for i in range((sampleLength - 1) // (batchSize + 1)):
                startI = i * batchSize
                endI = startI + batchSize

                xBatch = self.X[startI:endI]
                yBatch = self.y[startI:endI]

                yHat = self.sigmoid(np.dot(xBatch, weight) + bias)

                dWeight, dBias = self.gradients(xBatch, yBatch, yHat)

                weight -= learningRate * dWeight
                bias -= learningRate * dBias
            loss = self.costFunction(self.sigmoid(np.dot(X, weight) + bias))
            losses.append(loss)

        return weight, bias, losses

    def predict(self, X, weight, bias):
        X = self.normalize(X)
        preds = self.sigmoid(np.dot(X, weight) + bias)
        predictionClass = [1 if i > 0.5 else 0 for i in preds]
        return np.array(predictionClass)

    def gradients(self, X, y, yHat):
        sampleLength = X.shape[0]
        dWeight = (1/sampleLength) * np.dot(X.T, (yHat - y))
        dBias = (1/sampleLength) * np.sum((yHat - y))
        return dWeight, dBias

    def accuracy(self, y, predictions):
        accuracy = np.sum(y == predictions) / len(y)
        return accuracy

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    def normalize(self, X):
        for i in range(X.shape[0]):
            X = (X - X.mean(axis = 0)) / X.std(axis = 0)

        return X
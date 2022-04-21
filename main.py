from libraries import *

data = pd.read_csv("dataBinary.csv", index_col=0)
X = data.iloc[:, 0:data.iloc[0, :].__len__() - 1]
y = data.loc[:,"tag"]


model = LRCF(X, y)
w, b, l = model.train()
predictions = model.predict(model.X, w, b)
print(model.accuracy(y, predictions))

#print(LRCF().train(X, y, batchSize=100, epochs=1000, learningRate=0.01, norm = True))


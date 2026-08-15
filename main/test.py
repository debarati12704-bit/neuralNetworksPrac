import numpy as np
#testing class Perceptron on AND Gate
from perceptron import Perceptron
#AND Gate
X= np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y= np.array([0,0,0,1])

#test
model=Perceptron()
model.fit(X,y)
predictions = model.predict(X)

print("Predictions:", predictions)

print("Weights:", model.weights)
print("Bias:", model.bias)

#XOR Gate
y_xor = np.array([0, 1, 1, 0])

model = Perceptron(learning_rate=0.1, epochs=100)
model.fit(X, y_xor)

print("Predict XOR:",model.predict(X))
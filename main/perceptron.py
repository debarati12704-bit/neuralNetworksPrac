import numpy as np

class Perceptron:
    def __init__(self,learning_rate=0.1,epochs=10):
        self.learning_rate=learning_rate
        self.epochs=epochs
    def predict(self,X):
        Z=np.dot(X,self.weights)+self.bias
        return np.where(Z>=0,1,0)
    def fit(self,X,y):
        self.weights=np.zeros(X.shape[1])
        self.bias=0

        for _ in range(self.epochs):
            for xi,target in zip(X,y):
                predict=self.predict(xi)
                update=self.learning_rate*(target-predict)
                self.weights+=update*xi
                self.bias+=update



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

print("Predictions AND:", predictions)

print("Weights:", model.weights)
print("Bias:", model.bias)

#XOR Gate
y_xor = np.array([0, 1, 1, 0])

model = Perceptron(learning_rate=0.1, epochs=100)
model.fit(X, y_xor)

print("Predict XOR:",model.predict(X))
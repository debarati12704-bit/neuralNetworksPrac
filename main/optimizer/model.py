import numpy as np

class LinearRegression:
    def __init__(self):
        self.w=0.0
        self.b=0.0
    def forward(self,X):
        return self.w*X+self.b
    def loss(self,y_pred,y):
        return np.mean((y_pred-y)**2)
    def gradient(self,X,y_pred,y):
        n=len(X)
        dw=(2/n)*np.sum(X*(y_pred-y))
        db=(2/n)*np.sum(y_pred-y)
        return dw,db

import numpy as np

from model import LinearRegression
from optimizergd import GradientDescent


X=np.array([1,2,3,4])
y=np.array([3,5,7,9])

model=LinearRegression()
optimizer=GradientDescent(learning_rate=0.01)

for epoch in range(1000):
    #forward
    y_pred=model.forward(X)
    #loss
    loss=model.loss(y_pred,y)
    #backpropagation
    dw,db=model.gradient(X,y_pred,y)
    #update
    optimizer.step(model,db,dw)
    
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"Loss={loss:.4f}, "
            f"w={model.w:.4f}, "
            f"b={model.b:.4f}"
        )
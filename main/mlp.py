import numpy as np
from sklearn.neural_network import MLPClassifier

model=MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="relu",
    max_iter=3550,
    random_state=23
)
#test XOR
X= np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
#XOR Gate
y_xor = np.array([0, 1, 1, 0])
model.fit(X,y_xor)
prediction=model.predict(X)
print(
    "\n predicted:", prediction,
    "\n Actual:", y_xor
)
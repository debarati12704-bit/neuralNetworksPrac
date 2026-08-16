import numpy as np
#NN with:
# 2 inputs
# 2 hidden layers with layer1 with 2 neurons & layer2 with 1 neuron 
# 1 output
#activation func(a): ReLu(max(0,z))
#for regression problem

# 2 inputs
X=np.array([
    [2],
    [3]
])


#layer 1 with 2 neurons
W1=np.array([
    [0.5,0.2],
    [0.4,0.3]
])
B1=np.array([[0.1],[0.2]])

Z1=W1@X+B1
A1=np.maximum(0,Z1)

print("\n output of first layer =", A1)

#layer2 with 1 neuron
W2=np.array([[0.6,0.5]])
B2=np.array([[0.1]])
Z2=W2@A1+B2

print("\n output of second layer:",Z2)

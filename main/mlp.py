from sklearn.neural_network import MLPClassifier
def mlpclassifier():
    model=MLPClassifier(
        hidden_layer_sizes=(4,),
        activation="relu",
        max_iter=4000
    )
    return model

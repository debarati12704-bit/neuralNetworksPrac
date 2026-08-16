class GradientDescent:
    def __init__(self,learning_rate=0.01):
        self.learning_rate=learning_rate
    def step(self,model,db,dw):
        model.w-=dw*self.learning_rate
        model.b-=db*self.learning_rate
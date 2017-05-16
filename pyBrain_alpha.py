#Welcome to pyBrain Alpha 1.0 developed my Matthew Ognibene 14 May 2017
#My third NN project, now with multivariable calc and linear algebra!
import numpy as np

class Network():
    def __init__(self,numOfInputs,numOfOutputs,hiddenSize):
        #Hyperparameters
        #NOTE: In this version of pyBrain, there is only ONE hidden layer
        self.numOfInputs=numOfInputs
        self.numOfOutputs = numOfOutputs
        self.hiddenSize = hiddenSize



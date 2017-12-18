#Welcome to pyBrain Alpha 1.0 developed my Matthew Ognibene 14 May 2017
#My third NN project, now with multivariable calc and linear algebra!
#This project is simply to give you the basic foundations of NN
#It has creating neural netwokrs and forward propogation alrealdy built in, but back propogation, gradient descent, etc
#activation functions (except sigmoid) and training methods must be built in.
#this project is by no means meant to stand alone. Its purpose is to be built upon to try different training algorithms
#without having to rebuild all the basics each time.
import numpy as np

#Number Number [List-of Number] -> Network
#numOfInputs: number of input nodes
#numOfOutputs: number of output nodes
#hiddenSkeleton: list of numbers, a skeleton of the hidden layers
#ex [1 2 3 1] is a hidden layer where the first layer has one node, the next two, the next 3, and the next 1
class Network():
    def __init__(self,numOfInputs,numOfOutputs,hiddenSkeleton):
        #Hyperparameters
        self.numOfInputs=numOfInputs
        self.numOfOutputs = numOfOutputs
        self.hiddenSkeleton = hiddenSkeleton
        numOfHiddenLayers = len(hiddenSkeleton)
        size_list = [numOfInputs]
        for i in hiddenSkeleton:
            size_list.append(i)
        size_list.append(numOfOutputs)
        self.size_list = size_list
        weights = []
        for n in range(numOfHiddenLayers+1): #n = layer num
            w = np.random.randn(size_list[n],size_list[n+1])
            for r in range(len(w)):
                for c in range(len(w[0])):
                    w[r][c]= abs(w[r][c])
            weights.append(w)
        self.weights = weights

    def forward(self,x):
        # where x = l sub 0 (Refer to notes)
        l=x
        for w in self.weights:
            z = np.dot(l, w)
            l = self.sigmoid(z)
        return l

    def sigmoid(self,l):
        return 1 / (1 + np.exp(-l))




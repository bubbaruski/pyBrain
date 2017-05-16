#Welcome to pyBrain Alpha 1.0 developed my Matthew Ognibene 14 May 2017
#My third NN project, now with multivariable calc and linear algebra!
import numpy as np
import math
import random


class Network():
    def __init__(self,numOfInputs,numOfOutputs,hiddenSize):
        #Hyperparameters
        #NOTE: In this version of pyBrain, there is only ONE hidden layer
        self.numOfInputs=numOfInputs
        self.numOfOutputs = numOfOutputs
        self.hiddenSize = hiddenSize
        size_list = [numOfInputs, hiddenSize, numOfOutputs]
        self.size_list = size_list
        weights = []
        for n in range(2): #n = layer num TODO must change if more hidden layers are added

            w= np.random.randn(size_list[n],size_list[n+1])
            for r in range(len(w)):
                for c in range(len(w[0])):
                    w[r][c]= abs(w[r][c])
            weights.append(w)
        self.weights = weights

    def forward(self,x):
        # where x = l sub 0 (Refer to notes)
        #TODO for loop this bish

        l1 = np.dot(x, self.weights[0])
        l1 = self.sigmoid(l1)
        print (l1)
        l2 = np.dot(l1, self.weights[1])
        l2 = self.sigmoid(l2)
        print(l2)
        return l2

    def sigmoid(self,l):

        return 1 / (1 + np.exp(-l))



"""
LogisticRegression.py

CS440/640: PA1
Team Members: Khai Phan, Michael Deng, Nick Mauro

Assignment Part: "Logistic Regression"
"""

import numpy as np 
import matplotlib.pyplot as plt 

class LogisticRegression:
    """
    This class implements a Multinomial Logistic Regression Classifier, using h(z) = exp(z), which
    outputs K numbers, where K is the number of classes.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        
        self.theta = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, output_dim))
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self, X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        returns:
            cost: average cost per data sample
        """
        
        # Need to compute softmax scores for X
        X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
        y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        # Only the output of the correct class label contributes to the cost
        runningCost = 0
        
        for i in range(len(X)):
            yHot = np.zeros(len(self.bias))
            yHot[y[i]] = 1
            runningCost -= np.sum(yHot * np.log(softmax_scores[i]))
               
        avgCostPerSample = runningCost/len(X)   
         
        return avgCostPerSample

    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        """
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        """
        Learns model parameters to fit the data.
        """  
         
        X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
        y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')
        
        cost = self.compute_cost(X, y)
        
        while True:   
            # Need to compute softmax scores for X
            z = np.dot(X,self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            # Argmax returns the index of the highest -> prediction
            predictions = np.argmax(softmax_scores, axis=1)
            
            # SetUp OneHotY
            yHot = np.zeros(len(self.bias))
            yHot[y[i]] = 1
            np.dot(X,
        
        return 0

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def plot_decision_boundary(model, X, y):
    """
    Function to print the decision boundary given by model.
    
    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels
    """
    
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()

################################################################################    
#
#X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
#y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')
#why = np.random.randn(2, 3) / np.sqrt(2)
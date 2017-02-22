"""
LogisticRegression.py

CS440/640: Lab-Week5

Lab goal: 1) Implement logistic regression classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 

class LogisticRegression:
    """
    This class implements a Logistic Regression Classifier.
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
    
    def compute_cost(self,X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        returns:
            cost: average cost per data sample
        """
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        # Only the output of the correct class label contributes to the cost
        runningCost = 0
        
        for i in range(len(X)):
            yHot = np.zeros(self.bias.size)
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
        cost = self.compute_cost(X, y)
        
        for i in range(1000):   
            # Need to compute softmax scores for X (foreward propagation)
            z = np.dot(X,self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            # SetUp OneHotY
            yHot = np.zeros(X.shape)
            for j in range(len(X)):
                yHot[j,y[j]] = 1
            
            # backward propagation
            gradient_weights = np.dot(np.transpose(X), softmax_scores - yHot)
            gradient_biases = np.dot(np.ones(len(X)),softmax_scores - yHot)
            
            # Update model parameters
            self.theta -= gradient_weights * 0.05 # learning rate as 5%
            self.bias -= gradient_biases * 0.05
            
            # Argmax returns the index of the highest -> prediction
            # predictions = np.argmax(softmax_scores, axis=1)
            
            

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
    model.fit(X,y)
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()


################################################################################    

#1. Load data
X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')

#2. plot data
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.bwr)
plt.show()

#3. Initialize Logistic Regression object
input_dim = len(X[0,])
output_dim = 2
model = LogisticRegression(input_dim, output_dim)
plot_decision_boundary(model, X, y)

            
    
"""
LogisticRegression.py

CS440/640: PA1
Team Members: Khai Phan, Michael Deng, Nick Mauro, Stephanie Hsieh

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
        
        # Compute Dot Product of X and Theta, Summed With Biases:
        # i.e. z1 = w1x1 + b1
        # Thus, z is now (# samples by # classes array/matrix )
        z = np.dot(X,self.theta) + self.bias
        
        # Apply H(z) to each element of this matrix
        exp_z = np.exp(z)
        
        # Compute Softmax Scores are computed by scaling each element with the sum
        # of each row
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        # Only the Output of the CORRECT Class Label Contributes to the Cost
        runningCost = 0
        
        # Let's make a one_hot_y array...
        # The array should be sample by # classes 
        one_hot_y = np.zeros((len(y), len(self.bias.T)))
        
        runningCost = 0
        
        # Initialize this array with the proper hot ones
        # while initializing, perform cost computation
        length_y = len(y)
        for i in range(length_y):
            # This works because, the index of the hot one is whatever number
            # of the correct classification
            one_hot_y[i,int(y[i])] = 1
            # add the one cost value that had contribution to runningCost
            runningCost += -1 * np.sum(one_hot_y[i] * np.log(softmax_scores[i]))
                      
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

        # Current Cost & Learning Rate
        current_cost = self.compute_cost(X, y)
        learning_rate = 0.01
        
        # Initialize Ground Truth
        one_hot_y = np.zeros((len(y), len(self.bias.T)))
        length_y = len(y)
        for i in range(length_y):
            # This works because, the index of the hot one is whatever number
            # of the correct classification
            one_hot_y[i,int(y[i])] = 1
        
        # safety for while loop
        iterations = 0
        while True:   
            # Need to compute softmax scores for X
            z = np.dot(X,self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
            # Compute Gradients & update weights/biases
            gradient_weights = np.dot(X.T, softmax_scores - one_hot_y)
            gradient_biases = np.dot(np.ones(len(X)), softmax_scores - one_hot_y)
            
            self.theta -= gradient_weights * learning_rate
            self.bias -= gradient_biases * learning_rate
            
            # compute cost change
            old_cost = current_cost
            current_cost = self.compute_cost(X, y)
            diff_cost = current_cost - old_cost
            iterations += 1
            
            # break if converge value of 0.0001 OR iterations go too high
            if abs(diff_cost) < 0.0001 or iterations > 5000:
                break

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

# Here below is for Linear Data

##1. Load Data
#X = np.genfromtxt('DATA/NonLinear/X.csv', delimiter=',')
#y = np.genfromtxt('DATA/NonLinear/y.csv', delimiter=',')
#
##2. Initialize Logistic Regression Object
#input_dim = len(X[0,])
#output_dim = 2
#model = LogisticRegression(input_dim, output_dim)
#plot_decision_boundary(model, X, y)

# Start here for Digits Data

##1. Load Data
#X_train = np.genfromtxt('DATA/Digits/X_train.csv', delimiter=',')
#y_train = np.genfromtxt('DATA/Digits/y_train.csv', delimiter=',')
#X_test = np.genfromtxt('DATA/Digits/X_test.csv', delimiter=',')
#y_test = np.genfromtxt('DATA/Digits/y_test.csv', delimiter=',')
#
##2. Initialize Logistic Regression Object
#input_dim = len(X_train[0,])
#output_dim = 10
#model = LogisticRegression(input_dim, output_dim)
#
##3 Fit and predict
#model.fit(X_train,y_train)
#Z = model.predict(X_test)
#correct = 0
#for i in range(len(Z)):
#    correct += (Z[i] == y_test[i])
#print "{:.1%}".format(float(correct)/float(len(Z)))
#
## For Confusion Matrix
#model.fit(X_train, y_train)
#Z = model.predict(X_test)
#confusionMatrix = np.zeros((10,10))
#for i in range(len(Z)):
#    confusionMatrix[Z[i],int(y_test[i])] += 1
#print(confusionMatrix)

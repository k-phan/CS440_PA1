"""
NeuralNet.py

CS440/640: PA1
Khai Phan, Michael Deng, Nick Mauro, Stephanie Hsieh

Assignment Part: "Neural Networks"
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Although this is pretty much a given, it may be helpful to
define the following in terms of readability later.
NOTE: Could not decide between using sigmoid function vs. tanh (?)
"""

# h(z) and its derivative
def h(z):
	return np.exp(z)

def h_prime(z):
	return np.exp(z)
	
# sigma test function
def sigma(z):
        return 1 / (1 + np.exp(-z))
	
	

class NeuralNetworks:
	"""
	Implement a Neural Network w/ One Hidden Layer of varying
	Number of nodes (in all three layers)
	"""

	"""
	input/output dim = num. of inputs/outputs
	learn_rate = learning rate for the gradient updates
	hidden_dim = num. of nodes in hidden layer
	"""

	def __init__(self, input_dim, output_dim, learn_rate, hidden_dim):
		# Let's Keep a Record of These, Shall We?
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.learn_rate = learn_rate
		self.hidden_dim = hidden_dim

		# Now, let's randomly initialize some weights & biases ...
		# Note that this is hard-coded for a NN with 1 hidden layer
		self.w1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
		self.b1 = np.zeros((1, hidden_dim))
		self.w2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
		self.b2 = np.zeros((1, output_dim))

	# --------------------------------------------------------------------------

	"""
	Taking predict LogisticRegression.py:
	Purpose is to predict an output based on current weights/biases.
	"""

	def predict(self, X):
		# Hard-code Forward Propagation
		z1 = np.dot(X, self.w1) + self.b1
		a1 = np.tanh(z1)
		z2 = np.dot(a1, self.w2) + self.b2

		# Output Layer
		scores = h(z2)
		softmax_scores = scores / np.sum(scores, axis=1, keepdims=True)

		# Returns Index of Largest Value!
		predictions = np.argmax(softmax_scores, axis=1)
		return predictions

	# -------------------------------------------------------------------------

	"""
	Compute Cost Here
	Modifying from LogisticRegression.py as well.
	"""

	def compute_cost(self, X, y):
		# Forward Prop as in Predict
		z1 = np.dot(X, self.w1) + self.b1
		a1 = np.tanh(z1)
		z2 = np.dot(a1, self.w2) + self.b2
		
		# Output Layer
		scores = h(z2)
		softmax_scores = scores / np.sum(scores, axis=1, keepdims=True)

		# Calculate Cost
		# Only calculate the correct one!
		totalCost = 0
		lenY = len(y)
		for i in range(lenY):
			# Wanted value happens to be the column index of sm_scores!
			totalCost += -1 * np.log(softmax_scores[i, int(y[i])])

		avgCostPerSample = totalCost / len(X)
		return avgCostPerSample

	# ------------------------------------------------------------------------

	def fit(self, X, y):
		"""
		Learns model parameters to fit the data.
		costPlot variable and corresponding lines are used to plot answer to #4
		They are commented out when unnecessary.
		"""

		# Initialize Ground Truth
		lenY = len(y)
		one_hot_y = np.zeros((lenY, self.output_dim))
		for i in range(lenY):
			one_hot_y[i, int(y[i])] = 1

		# Current Cost
		current_cost = self.compute_cost(X, y)
                
                #costPlot = np.array([0, current_cost])
		
		# Safety For While Loop
		iterations = 0
		while True:
			# Need to Compute Softmax Scores
			z1 = np.dot(X, self.w1) + self.b1
			a1 = np.tanh(z1)
			z2 = np.dot(a1, self.w2) + self.b2
			scores = h(z2)
			softmax_scores = scores / np.sum(scores, axis=1, keepdims=True)

			"""
			Compute Gradients -- Had Some Help w/ Logic Here
			Diff Between SMScores & Ground Truth
			http://neuralnetworksanddeeplearning.com/chap2.html
			"""
			delt_3 = softmax_scores
			delt_3[range(lenY), y.astype(int)] -= 1
			grad_w2 = np.dot(a1.T, delt_3)

			# Dot Product with Column of 1s = SUM!
			grad_b2 = np.sum(delt_3, axis=0, keepdims=True)
			delt_2 = np.dot(delt_3, self.w2.T) * (1 - np.power(a1,2.0))
			grad_w1 = np.dot(X.T, delt_2)
			grad_b1 = np.sum(delt_2, axis=0, keepdims=True)

			# Apply Gradient Descent Update
			self.w2 += -1 * self.learn_rate * grad_w2
			self.b2 += -1 * self.learn_rate * grad_b2
			self.w1 += -1 * self.learn_rate * grad_w1
			self.b1 += -1 * self.learn_rate * grad_b1

			# Compute Cost Change
			old_cost = current_cost
			current_cost = self.compute_cost(X, y)
			diff_cost = current_cost - old_cost
			iterations += 1
			
			#costPlot = np.vstack((costPlot, np.array([iterations,current_cost])))	
			
			# Break if Converge Value of 0.0001 or Iterations Too high
			if abs(diff_cost) < 0.0001 or iterations > 5000:
			    break
			 
		#plt.scatter(costPlot[:,0],costPlot[:,1], s=10, cmap=plt.cm.Spectral)
		#plt.show()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

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

# Here below for Linear Data

# 1. Load Data Here To Begin:
X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')

# 2. Initialize Neural Network & Plot
input_dim = len(X[0,])
output_dim = 2
model = NeuralNetworks(input_dim, output_dim, 0.001, 8)
plot_decision_boundary(model, X, y)


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
#model = NeuralNetworks(input_dim, output_dim, 0.001, 8)
#
##3 Fit and predict
#model.fit(X_train,y_train)
#Z = model.predict(X_test)
#correct = 0
#for i in range(len(Z)):
#    correct += (Z[i] == y_test[i])
#print "{:.1%}".format(float(correct)/float(len(Z)))

## For Confusion Matrix
#model.fit(X_train, y_train)
#Z = model.predict(X_test)
#confusionMatrix = np.zeros((10,10))
#for i in range(len(Z)):
#    confusionMatrix[Z[i],int(y_test[i])] += 1
#print(confusionMatrix)
#percentages = confusionMatrix / np.sum(confusionMatrix, axis=0, keepdims=True)
#print(percentages)

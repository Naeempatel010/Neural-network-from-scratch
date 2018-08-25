import numpy as np 
from math import floor

def sigmoid(x):
		return 1.0/(1+np.exp(-x))

def sigmoid_derivatives(x):
		return x*(1.0-x)

class NeuralNetwork:
	
	def __init__(self,x,y):
		self.inputs=x
		self.weights1=np.random.rand(self.inputs.shape[1],4)
		self.weights2=np.random.rand(4,1)
		self.y=y
		self.output=np.zeros(y.shape)

	def feedforward(self):
		self.layer1=sigmoid(np.dot(self.inputs,self.weights1))
		self.output=sigmoid(np.dot(self.layer1,self.weights2))

	def backprop(self):
		d_weights2=np.dot(self.layer1.T,(2*(self.y-self.output)*sigmoid_derivatives(self.output)))
		d_weights1 = np.dot(self.inputs.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivatives(self.output), self.weights2.T) * sigmoid_derivatives(self.layer1)))
		self.weights1=self.weights1+d_weights1
		self.weights2=self.weights2+d_weights2


if __name__=='__main__':
	X=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
	y=np.array([[0],[1],[1],[0]])
	nn=NeuralNetwork(X,y)
	epoch=1000
	print(nn.weights1.shape,nn.weights2.shape)

	for i in range(epoch):
		#print("epoch:",i)
		#print(nn.weights1,nn.weights2)
		nn.feedforward()
		nn.backprop()

print(nn.output)
print([floor(x)for x in nn.output])	


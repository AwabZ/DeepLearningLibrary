import numpy as np
import random

class Network(object):

    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y,1) for y in layers[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(layers[:-1], layers[1:])]



    def feedforward(self, a):
        """The initial (a) is the input vector (x) and the final
        returned (a) is the output vector of the entire network"""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a) + b)
        return a


    def SGD(self, training_data, epochs, batch_size, eta, test_data=None):
         if test_data:
              num_tests = len(test_data)
         n = len(training_data)
         for epoch in range(epochs):
              random.shuffle(training_data)
              mini_batches = []
              for k in range(0, n, batch_size):
                   mini_batches.append(training_data[k:k+batch_size])
              for mini_batch in mini_batches:
                   self.update_mini_batch(mini_batch, eta)
              if test_data:
                   test_results = self.evaluate(test_data)
                   print(f"Epoch {epoch}: {test_results} / {num_tests}")
              else:
                   print(f"Epoch {epoch} complete")



def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

myLayers = (2, 3, 1)
myNetwork = Network(myLayers)
some_input = np.array([[0.5],[0.3]])
result = myNetwork.feedforward(some_input)
print(f"result: {result}")

# For Demonstration Purposes:
print(f"num_layers: {len(myLayers)}")


biases = [np.random.randn(y,1) for y in myLayers[1:]]
print(f"biases: {biases} \n \n")

print("out_zip")
out_zip = zip(myLayers[:-1], myLayers[1:])
for i in out_zip:
    print(i)


weights = [np.random.randn(y,x) for x,y in zip(myLayers[:-1], myLayers[1:])]
print(f"weights: {weights} \n \n")


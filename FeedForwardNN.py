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

    def update_mini_batch(self, mini_batch, eta):
         n = len(mini_batch)
         nabla_b = [np.zeros(b.shape) for b in self.biases]
         nabla_w = [np.zeros(w.shape) for w in self.weights]
         for x,y in mini_batch:
              single_nabla_b, single_nabla_w = self.backprop(x, y)
              nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, single_nabla_b)]
              nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, single_nabla_w)]
              
         self.weights = [w - (eta/len(n)) * nw for w,nw in zip(self.weights, nabla_w)]
         self.biases = [b - (eta/len(n)) * nb for b,nb in zip(self.biases, nabla_b)]






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



nabla_b = [np.zeros(b.shape) for b in myNetwork.biases]
nabla_w = [np.zeros(w.shape) for w in myNetwork.weights]
print(f"biases: {myNetwork.biases}")
print(f"nabla_b initial: {nabla_b}")
print(f"weights: {myNetwork.weights}")
print(f"nabla_w initial: {nabla_w}")
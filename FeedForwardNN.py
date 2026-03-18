import numpy as np

class Network(object):

    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y,1) for y in layers[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(layers[:-1], layers[1:])]



myLayers = (2, 3, 1)
myNetwork = Network(myLayers)

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


import random
import math

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(input_size)]
        self.bias = [random.uniform(-1, 1) for _ in range(output_size)]

    def forward(self, input):
        self.input = input
        output = [0] * len(self.bias)
        for i in range(len(self.bias)):
            for j in range(len(self.input)):
                output[i] += self.input[j] * self.weights[j][i]
            output[i] += self.bias[i]
        self.output = output
        return output



class Sigmoid(Layer):
    def forward(self, input):
        self.input = input
        self.output = [1 / (1 + math.exp(-x)) for x in input]
        return self.output


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
layers = [
    Dense(input_size=2, output_size=2),  
    Sigmoid(),                              
    Dense(input_size=2, output_size=1),  
    Sigmoid()                               
]

model = NeuralNetwork(layers)

# Define XOR dataset
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Training 
epochs = 10000
learning_rate = 0.1
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        # Forward pass
        output = model.forward(X[i])
        

print("\nTesting the trained model:")
for i in range(len(X)):
    output = model.forward(X[i])
    print(f"Input: {X[i]}, Output: {output[0]}")

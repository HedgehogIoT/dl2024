import random

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

    def backward(self, output_grad, learning_rate):
        input_grad = [0] * len(self.input)
        for i in range(len(self.input)):
            for j in range(len(output_grad)):
                input_grad[i] += output_grad[j] * self.weights[i][j]
            for j in range(len(output_grad)):
                self.weights[i][j] -= learning_rate * output_grad[j] * self.input[i]
        for i in range(len(output_grad)):
            self.bias[i] -= learning_rate * output_grad[i]
        return input_grad


class ReLU(Layer):
    def forward(self, input):
        self.input = input
        self.output = [max(0, x) for x in input]
        return self.output

    def backward(self, output_grad, learning_rate):
        return [output_grad[i] if self.input[i] > 0 else 0 for i in range(len(output_grad))]

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_grad, learning_rate):
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad, learning_rate)
        return output_grad
layers = [
    Dense(input_size=2, output_size=2),  
    ReLU(),                              
    Dense(input_size=2, output_size=1),  
    ReLU()                               
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
        
        loss = (output[0] - y[i]) ** 2
        total_loss += loss
        
        output_grad = [2 * (output[0] - y[i])] 
        model.backward(output_grad, learning_rate)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{epochs}, Average Loss: {total_loss / len(X)}")

print("\nTesting the trained model:")
for i in range(len(X)):
    output = model.forward(X[i])
    print(f"Input: {X[i]}, Output: {output[0]}")

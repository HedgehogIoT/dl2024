import math

class LogisticRegression:
    def __init__(self, learning_rate, no_ite):
        self.learning_rate = learning_rate
        self.no_ite = no_ite
        self.W1 = 0
        self.W2 = 0
        self.W0 = 0

    def fit(self, X1, X2, Y):
        self.X1 = X1
        self.X2 = X2
        self.Y = Y

    def sigmoid(self, x):
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            return math.exp(x) / (1 + math.exp(x))

    def update_weights(self):
        n = len(self.Y)
        sum_error_w1 = sum_error_w2 = sum_error_w0 = 0

        for i in range(n):
            # Calculate predicted value
            Y_hat = self.sigmoid(self.X1[i]*self.W1 + self.X2[i]*self.W2 + self.W0)

            # Calculate errors
            error = self.Y[i] - Y_hat

            # Update the sum of errors for each weight
            sum_error_w1 += error * self.X1[i]
            sum_error_w2 += error * self.X2[i]
            sum_error_w0 += error

        # Update weights
        self.W1 += (self.learning_rate * sum_error_w1)
        self.W2 += (self.learning_rate * sum_error_w2)
        self.W0 += (self.learning_rate * sum_error_w0)

    def predict(self, X1_test, X2_test):
        Y_pred = []
        for i in range(len(X1_test)):
            Y_pred.append(self.sigmoid(self.W1 * X1_test[i] + self.W2 * X2_test[i] + self.W0))
        return Y_pred


model = LogisticRegression(learning_rate=0.01, no_ite=100)

# Example 
X1_train = [1000, 2000, 3000, 4000]
X2_train = [2, 4, 6, 8]
Y_train = [0, 0, 1, 1]

# Fit the model
model.fit(X1_train, X2_train, Y_train)

# Train the model
for _ in range(model.no_ite):
    model.update_weights()

# Test data
X1_test = [1500, 2500]
X2_test = [3, 5]

# Make predictions
predictions = model.predict(X1_test, X2_test)
print(predictions)
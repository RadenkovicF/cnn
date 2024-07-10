import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 1,1, 1]) # or

W = [0.1, 0.2]
b = 0.1

eta = 0.2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def perceptron(x):
    global W, b
    return sigmoid(np.dot(W, x) + b)

def learn(x, y): 
    global W, b
    W = W - eta * (perceptron(x) - y) * (1- perceptron(x)) * x
    b = b - eta * (perceptron(x) - y) * (1- perceptron(x))

    # alternative with same solution:
    # W = W - eta * (perceptron(x) - y) * perceptron(x) * (1- perceptron(x)) * x
    # b = b - eta * (perceptron(x) - y) * perceptron(x) * (1- perceptron(x))


error = 1
for _ in range(1000):
    error = 0
    for x, y in zip(X, Y):
        learn(x, y)
        error += abs(perceptron(x) - y)
        print("x: ", x, "y: ", y, "prediction: ", perceptron(x))
    print(error)
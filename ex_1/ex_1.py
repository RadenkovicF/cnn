import numpy as np

# Sigmoid-Aktivierungsfunktion (siehe Formel 3 im PDF)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Ableitung der Sigmoid-Funktion
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Perzeptron-Klasse
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.random.randn(input_size)  # Gewichte für Eingaben initialisieren
        self.bias = np.random.randn()  # Bias initialisieren
        self.lr = lr

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias  # Bias hinzufügen
        return sigmoid(z)

    def fit(self, X, y, epochs=250):
        for epoch in range(epochs):
            for i in range(y.shape[0]):
                xi = X[i]
                yi_hat = self.predict(xi)
                error = y[i] - yi_hat

                # Berechnung der Aktivierung vor der Sigmoid-Funktion für die Ableitung
                z = np.dot(self.weights, xi) + self.bias

                # Gewichte und Bias anpassen gemäß Gradientenabstieg (siehe Formel 13 im PDF)
                self.weights += self.lr * error * sigmoid_derivative(z) * xi
                self.bias += self.lr * error * sigmoid_derivative(z)

# Trainingsdaten für OR, AND und XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_or = np.array([0, 1, 1, 1])
y_and = np.array([0, 0, 0, 1])
y_xor = np.array([0, 1, 1, 0])

# Perzeptron für OR trainieren
p_or = Perceptron(input_size=2)
p_or.fit(X, y_or)
print("OR Gate Predictions:")
print([np.round(p_or.predict(x)) for x in X])

# Perzeptron für AND trainieren
p_and = Perceptron(input_size=2)
p_and.fit(X, y_and)
print("AND Gate Predictions:")
print([np.round(p_and.predict(x)) for x in X])

# Perzeptron für XOR trainieren (wird nicht korrekt funktionieren, da XOR nicht linear separabel ist)
p_xor = Perceptron(input_size=2)
p_xor.fit(X, y_xor)
print("XOR Gate Predictions:")
print([np.round(p_xor.predict(x)) for x in X])
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

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

def create_perceptron_model():
    model = Sequential()
    model.add(Dense(1, input_dim=2, activation='sigmoid'))  # Ein Perzeptron mit Sigmoid-Aktivierung
    model.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Perzeptron für OR trainieren
model_or = create_perceptron_model()
model_or.fit(X, y_or, epochs=1000, verbose=0)
print("OR Gate Predictions:")
print(model_or.predict(X).round())

# Perzeptron für AND trainieren
model_and = create_perceptron_model()
model_and.fit(X, y_and, epochs=1000, verbose=0)
print("AND Gate Predictions:")
print(model_and.predict(X).round())

# Perzeptron für XOR trainieren (wird nicht korrekt funktionieren, da XOR nicht linear separabel ist)
model_xor = create_perceptron_model()
model_xor.fit(X, y_xor, epochs=1000, verbose=0)
print("XOR Gate Predictions:")
print(model_xor.predict(X).round())
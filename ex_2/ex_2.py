import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score



def create_model(input_shape):
    model = Sequential(name="digits") # create model and name it "digits"
    model.add(Input(shape=input_shape)) # define input layer with input_shape
    model.add(Dense(100, activation='sigmoid')) # hidden layer 1 with 100 neurons
    model.add(Dense(100, activation='sigmoid')) # hidden layer 2 with 100 neurons
    model.add(Dense(10, activation='sigmoid'))  # Output layer with 10 classes for mnist datasets
    model.compile(optimizer=SGD(learning_rate=0.2), loss='binary_crossentropy', metrics=['accuracy'])
    return model

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Load and preprocess the data

train_X = train_X.reshape(-1, 28*28).astype('float32') / 255
test_X = test_X.reshape(-1, 28*28).astype('float32') / 255
train_y = to_categorical(train_y, 10)
test_y = to_categorical(test_y, 10)


model = create_model(input_shape=(28*28,))
history = model.fit(train_X, train_y, epochs=10, verbose=1, validation_data=(test_X, test_y))

print(f"Test accuracy: {model.evaluate(test_X, test_y, verbose=0)[1]*100:.2f}%")
print(f"History accuracy: {history.history['accuracy'][-1]}")   

print(f"Test loss: {model.evaluate(test_X, test_y, verbose=0)[0]}")
print(f"History loss: {history.history['loss'][-1]}")

# Predictions
y_pred = model.predict(test_X)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_y, axis=1)

# Calculate precision and recall
precision = precision_score(y_true, y_pred_classes, average='macro')
recall = recall_score(y_true, y_pred_classes, average='macro')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def create_model(input_shape):
    model = Sequential(name="digits")
    model.add(Input(shape=input_shape))
    model.add(Dense(100, activation='sigmoid')) 
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))  # Output layer with 10 classes for MNIST
    model.compile(optimizer=SGD(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess the data
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape(-1, 28*28).astype('float32') / 255
test_X = test_X.reshape(-1, 28*28).astype('float32') / 255
train_y = to_categorical(train_y, 10)
test_y = to_categorical(test_y, 10)

# Create the model
input_shape = (28*28,)
model = create_model(input_shape)

# Train the model
history = model.fit(train_X, train_y, epochs=10, verbose=1, validation_data=(test_X, test_y))


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_X, test_y, verbose=0)
print(f"Test accuracy: {test_accuracy*100:.2f}%")
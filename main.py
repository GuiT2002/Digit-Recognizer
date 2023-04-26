import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense
import numpy as np

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Create a sequential model
model = Sequential()

# Add Conv2D and MaxPool2D layers
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D((3, 3)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D((3, 3)))

# Add Flatten and Dense layers
model.add(Flatten())
model.add(Dense(128, activation='swish'))
model.add(Dense(128, activation='swish'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=50, epochs=50, validation_data=(x_test, y_test))

# Save the model
model.save('DR_v6')

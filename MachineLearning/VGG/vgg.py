from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

model = Sequential([
    # Layer 1
    Conv2D(
        input_shape=(224,224,3),
        filters=3,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=3,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    MaxPool2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same'),
    # Layer 2
    Conv2D(
        filters=6,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=6,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    MaxPool2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same'),
    # Layer 3
    Conv2D(
        filters=12,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=12,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=12,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    MaxPool2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same'),
    # Layer 4
    Conv2D(
        filters=24,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=24,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=24,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    MaxPool2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same'),
    # Layer 5
    Conv2D(
        filters=48,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=48,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=48,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    MaxPool2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same'),
    # Layer 6
    Conv2D(
        filters=96,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Flatten(),
    # Layer 7
    Dense(units=4096, activation='relu'),
    Dropout(rate=0.4),
    # Layer 8
    Dense(units=4096, activation='relu'),
    Dropout(rate=0.4),
    # Layer 9
    Dense(units=1000, activation='relu'),
    # Output layer
    Dense(units=1000, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.summary()
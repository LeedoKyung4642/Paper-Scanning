from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model = Sequential([
    # The first convolutional layer
    Conv2D(
        input_shape=(224,224,3),
        kernel_size=(11,11),
        filters=96,
        strides=(4,4),
        padding='same',
        activation='relu'),
    # The second convolutional layer
    MaxPooling2D(
        pool_size=(5,5),
        strides=(1,1),
        padding='same'),
    # The third convolutional layer
    MaxPooling2D(
        pool_size=(3,3),
        strides=(1,1),
        padding='same'),
    # The fourth convolutional layer
    Conv2D(
        kernel_size=(3,3),
        filters=384,
        padding='same',
        activation='relu'),
    # The fifth convolutional layer
    Conv2D(
        kernel_size=(3,3),
        filters=256,
        padding='same',
        activation='relu'),
    # Connect between Convolutional layers and Fully-Connected layers
    MaxPooling2D(
        pool_size=(2,2),
        strides=(1,1),
        padding='same'),
    Flatten(),
    # 6th Layer: Fully-Connected
    Dense(
        units=4096,
        activation='relu'),
    Dropout(
        0.4),
    # 7th Layer: Fully-Connected
    Dense(
        units=4096,
        activation='relu'),
    Dropout(
        0.4),
    # 8th Layer: Fully-Connected
    Dense(
        units=1000,
        activation='relu'),
    Dropout(
        0.4),
    # Output Layer
    Dense(
        units=1000,
        activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary()

# for layer in model.layers:
#     if layer.name.startswith('conv2d'):
#         print(f'{layer.input_shape} --> {layer.output_shape}', end='\n')
#     if layer.name.startswith('max_pooling2d'):
#         print(f'\n{layer.input_shape} --> {layer.output_shape} ==> ', end='')
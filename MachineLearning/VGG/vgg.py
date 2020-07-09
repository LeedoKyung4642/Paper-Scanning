from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar100

(x_train,y_train),(x_test,y_test)=cifar100.load_data()
x_train = x_train / 255.0 #0~255값을 0~1사이로 변경
x_test = x_test / 255.0
print(x_train[0].shape)
model = Sequential([
    # Layer 1
    Conv2D(
        input_shape=(32,32,3),
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=32,
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
        filters=64,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=64,
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
        filters=128,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=128,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=128,
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
        filters=256,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=256,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=256,
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
        filters=512,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=512,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'),
    Conv2D(
        filters=512,
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
        filters=1024,
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
    Dense(units=100, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(x_train,y_train,epochs=250)
model.evaluate(x_test,y_test,verbose=2)
model.summary()
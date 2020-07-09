from tensorflow.keras.layers import Layer
from tensorflow.keras import backend

class LocalResponseNormalization(Layer):

    def __init__(self, n=5, alpha=1e-4, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x):
        _, r, c, f = self.shape 
        squared = backend.square(x)
        pooled = backend.pool2d(squared, (self.n, self.n), strides=(1,1), padding="same", pool_mode='avg')
        summed = backend.sum(pooled, axis=3, keepdims=True)
        averaged = self.alpha * backend.repeat_elements(summed, f, axis=3)
        denom = backend.pow(self.k + averaged, self.beta)
        return x / denom 
    
    def compute_output_shape(self, input_shape):
        return input_shape 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model = Sequential()

model.add(Conv2D(
    input_shape=(224,224,3),
    kernel_size=(11,11),
    filters=96,
    strides=(4,4),
    padding='valid',
    activation='relu'))


model.add(LocalResponseNormalization())
model.add(MaxPooling2D(
    pool_size=(5,5),
    strides=(1,1),
    padding='valid'))
model.add(Conv2D(
    filters=256,
    kernel_size=(5,5),
    padding='valid',
    activation='relu'))


model.add(LocalResponseNormalization())
model.add(MaxPooling2D(
    pool_size=(3,3),
    strides=(1,1),
    padding='valid'))
model.add(Conv2D(
    filters=384,
    kernel_size=(3,3),
    padding='valid',
    activation='relu'))


model.add(
    Conv2D(
        filters=384,
        kernel_size=(3,3),
        padding='valid',
        activation='relu'))


model.add(Conv2D(
    filters=256,
    kernel_size=(3,3),
    padding='valid',
    activation='relu'))


model.add(Flatten())


model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(units=1000, activation='relu'))


model.add(Dense(units=1000, activation='softmax'))


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary(line_length=72, positions=[.5, .86, 1., 1.])
model.save('./models/AlexNet.no-division.model')
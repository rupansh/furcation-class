import numpy as np
import tensorflow as tf
from tensorflow import keras

from loader import load_train_data

def preprocess(x, y):
    xp = keras.applications.resnet50.preprocess_input(x)
    yp = keras.utils.to_categorical(y, 2)

    return xp, yp

# Input Images => 200x RGB 256x256
# Ouput => True/False
X, Y = load_train_data()
X, Y = preprocess(X, Y) 

input_tensor = keras.Input(shape=X.shape[1:])
resnet = keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=input_tensor)

# Freeze all blocks except last
for l in resnet[:143]:
    l.trainable = False

model = keras.models.Sequential()
model.add(keras.layers.Lambda(lambda im: tf.image.resize(im, (256, 256))))
model.add(resnet)
model.add(keras.layers.Flatten())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])

history = model.fit(X, Y, batch_size=128, epochs=2, verbose=1)

model.save("split.h5")
import numpy as np
import tensorflow as tf
from tensorflow import keras

from loader import load_train_data

def preprocess(x, y):
    xp = keras.applications.resnet50.preprocess_input(x)

    return xp, y

print("loading training data")
X, Y = load_train_data(1000)
X, Y = preprocess(X, Y) 
print("loaded training data!")

print(X.shape)
input_tensor = keras.Input(shape=X.shape[1:])
resnet = keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=input_tensor)

# Freeze all blocks except last
for l in resnet.layers[:143]:
    l.trainable = False

model = keras.models.Sequential()
#model.add(keras.layers.Lambda(lambda im: tf.image.resize(im, (224, 224))))
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
#model.add(keras.layers.Dense(4, activation='sigmoid'))
model.add(keras.layers.Dense(3, activation="softmax"))

model.compile(#loss="mse" ,
                loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(learning_rate=0.01),
                metrics=['accuracy'])

history = model.fit(X, Y, batch_size=4, epochs=5, verbose=1)

model.save("split.h5")
print("done")

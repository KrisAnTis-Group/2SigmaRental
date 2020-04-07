import json
import numpy as np
from src import dataModifier as DM
from keras.preprocessing.text import Tokenizer

dataJS = DM.json_load("data/train.json/train.json")

TextTypes = ["description"]
AnswerTypes = ["interest_level"]

tupeConvert = {"interest_level": {"low": 0, "medium": 1, "high": 2}}

TextData = np.asarray(DM.get_arr(dataJS, TextTypes))
AnswerData = np.array(
    DM.get_arr(DM.modifier_fiches_type(dataJS, tupeConvert), AnswerTypes))

X = TextData
Y = DM.to_one_hot(AnswerData)

#нормализация
np.random.seed(2)

indices = DM.mixedIndex(X)
X = X[indices]
Y = Y[indices]

val_split = int(X.shape[0] * 0.6)

X_train = X[:val_split]
X_val = X[val_split:]
Y_train = Y[:val_split]
Y_val = Y[val_split:]

tokinizer = Tokenizer(num_words=3000)
tokinizer.fit_on_texts(X_train)
sequences = tokinizer.texts_to_sequences(X_train)

one_hot_results = tokinizer.texts_to_matrix(X_train, mode="binary")
X_train = np.array(one_hot_results)
X_train = np.asarray(X_train).astype('int')

tokinizer = Tokenizer(num_words=3000)
tokinizer.fit_on_texts(X_val)
sequences = tokinizer.texts_to_sequences(X_val)

one_hot_results = tokinizer.texts_to_matrix(X_val, mode="binary")
X_val = np.array(one_hot_results)
X_val = np.asarray(X_val).astype('int')

from keras import models
from keras import layers
from keras import regularizers
from keras.optimizers import RMSprop

model = models.Sequential()
model.add(
    layers.Dense(100, activation="relu", input_shape=(X_train.shape[1], )))
model.add(layers.Dropout(0.15))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.1))

#model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(3, activation="softmax"))

model.compile(optimizer=RMSprop(lr=2e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train,
                    Y_train,
                    epochs=35,
                    batch_size=128,
                    validation_data=(X_val, Y_val))

# model.save_weights('Dense_model.h5')
# графики изменения качества модели

import matplotlib.pyplot as plt

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation val_loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()

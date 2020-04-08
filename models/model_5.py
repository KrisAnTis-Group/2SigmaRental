import json
import numpy as np
from src import dataModifier as DM
from keras.preprocessing.text import Tokenizer

dataJS = DM.json_load("data/train.json/train.json")

DigitTypes = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]
TextTypes = ["description"]
FullTimeTypes = ["created"]
AnswerTypes = ["interest_level"]

tupeConvert = {"interest_level": {"low": 0, "medium": 1, "high": 2}}

DigitData = np.array(DM.get_categories(dataJS, DigitTypes))
FullTimeData = DM.get_categories(dataJS, FullTimeTypes)
Data = DM.data_to_MD(DM.fullData_to_data(FullTimeData))
Time = DM.time_to_HMS(DM.fullData_to_time(FullTimeData))
TextData = np.asarray(DM.get_arr(dataJS, TextTypes))
AnswerData = np.array(
    DM.get_arr(DM.modifier_fiches_type(dataJS, tupeConvert), AnswerTypes))

X_1 = np.column_stack((DigitData, Data, Time))

X_1 = np.asarray(X_1).astype('float32')
X_1 = DM.normalization(X_1)
X_2 = TextData

Y = DM.to_one_hot(AnswerData)

#нормализация

indices = DM.mixedIndex(X_1)
X_1 = X_1[indices]
X_2 = X_2[indices]
Y = Y[indices]

tokinizer = Tokenizer(num_words=3000)
tokinizer.fit_on_texts(X_2)
sequences = tokinizer.texts_to_sequences(X_2)

one_hot_results = tokinizer.texts_to_matrix(X_2, mode="binary")
X_2 = np.array(one_hot_results)
X_2 = np.asarray(X_2).astype('int')

from keras import models
from keras.models import Model
from keras import layers
from keras import Input
from keras import regularizers
from keras.models import load_model
from keras.optimizers import RMSprop

digit_input = Input(shape=(X_1.shape[1], ))
dense_digit_layer_1 = layers.Dense(32, activation='relu')(digit_input)
dense_digit_layer_1 = layers.Dropout(0.5)(dense_digit_layer_1)
dense_digit_layer_1 = layers.Dense(16, activation='relu')(dense_digit_layer_1)
dense_digit_layer_1 = layers.Dropout(0.3)(dense_digit_layer_1)
dense_digit_layer_1 = layers.Dense(8, activation='relu')(dense_digit_layer_1)

text_input = Input(shape=(X_2.shape[1], ))
dense_text_layer_2 = layers.Dense(64, activation='relu')(text_input)
dense_text_layer_2 = layers.Dropout(0.5)(dense_text_layer_2)
dense_text_layer_2 = layers.Dense(32, activation='relu')(dense_text_layer_2)
dense_text_layer_2 = layers.Dropout(0.3)(dense_text_layer_2)
dense_text_layer_2 = layers.Dense(8, activation='relu')(dense_text_layer_2)

concatenated = layers.concatenate([dense_digit_layer_1, dense_text_layer_2],
                                  axis=-1)

conc_layrs = layers.Dense(8, activation='relu')(concatenated)
conc_layrs = layers.Dropout(0.2)(conc_layrs)
conc_layrs = layers.Dense(8, activation='relu')(conc_layrs)
answer = layers.Dense(3, activation='softmax')(conc_layrs)

model = Model([digit_input, text_input], answer)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit([X_1, X_2], Y, epochs=15, batch_size=100)

result = model.evaluate([X_1, X_2], Y)

print(result)

model.save('models/weights/model5.h5')
#графики изменения качества модели

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation val_loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

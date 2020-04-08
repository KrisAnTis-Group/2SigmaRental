import json
import numpy as np
from keras import models
from keras import layers
from keras import regularizers
from keras.models import load_model
from keras.optimizers import RMSprop
from models.src import dataModifier as DM
from keras.preprocessing.text import Tokenizer

dataJS = DM.json_load("data/train.json/train.json")

DigitTypes = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]
FullTimeTypes = ["created"]
TextTypes = ["description"]
AnswerTypes = ["interest_level"]

tupeConvert = {'interest_level': {'low': 0, 'medium': 1, 'high': 2}}

target = np.array(
    DM.get_arr(DM.modifier_fiches_type(dataJS, tupeConvert), AnswerTypes))
#----------------------model1----------------------
X = np.array(DM.get_categories(dataJS, DigitTypes))
Y = target

X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('int')

indices = DM.mixedIndex(X)
X = X[indices]
Y = Y[indices]

#нормализация
X = DM.normalization(X)
Y = DM.to_one_hot(Y)

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X.shape[1], )))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.05))
#model.add(layers.Dense(32,activation='relu'))
#model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer=RMSprop(lr=1e-3), loss='mse', metrics=['mae'])

history = model.fit(X, Y, epochs=33, batch_size=64)

model.save('models/weights/model1.h5')

#----------------------model2----------------------

DigitData = np.array(DM.get_categories(dataJS, DigitTypes))
FullTimeData = DM.get_categories(dataJS, FullTimeTypes)
Y = target

Data = DM.data_to_days(DM.fullData_to_data(FullTimeData))
Time = DM.time_to_sec(DM.fullData_to_time(FullTimeData))

X = np.column_stack((DigitData, Data, Time))

X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('int')

indices = DM.mixedIndex(X)
X = X[indices]
Y = Y[indices]

X = DM.normalization(X)
Y = DM.to_one_hot(Y)

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X.shape[1], )))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(32, activation='relu'))
#model.add(layers.Dense(32,activation='relu'))
#model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer=RMSprop(lr=5e-4), loss='mse', metrics=['mae'])

history = model.fit(X, Y, epochs=31, batch_size=90)

model.save('models/weights/model2.h5')

#----------------------model3----------------------

DigitData = np.array(DM.get_categories(dataJS, DigitTypes))
FullTimeData = DM.get_categories(dataJS, FullTimeTypes)
Y = target

Data = DM.data_to_MD(DM.fullData_to_data(FullTimeData))
Time = DM.time_to_HMS(DM.fullData_to_time(FullTimeData))
X = np.column_stack((DigitData, Data, Time))

X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('int')

indices = DM.mixedIndex(X)
X = X[indices]
Y = Y[indices]

#нормализация
X = DM.normalization(X)
Y = DM.to_one_hot(Y)

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X.shape[1], )))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.02))
#model.add(layers.Dense(32,activation='relu'))
#model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer=RMSprop(lr=5e-4), loss='mse', metrics=['mae'])

history = model.fit(X, Y, epochs=35, batch_size=100)

model.save('models/weights/model3.h5')

#----------------------model4----------------------

TextData = np.asarray(DM.get_arr(dataJS, TextTypes))
AnswerData = target

X = TextData
Y = DM.to_one_hot(AnswerData)

indices = DM.mixedIndex(X)
X = X[indices]
Y = Y[indices]

tokinizer = Tokenizer(num_words=3000)
tokinizer.fit_on_texts(X)
sequences = tokinizer.texts_to_sequences(X)

one_hot_results = tokinizer.texts_to_matrix(X, mode="binary")
X = np.array(one_hot_results)
X = np.asarray(X).astype('int')

model = models.Sequential()
model.add(layers.Dense(100, activation="relu", input_shape=(X.shape[1], )))
model.add(layers.Dropout(0.15))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.1))

#model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer=RMSprop(lr=2e-5), loss='mse', metrics=['mae'])

history = model.fit(
    X,
    Y,
    epochs=35,
    batch_size=128,
)

model.save('models/weights/model4.h5')
import json
import numpy as np
from src import dataModifier as DM

dataJS = DM.json_load("data/train.json/train.json")

ModelTypesX = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "created"]
ModelTypesY = ["interest_level"]

tupeConvert = {
    'interest_level':{
        'low': 0,
        'medium': 1,
        'high': 2
    }
}

X = DM.get_categories(dataJS,ModelTypesX)
X = np.array(X)
Y = DM.get_categories(DM.modifier_fiches_type(dataJS,tupeConvert),ModelTypesY)
Y = np.array(Y) 

FullData = X[:,-1:]
X = X[:,:-1]

Data = DM.data_to_MD(DM.fullData_to_data(FullData))
Time = DM.time_to_HMS(DM.fullData_to_time(FullData))
X = np.column_stack((X, Data, Time))



X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('int') 

#нормализация
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std



np.random.seed(2)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]






def to_one_hot(labels, demension=3):
    results = np.zeros((len(labels),demension))
    for i, label in enumerate(labels):
        results[i,label]=1
    return results

Y = to_one_hot(Y)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(X.shape[1],)))
model.add(layers.Dense(32,activation='relu'))
#model.add(layers.Dense(32,activation='relu'))
#model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X, Y, epochs=35, batch_size=128, validation_split=0.4)




#model.save_weights('Dense_model.h5')

#графики изменения качества модели

import matplotlib.pyplot as plt

acc=history.history['accuracy']
val_acc = history.history['val_accuracy']
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


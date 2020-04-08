
#%%
# импорт библиотек
import json
import numpy as np
from models.src import dataModifier as DM
from keras import models
from keras.models import Model
from keras import layers
from keras import regularizers
from keras.models import load_model
from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer
#%%
# открываем файл для прогнозирования целей по данным фичам
dataJS = DM.json_load("data/test.json/test.json")

# выделим различные типы данные, выделяемые из предоставленных фич
DigitTypes = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]
FullTimeTypes = ["created"]
TextTypes = ["description"]

# итоговое решение ансамблируется из 5 моделей, получим прогнозы для каждой из моделей
#%%
# модель 1 
# # состоит только из прямых числовых данных: количество ванных и спальных комнат,
# долгота и широта, а также цена

# парсим фичи из данных
DigitData = np.array(DM.get_categories(dataJS, DigitTypes))

m1_in = DigitData
# выполняем нормализацию данных
m1_in = DM.normalization(m1_in)
# загружаем обученную модель 1 и делаем прогноз
model1 = load_model('models/weights/model1.h5')
preds_1 = model1.predict(m1_in)
# добавляем прогноз в сумму общего решения
preds = preds_1

del model1
del preds_1
del m1_in
#%%
# модель 2
# к числовым данным добавляется дата создания объявления
# в данной модели было решено вычислить две фичи:
# суммарная дата в днях
# суммарное время в секундах

FullTimeData = DM.get_categories(dataJS, FullTimeTypes)

DataSum = DM.data_to_days(DM.fullData_to_data(FullTimeData))
TimeSum = DM.time_to_sec(DM.fullData_to_time(FullTimeData))

# объединяем фичи и нормализуем их
m2_in = np.column_stack((DataSum, TimeSum))
m2_in = np.asarray(m2_in).astype('float32')
m2_in = DM.normalization(m2_in)
m2_in = np.column_stack((DigitData, m2_in))

model2 = load_model('models/weights/model2.h5')
preds_2 = np.array(model2.predict(m2_in))
# добавляем прогноз в общее решение
preds += preds_2

del model2
del m2_in
del preds_2
del DataSum
del TimeSum
#%%
# модель 3
# в данной модели дата представляется в виде набора фич вида
# месяц, день, час, минута, секунда
# год - игнорируется, т.к. он один и тот же для всего набора данных
FullTimeData = DM.get_categories(dataJS, FullTimeTypes)

DataMD = DM.data_to_MD(DM.fullData_to_data(FullTimeData))
TimeHMS = DM.time_to_HMS(DM.fullData_to_time(FullTimeData))

m3_in = np.column_stack((DataMD, TimeHMS))
m3_in = np.asarray(m3_in).astype('float32')
m3_in = DM.normalization(m3_in)
m3_in = np.column_stack((DigitData, m3_in))

model3 = load_model('models/weights/model3.h5')
preds_3 = np.array(model3.predict(m3_in))

preds += preds_3

del model3
del preds_3
del DataMD
del TimeHMS
del DigitData
#%%
#  модель 4
# данная модель состоит лишь из одной фичи - описания
# текстовое представление данных выполняется самым простым прямым кодированием в бинарные разреженные вектора

# получаем массив текстовых описаний
TextData = DM.get_arr(dataJS, TextTypes)

# определяем, что кодировка будет происходить по 3000 наиболее встречаемым слов
tokinizer = Tokenizer(num_words=3000)
# происходит процесс токенизации и последующим получением матрицы разреженных векторов
tokinizer.fit_on_texts(TextData)
sequences = tokinizer.texts_to_sequences(TextData)

one_hot_results = tokinizer.texts_to_matrix(TextData, mode="binary")
TextData = np.array(one_hot_results)
TextData = np.asarray(TextData).astype('int')

model4 = load_model('models/weights/model4.h5')
m4_in = TextData
preds_4 = np.array(model4.predict(m4_in))

preds += preds_4

del model4
del preds_4
del TextData
#%%
# модель 5
# модель включает в себя две группы входных данных:
# все фичи, использованные в третьей моделе + текст
# две группы входов вычисляются в сети отдельно, а затем конкатенируются в единый выход
model5 = models.load_model('models/weights/model5.h5')
m5_in = [m3_in, m4_in]
preds_5 = np.array(model5.predict(m5_in))

preds += preds_5
#%%
# ансамблируем итоговое решение
final_preds = 0.2 * preds

# берём из "файла примера" листинги объявлений
listing_id = np.loadtxt("data/sample_submission.csv/sample_submission.csv",
                        skiprows=1,
                        delimiter=",")

# технические детали оформления выходного файла
listing = []
for q in listing_id[:, :1]:
    listing.append(q[0])

listing_id = np.array(listing)
listing_id = np.asarray(listing_id).astype('int')

final_preds = np.asarray(final_preds).astype('float32')

final_preds = np.column_stack(
    (final_preds[:, 2], final_preds[:, 1], final_preds[:, 0]))

# сохраняем вычесленные вероятности с заданным форматом
np.savetxt('out.csv',
           np.column_stack((listing_id, final_preds)),
           fmt='%2d,%1.17f,%1.17f,%1.17f',
           delimiter=',')


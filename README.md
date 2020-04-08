# SigmaRental
Материалы тестового задания: 
https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/overview

## Инструкция по запуску
### Чтобы запустить скрипт на вашей машине:

1) Cклонируйте репозиторий:

`git clone https://github.com/KrisAnTis-Group/2SigmaRental.git`

2) В терминале выполните команду:

`pip install -r requirements.txt`

3) В папке models имеются 5 моделей нс. В репозиторий уже загружены обученные модели и находятся в папке `models/weights`. Если вы хотите поправить парметры какой-либо модели вы можете это сделать, обратившись к одному из скриптов:
`models/model_1.py`
`models/model_2.py`
`models/model_3.py`
`models/model_4.py`
`models/model_5.py`

Не забывайте сохранять ваши модели `*.h5`. После того как все модели сохранены, в работу вступает скрипт: `predict.py` - здесь находятся все модели, которые учавствуют в итоговом ансамбле. Здесь же вы сохраняете спрогшнозированные данные в файл: `out.csv`

#### Замечание: Результатом работы считается файл `predict.py` - он объединяет все созданные модели

### Чтобы запустить ноутбук на [Google Colab](https://colab.research.google.com):

0) Откройте [Google Colab](https://colab.research.google.com)

1) Скачайте ноутбук (вкладка Github), затем прописываете адрес репозитория.

2) Чтобы выкачать на colab библиотеку, не забудьте выполнить команду в первой ячейке:

```
import sys; sys.path.append('/content/SigmaRental')
!git clone https://github.com/KrisAnTis-Group/SigmaRental.git && pip install -r SigmaRental/requirements.txt
```

3) Позаботьтесь о входных данных `test.json`, которые требуются для обучения моделей. Файл можно загрузить в colab с помощью команд:

```
from google.colab import files
uploaded = files.upload()
```
Не забудьте изменить путь до файла и его имя (при необходимости) в строке: 

```
# открываем файл для прогнозирования целей по данным фичам
dataJS = DM.json_load("data/test.json/test.json")
```
4) Не забудьте настроить `device='cpu'` (модель работает на cpu - установлено по умолчанию на Google Colab), а также выбрать подходящий Runtime в Google Colab (CPU/TPU/GPU).

5) Запустите ноутбук.

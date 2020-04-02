import json
from collections import OrderedDict

with open("data/train.json/train.json", "r") as read_file:
    data = json.load(read_file)


NData = {}
for k in data:
    for q in data[k]:
        if q in NData:
            NData[q].append(data[k][q])
        else:
            NData[q] = []
            NData[q].append(data[k][q])

#Data = []
#for q in NData:
 #    Data.append(NData[q])

Model_1_types = ["bathrooms", "bedrooms", "created", "latitude", "longitude", "price", "interest_level"]

data["bathrooms"]= {int(k):float(v) for k,v in data["bathrooms"].items()}
data["bedrooms"]= {int(k):int(v) for k,v in data["bedrooms"].items()}
data["interest_level"]= {int(k):str(v) for k,v in data["interest_level"].items()}
data["latitude"]= {int(k):float(v) for k,v in data["latitude"].items()}
data["longitude"]= {int(k):float(v) for k,v in data["longitude"].items()}
#отсортирует по возрастанию ключей словаря

print(data["bathrooms"])

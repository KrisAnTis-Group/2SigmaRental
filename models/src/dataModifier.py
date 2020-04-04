import json
import re
import numpy as np

def json_load(path = "data/train.json/train.json"):

    with open(path, "r") as read_file:
        dataJS = json.load(read_file)

    return dataJS

def get_categories(dataJS, types):
    
    Data = {}
    Rez = []
    for k in types:
        for q in dataJS[k]:
            if not q in Data:
                Data[q] = []
            Data[q].append(dataJS[k][q])
    for k in Data:
        Rez.append(Data[k])
        
    return Rez

def modifier_fiches_type (dataJS, tupeConvert):

    for k in tupeConvert:
        for q in dataJS[k]:
            dataJS[k][q] = tupeConvert[k][dataJS[k][q]]

    return dataJS

def fullData_to_data (fullData):
    #'2016-06-16 05:55:27'
    Days = []
    for d in fullData:
        full2 = re.findall('\d{2}',d[0])
        t = full2[0] + full2[1] + '-'+ full2[2] +'-'+ full2[3]
        Days.append(t)

    return np.array(Days)


def fullData_to_time (fullData):
    #'2016-06-16 05:55:27'
    Time = []
    for d in fullData:
        full2 = re.findall('\d{2}',d[0])
        t = full2[4] +'-'+ full2[5] +'-'+ full2[6]
        Time.append(t)

    return np.array(Time)

def data_to_days (DataStr):

    Days = []
    for d in DataStr:
        full2 = re.findall('\d{2}',d)

        D = int(full2[3])
        D += int(full2[2])*30
        Y = (int(full2[0])*100+int(full2[1]))
        D += Y*365

        Days.append(D)

    return np.array(Days)

def time_to_sec (TimeStr):
    Time = []
    for d in TimeStr:
        full2 = re.findall('\d{2}',d)

        T = int(full2[2])
        T += int(full2[1])*60
        T += int(full2[0])*3600

        Time.append(T)

    return np.array(Time)

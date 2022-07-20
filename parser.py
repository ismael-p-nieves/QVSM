import re
import csv
from tkinter import E
import numpy
import math
import random
import time
from ast import literal_eval
import threading
import mmap

CONFIDENCE_DEGREE = 2.575
SEED = 10

def readDataset(path):
    rows = ''
    with open(path) as file:
        reader = csv.reader(file, delimiter='\n')
        rows = []
        for row in reader:
            msg = row[0].split(',')
            rows.append(msg)

    return rows

def readSample(path):
    sample = []
    with open(path) as file:
        reader = csv.reader(file)

        for element in reader:
            parameters_string, flag = element
            parameters = literal_eval(parameters_string)

            row = []
            for i in range(0, 12):
                row.append(int(parameters[i]))

            sample.append([row, int(flag)])

    return sample

        
def countFalses(path):
    data = readDataset(path)
    pattern = 'T'
    falses = len(re.findall(pattern, data))
    return falses

def printFalses(path):
    data = readDataset(path)
    pattern = '([0-9a-f.]*,)*T'
    falses = re.findall(pattern, data)
    for match in falses:
        print(match)

def constructDataset(path):
    time1 = time.time()
    rows = readDataset(path)
    messages = []
    for msg in rows:
        if len(msg) == 12:
            first_parameter = msg[0].split('.')
            features = [[int(first_parameter[0]), int(first_parameter[1])]]
            features[0].append(hex2Decimal(list(msg[1])))
            features[0].append(int(msg[2]))
            for j in range(3, 11):
                value = hex2Decimal(list(msg[j]))
                features[0].append(value)
            if msg[-1] == 'R':
                features.append(1)
            else:
                features.append(-1)
            messages.append(features)

    time2 = time.time()
    print('Tiempo de construcción de dataset: ' + str(time2 - time1))

    return messages
        
def intervaloConfianza(messages, ressources, size):
    experimentos = []

    time1 = time.time()

    for i in range(ressources):
        poblacion = messages.copy()
        muestra = 0
        falses = 0
        while falses < size:
            msg = random.choice(poblacion)
            poblacion.remove(msg)
            if msg[1] == -1:
                falses += 1
            muestra += 1

        experimentos.append(muestra)

    media = numpy.mean(experimentos)
    varianza = numpy.var(experimentos)
    delta = CONFIDENCE_DEGREE * math.sqrt(varianza / ressources)

    time2 = time.time()
    print('Tiempo por recurso: ' + str(time2 - time1))

    return media + delta

def hex2Decimal(digits):
    value = 0
    for position, bit in zip(range(len(digits)), digits):
        if bit >= 'a' and bit <= 'f':
            base = (ord(bit) - 87) #ASCII('a') - 10
        else:
            base = (int(bit))
        value += base*(16**position)
        
    return value

def createSample(dataset, size):
    sample = []
    poblacion = dataset.copy()
    i = 0
    while i < size:
        random.seed(SEED)
        msg = random.choice(poblacion)
        poblacion.remove(msg)
        sample.append(msg)
        i += 1

    return sample

def writeSample(dataset, size, path):
    sample = createSample(dataset, size)
    with open(path, 'w', newline= '') as file:
        writer = csv.writer(file)
        for msg in sample:
            writer.writerow(msg)

#sample = readSample('decreased_dataset.csv')
#print('Dataset leído')

#size_250 = intervaloConfianza(sample, 101, 250)
#print('Tamaño del sample: ' + str(size_250))

#writeSample(sample, size_250, 'sample_250.csv')
#print('Sample creado')
from sqlite3 import Time
from generador import *
import numpy as np
from sklearn import svm
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import svm
import parser
from datetime import datetime

N_SAMPLES = 100
N_FEATURES = 12


def get_training(data, percentage_training, n_samples):
    m = int(percentage_training * n_samples)

    x_train = np.array([data[i][0] for i in range(m)])
    y_train = np.array([data[i][1] for i in range(m)])

    x_test = np.array([data[i][0] for i in range(m, n_samples)])
    y_test = np.array([data[i][1] for i in range(m, n_samples)])

    return x_train, y_train, x_test, y_test

def readJSON(path):
    with open(path) as file:
        try:
            jsoned = json.load(file)
            issues = jsoned['issues']
            data, n_features = createDataset(issues)
        except:
            print('The state is incorrect.')

    return data, n_features

def getMessagesArray(path):
    messages = parser.readSample(path)
    return messages, len(messages[0][0])

#data, N_FEATURES = getMessagesArray('dataset_100.csv')
time1 = datetime.now()
data = parser.readSample('datasets\sample_250.csv')
print(len(data))
x_train, y_train, x_test, y_test = get_training(data, 0.5, len(data))

n = len(x_train[0])

@qml.template
def feature_map(x):
    
    # ZZMap modificado

    for i in range(N_FEATURES):
        qml.Hadamard(wires = i)
        qml.RZ(2*x[i], wires = i)
        if(i > 0):
            qml.CNOT(wires = [i-1,i])
            
    
dev = qml.device("default.qubit", wires = n)      
@qml.qnode(dev)
def circuit(x,y):
    feature_map(y)
    qml.adjoint(feature_map)(x)
    return qml.probs(wires = range(n))

# devuelve el valor final del producto interno
def scalar_product(x,y):
    probs = circuit(x,y)
    return probs[0]

def KernelGramMatrixFull(X1, X2):
    print("Calculando matriz de Gram")

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        print(int(i / len(X1) *100), "%")
        for j, x2 in enumerate(X2):
            
            x1 = x1.flatten()
            x2 = x2.flatten()
            
            gram_matrix[i, j] = scalar_product(x1,x2)
            
    return gram_matrix

clf = svm.SVC(kernel="precomputed")

matrix = KernelGramMatrixFull(x_train,x_train)

print("Entrenando...")
clf.fit(matrix, y_train)

#test
print("Comprobando con test...")

sol = clf.predict(KernelGramMatrixFull(x_test,x_train))

success = 0
for i in range(len(y_test)):
    if sol[i] == y_test[i]:
        success += 1
        
time2 = datetime.now()
print("Precisión del test: ", success/len(sol)*100, "%")
print("Tiempo: " + str(time2 - time1))
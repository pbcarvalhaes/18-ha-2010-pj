from cProfile import label
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from os import listdir
from os.path import isfile, join
import scipy.io as sio


dataPath = "alternative/data/"
scale = 1000

def removeZeroPadding(arr: np.ndarray):
    for row in arr:
        i = len(row)-1
        while (row[-i]==0):
            i-=1
        i+=1
        copyIndex=0
        for j in range(i,len(row)):
            row[j]=row[copyIndex]
            copyIndex+=1

def load_references(folder: str = dataPath) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(folder+'ecgLeads.npy')
    labels = np.load(folder+'ecgLabels.npy')

    return (arr,labels)

def load_alternative_encoder() -> np.ndarray:
    encoderModel = np.load('encoder.npy', allow_pickle=True)

    encoder = np.copy(encoderModel)

    for i in range(1, len(encoderModel)):
        encoder[i] = 'A'
    
    return encoder

def matFiles2Array(path: str, label: str):
    # loading arrats from csv
    fileNames = [f for f in listdir(path) if isfile(join(path, f))]


    data = sio.loadmat(path + fileNames[0])
    ecg_lead = data['sample']

    leads = np.zeros((len(fileNames),len(ecg_lead)))
    labels=np.full(len(fileNames),label)

    counter = 0
    for file in fileNames:
        data = sio.loadmat(path + file)
        ecg_lead = data['sample']
        leads[counter]=ecg_lead
        counter+=1

    return(leads,labels)

def main():
    norm = matFiles2Array('alternative/normal/', "N")
    arr = matFiles2Array('alternative/arrhythmia/', "A")


    leads= np.append(norm[0],arr[0], axis=0)
    labels= np.append(norm[1],arr[1], axis=0)

    np.save(dataPath+'ecgLeads.npy',leads)
    np.save(dataPath+'ecgLabels.npy',labels)

if __name__ == '__main__':
    main()
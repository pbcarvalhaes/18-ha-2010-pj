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
        while (row[-i] == 0):
            i -= 1
        i += 1
        copyIndex = 0
        for j in range(i, len(row)):
            row[j] = row[copyIndex]
            copyIndex += 1


def load_references_normal(folder: str = dataPath) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(folder+'ecgLeadsNormal.npy')
    labels = np.load(folder+'ecgLabelsNormal.npy')

    return (arr, labels)


def load_references_arrhythmia(folder: str = dataPath) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(folder+'ecgLeadsArrhythmia.npy')
    labels = np.load(folder+'ecgLabelsArrhythmia.npy')

    return (arr, labels)


def load_alternative_encoder() -> np.ndarray:
    encoderModel = np.load('encoder.npy', allow_pickle=True)

    encoder = np.copy(encoderModel)

    for i in range(1, len(encoderModel)):
        encoder[i] = 'A'

    return encoder


def matFiles2Array(path: str, label: str, limitedLength=True):
    # loading arrats from csv
    fileNames = [f for f in listdir(path) if isfile(join(path, f))]
    if (limitedLength):
        fileNames = fileNames[:5000]

    data = sio.loadmat(path + fileNames[0])
    ecg_lead = data['sample']

    leads = np.zeros((len(fileNames), len(ecg_lead)))
    labels = np.full(len(fileNames), label)

    counter = 0
    for file in fileNames:
        data = sio.loadmat(path + file)
        ecg_lead = data['sample']
        leads[counter] = ecg_lead.flatten()
        counter += 1

    return(leads, labels)


def main():
    norm = matFiles2Array('alternative/normal/', "N")
    arr = matFiles2Array('alternative/arrhythmia/', "A")


    np.save(dataPath+'ecgLeadsNormal.npy', norm[0])
    np.save(dataPath+'ecgLabelsNormal.npy', norm[1])
    np.save(dataPath+'ecgLeadsArrhythmia.npy', arr[0])
    np.save(dataPath+'ecgLabelsArrhythmia.npy', arr[1])
if __name__ == '__main__':
    main()

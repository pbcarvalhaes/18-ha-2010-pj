from cProfile import label
from typing import List, Tuple
import numpy as np
import pandas as pd

dataPath = "alternative/data/"

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


# loading arrats from csv
arrN = np.genfromtxt(dataPath+'ptbdb_normal.csv', delimiter=',')
arrAN = np.genfromtxt(dataPath+'ptbdb_abnormal.csv', delimiter=',')

# removing zero padding, by adding the beginning until the end
removeZeroPadding(arrN)
removeZeroPadding(arrAN)

labelsN=np.full(len(arrN),'N')
labelsAN=np.full(len(arrAN),'X')

arr= np.append(arrN,arrAN)
labels= np.append(labelsN,labelsAN)

np.save(dataPath+'ecgLeads.npy',arr)
np.save(dataPath+'ecgLabels.npy',labels)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import gc
import pickle
import time

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix
from prepareEcgLeads import load_references_normal, load_alternative_encoder, load_references_arrhythmia

import sys
sys.path.append("../18-ha-2010-pj_Team_PMR")
from dataPrep import extractFeatures


def printScore(y_pred, modelName:str = "No model Explicited"):
    counter = {
        "right": 0,
        'wrong': 0,
    }
    indexes = {
        "right": [],
        'wrong': [],
    }

    i = 0
    while i < len(ecg_labels):
        if (ecg_labels[i] == y_pred[i]):  # Right prediction
            counter['right'] += 1
            indexes['right'].append(i)
        else:  # wrong  prediction
            counter['wrong'] += 1
            indexes['wrong'].append(i)

        i += 1

    print(modelName)
    print("Total right: ", counter['right'])
    print("Total wrong: ", counter['wrong'])
    print("Accuracy: ", accuracy_score(ecg_labels, y_pred))
    print("F1: ", f1_score(ecg_labels, y_pred,average='weighted'))
    
    cf_matrix = confusion_matrix(ecg_labels, y_pred)
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


    print("--- %s seconds ---" % (time.time() - start_time))

def testXGBmodel(df, modelfilepath:str, modelName:str = "No XGB model Explicited"):
    start_time = time.time()
    model = xgb.XGBClassifier()
    model.load_model(modelfilepath)

    encoder = LabelEncoder()
    encoder.classes_ = load_alternative_encoder()

    y_pred = model.predict(df)
    y_pred = encoder.inverse_transform(y_pred)

    printScore(y_pred, modelName)

def testRFmodel(df, modelfilepath:str, modelName:str = "No XGB model Explicited"):
    start_time = time.time()
    model = RandomForestClassifier()
    model = pickle.load(open(modelfilepath, 'rb'))

    y_pred = model.predict(df)
    
    printScore(y_pred, modelName)

def testNNmodel(df, modelfilepath:str, modelName:str = "No XGB model Explicited"):
    start_time = time.time()
    model = MLPClassifier()
    model = pickle.load(open(modelfilepath, 'rb'))

    y_pred = model.predict(df)
    
    printScore(y_pred, modelName)

#   -----------------------------------------------------------------------------------------
#   Start Data preparation
start_time = time.time()

normal = load_references_normal()
arr = load_references_arrhythmia()
ecg_labels = np.append(normal[1], arr[1], axis=0)


df = extractFeatures(normal[0],128, False)
del normal
gc.collect()
df1 = extractFeatures(arr[0],360, False)
df = pd.concat([df, df1], axis=0)
del arr
del df1
gc.collect()


# df = pd.read_csv('alternative/data/featuresCsv.csv')

print("---Data prep %s seconds ---" % (time.time() - start_time))
#   End Data Preparation
#   -----------------------------------------------------------------------------------------
#   Start Machine Learning
start_time = time.time()

testXGBmodel(df,modelfilepath="models/XGB_model.txt", modelName="XGB hyperparameter selected")
testXGBmodel(df,modelfilepath="models/XGB_model2.txt", modelName="XGB no hyperparameter selected")
testRFmodel(df,modelfilepath="models/RF_model.pkl", modelName="RF hyperparameter selected")
testRFmodel(df,modelfilepath="models/RF_model2.pkl", modelName="RF no hyperparameter selected")
testNNmodel(df,modelfilepath="models/NN.pkl", modelName="NN selected")
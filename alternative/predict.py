import gc
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from prepareEcgLeads import load_references_normal, load_alternative_encoder, load_references_arrhythmia
from hrv import HRV
from ecgdetectors import Detectors


def extractFeatures(ecg_leads, frequency):
    detectors = Detectors(frequency)    

    # Initialisierung der Feature-Arrays für Stationary Wavelet Transform Detector Detecto 
    sdnn_swt = np.array([])                                       
    hr_swt = np.array([])                                  
    pNN20_swt = np.array([])                                  
    pNN50_swt = np.array([])      
    fAnalysis_swt = np.array([])   

    # Signale verarbeitung
    hrv_class = HRV(frequency)

    arr = np.zeros((len(ecg_leads),6))
    print("Ecg leads to prapare: {}".format(len(ecg_leads)))
    for idx, ecg_lead in enumerate(ecg_leads): 
        r_peaks_swt = detectors.swt_detector(ecg_lead)              # Detektion der QRS-Komplexe mit Stationary Wavelet Transform Detector

    #   sdnn_swt = np.std(np.diff(r_peaks_swt)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Stationary Wavelet Transform
        sdnn_swt = hrv_class.SDNN(r_peaks_swt, True)
        rmssd_swt = hrv_class.RMSSD(r_peaks_swt)
        sdsd_swt = hrv_class.SDSD(r_peaks_swt)
        NN20_swt = hrv_class.NN20(r_peaks_swt)
        NN50_swt = hrv_class.NN50(r_peaks_swt)

    #   fAnalysis_swt
        arr[idx][0] = idx
        arr[idx][1] = sdnn_swt
        arr[idx][2] = rmssd_swt
        arr[idx][3] = sdsd_swt
        arr[idx][4] = NN20_swt
        arr[idx][5] = NN50_swt  

        if (idx % 1000) == 0:
            print(str(idx) + "\t Ecg signals processed.")
        
    # Save data with pandas
    df = pd.DataFrame(arr, columns = ['index', 'SDNN', 'RMSSD', 'SDSD', 'NN20', 'NN50'])
    df.drop(['index'],axis=1, inplace = True)
    df.rename(columns = {'level_0':'index'}, inplace = True)
    df.dropna(inplace=True)

    return df

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
    print("Accuracy: ", counter["right"]/(counter["right"]+counter["wrong"]))
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

#   -----------------------------------------------------------------------------------------
#   Start Data preparation
start_time = time.time()

normal = load_references_normal()
arr = load_references_arrhythmia()
ecg_labels = np.append(normal[1], arr[1], axis=0)


df = extractFeatures(normal[0],128)
del normal
gc.collect()
df1 = extractFeatures(arr[0],360)
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
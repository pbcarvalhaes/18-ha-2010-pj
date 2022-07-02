# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
import pandas as pd
from typing import List, Tuple
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from hrv import HRV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


# Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden


def predict_labels(ecg_leads: List[np.ndarray], fs: float, ecg_names: List[str], model_name: str = 'model.npy', is_binary_classifier: bool = False) -> List[Tuple[str, str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

# ------------------------------------------------------------------------------
# Euer Code ab hier

    fs = 300                                                  # Sampling-Frequenz 300 Hz


    # Signale verarbeitung
    detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
    hrv_class = HRV(fs)
    

    detections = np.zeros((len(ecg_leads),8))

    for idx, ecg_lead in enumerate(ecg_leads):

        if len(ecg_lead) != 18000:                                  # Length normalization
            m = 18000 // len(ecg_lead)
            new_ecg_lead = ecg_lead
            for i in range(0, m-1):
                new_ecg_lead = np.append(new_ecg_lead, ecg_lead)
            r = 18000 - len(new_ecg_lead)
            new_ecg_lead = np.append(new_ecg_lead, ecg_lead[:r])
            ecg_leads[idx] = new_ecg_lead
    
        r_peaks_swt = detectors.swt_detector(ecg_lead)              # Detektion der QRS-Komplexe mit Stationary Wavelet Transform Detector

    #   sdnn_swt = np.std(np.diff(r_peaks_swt)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Stationary Wavelet Transform
        sdnn_swt = hrv_class.SDNN(r_peaks_swt, True)
        rmssd_swt = hrv_class.RMSSD(r_peaks_swt)
        sdsd_swt = hrv_class.SDSD(r_peaks_swt)
        NN20_swt = hrv_class.NN20(r_peaks_swt)
        NN50_swt = hrv_class.NN50(r_peaks_swt)
        HR_swt_mean = np.mean(hrv_class.HR(r_peaks_swt))
        succ_diffs_mean = np.mean(hrv_class._succ_diffs(r_peaks_swt))
        

    #   fAnalysis_swt
        detections[idx][0] = idx
        detections[idx][1] = sdnn_swt
        detections[idx][2] = rmssd_swt
        detections[idx][3] = sdsd_swt
        detections[idx][4] = NN20_swt
        detections[idx][5] = NN50_swt
        detections[idx][6] = HR_swt_mean
        detections[idx][7] = succ_diffs_mean
        
        if (idx % 1000) == 0:
            print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    # Save data with pandas
    df = pd.DataFrame(detections, columns = ['index', 'SDNN', 'RMSSD', 'SDSD', 'NN20', 'NN50', 'HR mean', 'SD mean'])
    df.drop(['index'],axis=1, inplace = True)
    df.rename(columns = {'level_0':'index'}, inplace = True)
    df.dropna(inplace=True)



#   End Data Preparation
#   -----------------------------------------------------------------------------------------
#   Start Machine Learning

    path = "./models/{foo}".format(foo = model_name)

    if model_name == "XGB_model.txt" or model_name =="XGB_model2.txt":
        model = xgb.XGBClassifier()
        model.load_model(path)

        encoder = LabelEncoder()
        encoder.classes_ = np.load('encoder.npy', allow_pickle=True)

        y_pred = model.predict(df)
        y_pred = encoder.inverse_transform(y_pred)
        predictions = []
        counter = 0
        for row in y_pred:
            predictions.append(
                (
                    ecg_names[counter],
                    row
                )
            )
            counter += 1
    elif model_name == "RF_model.pkl" or model_name =="RF_model2.pkl":
        model = RandomForestClassifier()
        model.load_model(path)

        y_pred = model.predict(df)
        y_pred = encoder.inverse_transform(y_pred)
        predictions = []
        counter = 0
        for row in y_pred:
            predictions.append(
                (
                    ecg_names[counter],
                    row
                )
            )
            counter += 1

# ------------------------------------------------------------------------------
    # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
    return predictions

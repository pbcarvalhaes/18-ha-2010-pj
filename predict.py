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
from sklearn.preprocessing import LabelEncoder

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
    detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
    detections = np.zeros((len(ecg_leads),3))
    counter = 0
    for ecg_lead in ecg_leads:
        r_peaks = detectors.hamilton_detector(ecg_lead)
        sdnn = np.std(np.diff(r_peaks)/fs*1000)
        detections[counter][0] = sdnn
        r_peaks = detectors.two_average_detector(ecg_lead)
        sdnn = np.std(np.diff(r_peaks)/fs*1000)
        detections[counter][1] = sdnn
        r_peaks = detectors.swt_detector(ecg_lead)
        sdnn = np.std(np.diff(r_peaks)/fs*1000)
        detections[counter][2] = sdnn

        counter = counter + 1
        if (counter % 1000)==0:
            print(str(counter) + "\t Dateien wurden verarbeitet.")

    df = pd.DataFrame(detections, columns=['ham', 'two', 'swt'])

    df['ham'].fillna(df['ham'].mean(),inplace=True)
    df['two'].fillna(df['two'].mean(),inplace=True)
    df['swt'].fillna(df['swt'].mean(),inplace=True)
#   End Data Preparation
#   -----------------------------------------------------------------------------------------
#   Start Machine Learning
    model = xgb.XGBClassifier()
    model.load_model("model.txt")

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
# ------------------------------------------------------------------------------
    # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
    return predictions

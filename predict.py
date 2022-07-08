# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import List, Tuple
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier


from dataPrep import extractFeatures


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

    fs = 300  # Sampling-Frequenz 300 Hz
    df = extractFeatures(ecg_leads, fs)  # getting features from leads


#   End Data Preparation
#   -----------------------------------------------------------------------------------------
#   Start Machine Learning

    path = "./models/{foo}".format(foo=model_name)

    # calling the right modedl for different types
    if model_name == "XGB_model.txt" or model_name == "XGB_model2.txt":
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
    elif model_name == "RF_model.pkl" or model_name == "RF_model2.pkl":
        model = RandomForestClassifier()
        model = pickle.load(open(path, 'rb'))

        y_pred = model.predict(df)
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
    elif model_name == "NN.pkl":
        model = MLPClassifier()
        model = pickle.load(open(path, 'rb'))

        y_pred = model.predict(df)
        predictions = []
        counter = 0
        for row in y_pred:
            predictions.append(
                (
                    ecg_names[counter],
                    row
                )
            )




# ------------------------------------------------------------------------------
    # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
    return predictions

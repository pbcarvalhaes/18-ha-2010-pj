import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from prepareEcgLeads import load_references

ecg_leads, ecg_labels = load_references()


fs = 125                                                  # Sampling-Frequenz 300 Hz
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
model.load_model("../model.txt")

encoder = LabelEncoder()
encoder.classes_ = np.load('encoder.npy', allow_pickle=True)

y_pred = model.predict(df)
y_pred = encoder.inverse_transform(y_pred)
predictions = []
# counter = 0
# for row in y_pred:
#     predictions.append(
#         (
#             ecg_names[counter],
#             row
#         )
#     )
#     counter += 1



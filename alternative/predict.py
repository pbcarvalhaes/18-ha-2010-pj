import time
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from prepareEcgLeads import load_references_normal, load_alternative_encoder, load_references_arrhythmia


def extractSDNN(ecg_leads, frequency):
    detectors = Detectors(frequency)
    detections = np.zeros((len(ecg_leads), 3))
    counter = 0
    for ecg_lead in ecg_leads:

        r_peaks = detectors.hamilton_detector(ecg_lead)
        sdnn = np.std(np.diff(r_peaks)/frequency*1000)
        detections[counter][0] = sdnn
        r_peaks = detectors.two_average_detector(ecg_lead)
        sdnn = np.std(np.diff(r_peaks)/frequency*1000)
        detections[counter][1] = sdnn
        r_peaks = detectors.swt_detector(ecg_lead)
        sdnn = np.std(np.diff(r_peaks)/frequency*1000)
        detections[counter][2] = sdnn

        counter = counter + 1
        if (counter % 1000) == 0:
            print(str(counter) + "\t Dateien wurden verarbeitet.")

    df = pd.DataFrame(detections, columns=['ham', 'two', 'swt'])

    df['ham'].fillna(df['ham'].mean(), inplace=True)
    df['two'].fillna(df['two'].mean(), inplace=True)
    df['swt'].fillna(df['swt'].mean(), inplace=True)

    return df


normal = load_references_normal()
arr = load_references_arrhythmia()

# df = extractSDNN(normal[0],128)
# df1 = extractSDNN(arr[0],360)
# df = pd.concat([df, df1], axis=0)
ecg_labels = np.append(normal[1], arr[1], axis=0)

# df = pd.read_csv('alternative/data/featuresCsv.csv')
#   End Data Preparation
#   -----------------------------------------------------------------------------------------
#   Start Machine Learning
start_time = time.time()
model = xgb.XGBClassifier()
model.load_model("model.txt")

encoder = LabelEncoder()
encoder.classes_ = load_alternative_encoder()

y_pred = model.predict(df)
y_pred = encoder.inverse_transform(y_pred)

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

print("Total right: ", counter['right'])
print("Total wrong: ", counter['wrong'])
print("Accuracy: ", counter["right"]/(counter["right"]+counter["wrong"]))
print("--- %s seconds ---" % (time.time() - start_time))
import numpy as np
import pandas as pd

from ecgdetectors import Detectors
from hrv import HRV


def normalizeLead(ecg_lead):
    # normalize each lead to 18000, by copying the lead until 1800 points are reached
    if (len(ecg_lead) != 18000):
        m = 18000 // len(ecg_lead)
        new_ecg_lead = ecg_lead
        for i in range(0, m-1):
            new_ecg_lead = np.append(new_ecg_lead, ecg_lead)
        r = 18000 - len(new_ecg_lead)
        new_ecg_lead = np.append(new_ecg_lead, ecg_lead[:r])
        return new_ecg_lead
    else:
        return ecg_lead


def extractFeatures(ecg_leads, fs, leadLengthNormalizer=True):
    detectors = Detectors(fs)
    # Signale verarbeitung
    hrv_class = HRV(fs)

    # creating a list of methods to extract each feature
    feautures = [
        lambda obj: hrv_class.SDNN(obj["r_peaks_swt"], True),
        lambda obj: hrv_class.RMSSD(obj["r_peaks_swt"]),
        lambda obj: hrv_class.SDSD(obj["r_peaks_swt"]),
        lambda obj: hrv_class.NN20(obj["r_peaks_swt"]),
        lambda obj: hrv_class.NN50(obj["r_peaks_swt"]),

    ]
    nfeatures = len(feautures)

    arr = np.zeros((len(ecg_leads), nfeatures))
    for idx, ecg_lead in enumerate(ecg_leads):
        # normalizing each lead if asked
        if (leadLengthNormalizer):
            ecg_lead = normalizeLead(ecg_lead)

        # Detektion der QRS-Komplexe mit Stationary Wavelet Transform Detector
        r_peaks_swt = detectors.swt_detector(ecg_lead)

        # calling each feature method and sending every parameter to extract desired feature
        for i in range(nfeatures):
            leadElements = {
                "ecg_lead": ecg_lead,
                "r_peaks_swt": r_peaks_swt,
            }
            arr[idx][i] = feautures[i](leadElements)

        if (idx % 1000) == 0:
            print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    # Save data with pandas
    df = pd.DataFrame(arr)
    df.drop(['index'], axis=1, inplace=True)
    df.rename(columns={'level_0': 'index'}, inplace=True)

    return df

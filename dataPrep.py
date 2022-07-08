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

def extractFeatures(ecg_leads, fs, leadLengthNormalizer = True):
    detectors = Detectors(fs)
    # Signale verarbeitung
    hrv_class = HRV(fs)

    feautures = [

    ]
    nfeatures = len(feautures)

    arr = np.zeros((len(ecg_leads), nfeatures))
    for idx, ecg_lead in enumerate(ecg_leads):
        # normalizing each lead if asked
        if (leadLengthNormalizer):
            ecg_lead = leadLengthNormalizer(ecg_lead)

        # Detektion der QRS-Komplexe mit Stationary Wavelet Transform Detector
        r_peaks_swt = detectors.swt_detector(ecg_lead)

    #   sdnn_swt = np.std(np.diff(r_peaks_swt)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Stationary Wavelet Transform
        sdnn_swt = hrv_class.SDNN(r_peaks_swt, True)
        rmssd_swt = hrv_class.RMSSD(r_peaks_swt)
        sdsd_swt = hrv_class.SDSD(r_peaks_swt)
        NN20_swt = hrv_class.NN20(r_peaks_swt)
        NN50_swt = hrv_class.NN50(r_peaks_swt)

        for i in range(nfeatures):
            arr[idx][i] = feautures(
                ecg_lead,
                r_peaks_swt
                )

        if (idx % 1000) == 0:
            print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    # Save data with pandas
    df = pd.DataFrame(arr, columns=['index', 'SDNN', 'RMSSD', 'SDSD',
                      'NN20', 'NN50', 'min moduled', 'max moduled', 'mean moduled'])
    df.drop(['index'], axis=1, inplace=True)
    df.rename(columns={'level_0': 'index'}, inplace=True)
    
    return df
import numpy as np
import pandas as pd

from ecgdetectors import Detectors
from hrv import HRV


def extractFeatures(ecg_leads, fs, leadLengthNormalizer = True):
    detectors = Detectors(fs)

    # Initialisierung der Feature-Arrays für Stationary Wavelet Transform Detector Detecto
    sdnn_swt = np.array([])
    hr_swt = np.array([])
    pNN20_swt = np.array([])
    pNN50_swt = np.array([])
    fAnalysis_swt = np.array([])

    # Signale verarbeitung
    hrv_class = HRV(fs)

    arr = np.zeros((len(ecg_leads), 9))

    for idx, ecg_lead in enumerate(ecg_leads):

        if (len(ecg_lead) != 18000 and leadLengthNormalizer):                                  # Length normalization
            m = 18000 // len(ecg_lead)
            new_ecg_lead = ecg_lead
            for i in range(0, m-1):
                new_ecg_lead = np.append(new_ecg_lead, ecg_lead)
            r = 18000 - len(new_ecg_lead)
            new_ecg_lead = np.append(new_ecg_lead, ecg_lead[:r])
            ecg_lead = new_ecg_lead

        min_ecg = np.amin(ecg_lead)
        max_ecg = np.amax(ecg_lead)
        mean_ecg = np.mean(ecg_lead)

        min_mod = min_ecg/mean_ecg
        max_mod = max_ecg/mean_ecg
        mean_mod = mean_ecg/(max_ecg-min_ecg)

        # Detektion der QRS-Komplexe mit Stationary Wavelet Transform Detector
        r_peaks_swt = detectors.swt_detector(ecg_lead)

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
        arr[idx][6] = min_mod
        arr[idx][7] = max_mod
        arr[idx][8] = mean_mod

        if (idx % 1000) == 0:
            print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    # Save data with pandas
    df = pd.DataFrame(arr, columns=['index', 'SDNN', 'RMSSD', 'SDSD',
                      'NN20', 'NN50', 'min moduled', 'max moduled', 'mean moduled'])
    df.drop(['index'], axis=1, inplace=True)
    df.rename(columns={'level_0': 'index'}, inplace=True)
    
    return df
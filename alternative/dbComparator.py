from ecgdetectors import Detectors
import numpy as np
import pandas as pd
from prepareEcgLeads import load_references as altDb

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wettbewerb import load_references as officialDb

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.

    returnData = data[s<m]
    print("Outliers removed: ", (len(data)- len(returnData)), "from: ",len(data))
    return returnData


ecg_leads, ecg_labels = altDb()


fs = 125                                                  # Sampling-Frequenz 300 Hz
detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
rPeaksValues_Alt = []
for ecg_lead in ecg_leads:
    r_peaks = detectors.hamilton_detector(ecg_lead)
    for i in r_peaks:
        rPeaksValues_Alt.append(ecg_lead[i])
    

print("Alt calculated, going to offical")
ecg_leads,ecg_labels,fs,ecg_names = officialDb('../../training')

fs = 300                                                  # Sampling-Frequenz 300 Hz
detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
rPeaksValues_Off = []
for ecg_lead in ecg_leads:
    r_peaks = detectors.hamilton_detector(ecg_lead)
    for i in r_peaks:
        rPeaksValues_Off.append(ecg_lead[i])

rPeaksValues_Alt = np.array(rPeaksValues_Alt)
rPeaksValues_Off = np.array(rPeaksValues_Off)

rPeaksValues_Alt = rPeaksValues_Alt[rPeaksValues_Alt>0]
rPeaksValues_Off = rPeaksValues_Off[rPeaksValues_Off>0]

rPeaksValues_Alt = reject_outliers(rPeaksValues_Alt)
rPeaksValues_Off = reject_outliers(rPeaksValues_Off)

print(rPeaksValues_Off.mean()/rPeaksValues_Alt.mean())
import numpy as np
import pandas as pd

from ecgdetectors import Detectors
from hrv import HRV
import neurokit2 as nk


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
        lambda obj: obj["nkTime"]["HRV_MeanNN"],
        lambda obj: obj["nkTime"]["HRV_SDNN"],
        lambda obj: obj["nkTime"]["HRV_RMSSD"],
        lambda obj: obj["nkTime"]["HRV_SDSD"],
        lambda obj: obj["nkTime"]["HRV_CVNN"],
        lambda obj: obj["nkTime"]["HRV_pNN50"],
        lambda obj: obj["nkTime"]["HRV_pNN20"],
        lambda obj: obj["nkTime"]["HRV_MedianNN"],
        lambda obj: obj["nkTime"]["HRV_CVSD"],
        lambda obj: obj["nkTime"]["HRV_HTI"],
        lambda obj: obj["nkTime"]["HRV_IQRNN"],
        lambda obj: obj["nkTime"]["HRVHRV_MCVNN_VLF"],
        lambda obj: obj["nkTime"]["HRV_MadNN"],
        lambda obj: obj["nkFreq"]["HRV_VLF"],
        lambda obj: obj["nkFreq"]["HRV_LF"],
        lambda obj: obj["nkFreq"]["HRV_HF"],
        lambda obj: obj["nkFreq"]["HRV_VHF"],
    ]
    nfeatures = len(feautures)

    arr = np.zeros((len(ecg_leads), nfeatures))
    for idx, ecg_lead in enumerate(ecg_leads):
        # normalizing each lead if asked
        if (leadLengthNormalizer):
            ecg_lead = normalizeLead(ecg_lead)

        # getting libraries classes
        try: 
            r_peaks_swt = detectors.swt_detector(ecg_lead)
            nkTime = nk.hrv_time(r_peaks_swt, sampling_rate=fs)
            nkFreq = nk.hrv_frequency(r_peaks_swt, sampling_rate=fs)
        except:
            print("Ecg lead {} was skipped.".format(idx))
            continue


        # calling each feature method and sending every parameter to extract desired feature
        leadElements = {
            "ecg_lead": ecg_lead,
            "r_peaks_swt": r_peaks_swt,
            "nkTime" : nkTime,
            "nkFreq" : nkFreq,
        }
        for i in range(nfeatures):
            # saving each feature on array of leads' features
            try:
                arr[idx][i] = feautures[i](leadElements)
            except:  # setting 0 when feature is not available
                arr[idx][i] = 0

        if (idx % 1000) == 0:
            print(str(idx) + "\t ecg signals processed")

    # Save data with pandas
    df = pd.DataFrame(arr)
    df.fillna(value=0,inplace=True)

    return df

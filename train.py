# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""


import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from wettbewerb import load_references

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

#ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz
ecg_leads,ecg_labels,fs,ecg_names = load_references('D:/BACKUP/DD-Poli/3st_2022_SoSe/Aulas/Wettbewerb_künstliche_Intelligenz_in_der_Medizin/training') # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz


detectors = Detectors(fs)                                     # Initialisierung des QRS-Detektors

                                                              # Initialisierung der Feature-Arrays für Hamilton Detector
sdnn_normal_ham = np.array([])                                # Initialisierung der Normal Feature-Arrays
sdnn_afib_ham = np.array([])                                  # Initialisierung der Vorhofflimmern Feature-Arrays
sdnn_ryth_ham = np.array([])                                  # Initialisierung der anderer Rhythmus Feature-Arrays
sdnn_unbr_ham = np.array([])                                  # Initialisierung der unbrauchbar Feature-Arrays

                                                              # Initialisierung der Feature-Arrays für Christov Detector
sdnn_normal_chr = np.array([])                                # Initialisierung der Normal Feature-Arrays
sdnn_afib_chr = np.array([])                                  # Initialisierung der Vorhofflimmern Feature-Arrays
sdnn_ryth_chr = np.array([])                                  # Initialisierung der anderer Rhythmus Feature-Arrays
sdnn_unbr_chr = np.array([])                                  # Initialisierung der unbrauchbar Feature-Arrays

                                                              # Initialisierung der Feature-Arrays für Stationary Wavelet Transform Detector Detector
sdnn_normal_swt = np.array([])                                # Initialisierung der Normal Feature-Arrays
sdnn_afib_swt = np.array([])                                  # Initialisierung der Vorhofflimmern Feature-Arrays
sdnn_ryth_swt = np.array([])                                  # Initialisierung der anderer Rhythmus Feature-Arrays
sdnn_unbr_swt = np.array([])                                  # Initialisierung der unbrauchbar Feature-Arrays

"""
                                                              # Initialisierung der Feature-Arrays für Engelse and Zeelenberg Detector
sdnn_normal_eng = np.array([])                                # Initialisierung der Normal Feature-Arrays
sdnn_afib_eng = np.array([])                                  # Initialisierung der Vorhofflimmern Feature-Arrays
sdnn_ryth_eng = np.array([])                                  # Initialisierung der anderer Rhythmus Feature-Arrays
sdnn_unbr_eng = np.array([])                                  # Initialisierung der unbrauchbar Feature-Arrays
"""

for idx, ecg_lead in enumerate(ecg_leads):
    r_peaks_ham = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe mit Hamilton Detector
    r_peaks_chr = detectors.christov_detector(ecg_lead)       # Detektion der QRS-Komplexe mit Christov Detector
    r_peaks_swt = detectors.swt_detector(ecg_lead)          # Detektion der QRS-Komplexe mit Stationary Wavelet Transform Detector
   # r_peaks_eng = detectors.engzee_detector(ecg_lead)       # Detektion der QRS-Komplexe mit Engelse and Zeelenberg Detector

    sdnn_ham = np.std(np.diff(r_peaks_ham)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Hamilton
    sdnn_chr = np.std(np.diff(r_peaks_chr)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Christov
    sdnn_swt = np.std(np.diff(r_peaks_swt)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Stationary Wavelet Transform
    #sdnn_eng = np.std(np.diff(r_peaks_eng)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Engelse and Zeelenberg

    if ecg_labels[idx]=='N':
      sdnn_normal_ham = np.append(sdnn_normal_ham,sdnn_ham)         # Zuordnung zu "Normal"
      sdnn_normal_chr = np.append(sdnn_normal_chr,sdnn_chr)
      sdnn_normal_swt = np.append(sdnn_normal_swt,sdnn_swt)
      #sdnn_normal_eng = np.append(sdnn_normal_eng,sdnn_eng)
    if ecg_labels[idx]=='A':
      sdnn_afib_ham = np.append(sdnn_afib_ham,sdnn_ham)             # Zuordnung zu "Vorhofflimmern"
      sdnn_afib_chr = np.append(sdnn_afib_chr,sdnn_chr) 
      sdnn_afib_swt = np.append(sdnn_afib_swt,sdnn_swt) 
      #sdnn_afib_eng = np.append(sdnn_afib_eng,sdnn_eng) 
    if ecg_labels[idx]=='O':
      sdnn_ryth_ham = np.append(sdnn_ryth_ham,sdnn_ham)             # Zuordnung zu "anderer Rhythmus"
      sdnn_ryth_chr = np.append(sdnn_ryth_chr,sdnn_chr) 
      sdnn_ryth_swt = np.append(sdnn_ryth_swt,sdnn_swt) 
      #sdnn_ryth_eng = np.append(sdnn_ryth_eng,sdnn_eng) 
    if ecg_labels[idx]=='~':
      sdnn_unbr_ham = np.append(sdnn_unbr_ham,sdnn_ham)             # Zuordnung zu "unbrauchbar"
      sdnn_unbr_chr = np.append(sdnn_unbr_chr,sdnn_chr) 
      sdnn_unbr_swt = np.append(sdnn_unbr_swt,sdnn_swt) 
      #sdnn_ryth_eng = np.append(sdnn_ryth_eng,sdnn_eng)
    if (idx % 100)==0:
      print(str(idx) + "\t EKG Signale wurden verarbeitet.")

fig, axs = plt.subplots(4,3, constrained_layout=True)
axs[0][0].hist(sdnn_normal_ham,2000)
axs[0][0].set_xlim([0, 300])
axs[0][0].set_title("Normal - Hamilton")
axs[0][0].set_xlabel("SDNN (ms)")
axs[0][0].set_ylabel("Anzahl")
axs[1][0].hist(sdnn_afib_ham,300)
axs[1][0].set_xlim([0, 300])
axs[1][0].set_title("Vorhofflimmern - Hamilton")
axs[1][0].set_xlabel("SDNN (ms)")
axs[1][0].set_ylabel("Anzahl")
axs[2][0].hist(sdnn_ryth_ham,300)
axs[2][0].set_xlim([0, 300])
axs[2][0].set_title("Anderer Rhtymus - Hamilton")
axs[2][0].set_xlabel("SDNN (ms)")
axs[2][0].set_ylabel("Anzahl")
axs[3][0].hist(sdnn_unbr_ham,300)
axs[3][0].set_xlim([0, 300])
axs[3][0].set_title("unbrauchbar - Hamilton")
axs[3][0].set_xlabel("SDNN (ms)")
axs[3][0].set_ylabel("Anzahl")

axs[0][1].hist(sdnn_normal_chr,2000)
axs[0][1].set_xlim([0, 300])
axs[0][1].set_title("Normal - Christov")
axs[0][1].set_xlabel("SDNN (ms)")
axs[0][1].set_ylabel("Anzahl")
axs[1][1].hist(sdnn_afib_chr,300)
axs[1][1].set_xlim([0, 300])
axs[1][1].set_title("Vorhofflimmern - Christov")
axs[1][1].set_xlabel("SDNN (ms)")
axs[1][1].set_ylabel("Anzahl")
axs[2][1].hist(sdnn_ryth_chr,300)
axs[2][1].set_xlim([0, 300])
axs[2][1].set_title("Anderer Rhtymus - Christov")
axs[2][1].set_xlabel("SDNN (ms)")
axs[2][1].set_ylabel("Anzahl")
axs[3][1].hist(sdnn_unbr_chr,300)
axs[3][1].set_xlim([0, 300])
axs[3][1].set_title("unbrauchbar - Christov")
axs[3][1].set_xlabel("SDNN (ms)")
axs[3][1].set_ylabel("Anzahl")

axs[0][2].hist(sdnn_normal_swt,2000)
axs[0][2].set_xlim([0, 300])
axs[0][2].set_title("Normal - Stationary Wavelet Transform")
axs[0][2].set_xlabel("SDNN (ms)")
axs[0][2].set_ylabel("Anzahl")
axs[1][2].hist(sdnn_afib_swt,300)
axs[1][2].set_xlim([0, 300])
axs[1][2].set_title("Vorhofflimmern - Stationary Wavelet Transform")
axs[1][2].set_xlabel("SDNN (ms)")
axs[1][2].set_ylabel("Anzahl")
axs[2][2].hist(sdnn_ryth_swt,300)
axs[2][2].set_xlim([0, 300])
axs[2][2].set_title("Anderer Rhtymus - Stationary Wavelet Transform")
axs[2][2].set_xlabel("SDNN (ms)")
axs[2][2].set_ylabel("Anzahl")
axs[3][2].hist(sdnn_unbr_swt,300)
axs[3][2].set_xlim([0, 300])
axs[3][2].set_title("unbrauchbar - Stationary Wavelet Transform")
axs[3][2].set_xlabel("SDNN (ms)")
axs[3][2].set_ylabel("Anzahl")

"""
axs[0][3].hist(sdnn_normal_eng,2000)
axs[0][3].set_xlim([0, 300])
axs[0][3].set_title("Normal - Engelse and Zeelenberg")
axs[0][3].set_xlabel("SDNN (ms)")
axs[0][3].set_ylabel("Anzahl")
axs[1][3].hist(sdnn_afib_eng,300)
axs[1][3].set_xlim([0, 300])
axs[1][3].set_title("Vorhofflimmern - Engelse and Zeelenberg")
axs[1][3].set_xlabel("SDNN (ms)")
axs[1][3].set_ylabel("Anzahl")
axs[2][3].hist(sdnn_ryth_eng,300)
axs[2][3].set_xlim([0, 300])
axs[2][3].set_title("Anderer Rhtymus - Engelse and Zeelenberg")
axs[2][3].set_xlabel("SDNN (ms)")
axs[2][3].set_ylabel("Anzahl")
axs[3][3].hist(sdnn_unbr_eng,300)
axs[3][3].set_xlim([0, 300])
axs[3][3].set_title("unbrauchbar - Engelse and Zeelenberg")
axs[3][3].set_xlabel("SDNN (ms)")
axs[3][3].set_ylabel("Anzahl")
"""

plt.show()

"""

sdnn_total = np.append(sdnn_normal,sdnn_afib) # Kombination der beiden SDNN-Listen
p05 = np.nanpercentile(sdnn_total,5)          # untere Schwelle
p95 = np.nanpercentile(sdnn_total,95)         # obere Schwelle
thresholds = np.linspace(p05, p95, num=20)    # Liste aller möglichen Schwellwerte
F1 = np.array([])
for th in thresholds:
  TP = np.sum(sdnn_afib>=th)                  # Richtig Positiv
  TN = np.sum(sdnn_normal<th)                 # Richtig Negativ
  FP = np.sum(sdnn_normal>=th)                # Falsch Positiv
  FN = np.sum(sdnn_afib<th)                   # Falsch Negativ
  F1 = np.append(F1, TP / (TP + 1/2*(FP+FN))) # Berechnung des F1-Scores

th_opt=thresholds[np.argmax(F1)]              # Bestimmung des Schwellwertes mit dem höchsten F1-Score

if os.path.exists("model.npy"):
    os.remove("model.npy")
with open('model.npy', 'wb') as f:
    np.save(f, th_opt)

fig, ax = plt.subplots()
ax.plot(thresholds,F1)
ax.plot(th_opt,F1[np.argmax(F1)],'xr')
ax.set_title("Schwellwert")
ax.set_xlabel("SDNN (ms)")
ax.set_ylabel("F1")
plt.show()

fig, axs = plt.subplots(2,1, constrained_layout=True)
axs[0].hist(sdnn_normal,2000)
axs[0].set_xlim([0, 300])
tmp = axs[0].get_ylim()
axs[0].plot([th_opt,th_opt],[0,10000])
axs[0].set_ylim(tmp)
axs[0].set_title("Normal")
axs[0].set_xlabel("SDNN (ms)")
axs[0].set_ylabel("Anzahl")
axs[1].hist(sdnn_afib,300)
axs[1].set_xlim([0, 300])
tmp = axs[1].get_ylim()
axs[1].plot([th_opt,th_opt],[0,10000])
axs[1].set_ylim(tmp)
axs[1].set_title("Vorhofflimmern")
axs[1].set_xlabel("SDNN (ms)")
axs[1].set_ylabel("Anzahl")
plt.show()

"""
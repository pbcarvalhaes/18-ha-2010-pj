from sklearn import metrics
from wettbewerb import load_references
from ecgdetectors import Detectors
import numpy as np
from hrv import HRV
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle

# Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name
ecg_leads,ecg_labels,fs,ecg_names = load_references('./training')                 # Wenn der Ordner mit der Database Training von moodle in einem anderen Ordner liegt, fügen Sie hier den korrigierten Pfad ein 
detectors = Detectors(fs)                                                         # Initialisierung des QRS-Detektors  

# Initialisierung der Feature-Arrays für Stationary Wavelet Transform Detector 
sdnn_swt = np.array([])                                       
hr_swt = np.array([])                                  
pNN20_swt = np.array([])                                  
pNN50_swt = np.array([])      
fAnalysis_swt = np.array([]) 
r_peak_values =  np.array([])

# Signale verarbeitung
hrv_class = HRV(fs)

arr = np.zeros((6000,9))
diagnosis = np.zeros(6000, dtype = 'str')

for idx, ecg_lead in enumerate(ecg_leads):

    if len(ecg_lead) != 18000:                                  # Length normalization
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

    r_peaks_swt = detectors.swt_detector(ecg_lead)              # Detektion der QRS-Komplexe mit Stationary Wavelet Transform Detector 

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
    
    diagnosis[idx] = ecg_labels[idx]
    if (idx % 1000) == 0:
        print(str(idx) + "\t EKG Signale wurden verarbeitet.")

# Save data with pandas
df = pd.DataFrame(arr, columns = ['index', 'SDNN', 'RMSSD', 'SDSD', 'NN20', 'NN50', 'min moduled', 'max moduled', 'mean moduled'])
df.drop(['index'], axis = 1, inplace = True)
df.rename(columns = {'level_0':'index'}, inplace = True)
df['diagnosis'] = diagnosis
df.dropna(inplace = True)

# Setting input/output
X=df[df.columns[:-1]]
print(X)
y=df['diagnosis']

# Applying oversampling
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# MLP model
clf = MLPClassifier(hidden_layer_sizes = (140, 150, 140), max_iter = 300, activation = 'tanh', alpha = 0.0005, solver = 'adam')
clf.fit(X, y)
predict_train = clf.predict(X_train)
predict_test = clf.predict(X_test)
pickle.dump(clf, open('NN.pkl', 'wb'))

print("Accuracy:",metrics.accuracy_score(y_test, predict_test))
print("F1:", metrics.f1_score(y_test, predict_test, average = 'weighted'))
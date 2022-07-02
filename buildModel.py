from sklearn import metrics
from wettbewerb import load_references
from ecgdetectors import Detectors
import numpy as np
from hrv import HRV
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pickle

# Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name
ecg_leads,ecg_labels,fs,ecg_names = load_references('./training')                 # Wenn der Ordner mit der Database Training von moodle in einem anderen Ordner liegt, fügen Sie hier den korrigierten Pfad ein 
detectors = Detectors(fs)                                                         # Initialisierung des QRS-Detektors  

# Initialisierung der Feature-Arrays für Stationary Wavelet Transform Detector Detecto 
sdnn_swt = np.array([])                                       
hr_swt = np.array([])                                  
pNN20_swt = np.array([])                                  
pNN50_swt = np.array([])      
fAnalysis_swt = np.array([])   

# Signale verarbeitung
hrv_class = HRV(fs)

arr = np.zeros((6000,8))
diagnosis = np.zeros(6000,dtype='str')

for idx, ecg_lead in enumerate(ecg_leads):

    if len(ecg_lead) != 18000:                                  # Length normalization
        m = 18000 // len(ecg_lead)
        new_ecg_lead = ecg_lead
        for i in range(0, m-1):
            new_ecg_lead = np.append(new_ecg_lead, ecg_lead)
        r = 18000 - len(new_ecg_lead)
        new_ecg_lead = np.append(new_ecg_lead, ecg_lead[:r])
        ecg_leads[idx] = new_ecg_lead
   
    r_peaks_swt = detectors.swt_detector(ecg_lead)              # Detektion der QRS-Komplexe mit Stationary Wavelet Transform Detector

#   sdnn_swt = np.std(np.diff(r_peaks_swt)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden - Stationary Wavelet Transform
    sdnn_swt = hrv_class.SDNN(r_peaks_swt, True)
    rmssd_swt = hrv_class.RMSSD(r_peaks_swt)
    sdsd_swt = hrv_class.SDSD(r_peaks_swt)
    NN20_swt = hrv_class.NN20(r_peaks_swt)
    NN50_swt = hrv_class.NN50(r_peaks_swt)
    HR_swt_mean = np.mean(hrv_class.HR(r_peaks_swt))
    succ_diffs_mean = np.mean(hrv_class._succ_diffs(r_peaks_swt))

#   fAnalysis_swt
    arr[idx][0] = idx
    arr[idx][1] = sdnn_swt
    arr[idx][2] = rmssd_swt
    arr[idx][3] = sdsd_swt
    arr[idx][4] = NN20_swt
    arr[idx][5] = NN50_swt
    arr[idx][6] = HR_swt_mean
    arr[idx][7] = succ_diffs_mean
    
    diagnosis[idx] = ecg_labels[idx]
    if (idx % 1000) == 0:
        print(str(idx) + "\t EKG Signale wurden verarbeitet.")

# Save data with pandas
df = pd.DataFrame(arr, columns = ['index', 'SDNN', 'RMSSD', 'SDSD', 'NN20', 'NN50', 'HR mean', 'SD mean'])
df.drop(['index'],axis=1, inplace = True)
df.rename(columns = {'level_0':'index'}, inplace = True)
df['diagnosis'] = diagnosis
df.dropna(inplace=True)

# Setting input/output
X=df[df.columns[:-1]]
y=df['diagnosis']

# Applying oversampling
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# RandomForest model
clf = RandomForestClassifier(n_estimators=29,max_depth=30,bootstrap=True,min_samples_leaf=1, class_weight="balanced")
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("RF Hyperparameter selected ")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred, average='weighted'))
pickle.dump(clf, open('RF_model.pkl', 'wb'))

# RandomForest model
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("RF")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred, average='weighted'))
pickle.dump(clf, open('RF_model2.pkl', 'wb'))

# XGB model
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
np.save('encoder.npy', encoder.classes_)
#model = xgb.XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=5)
model = xgb.XGBClassifier(max_depth=8, learning_rate=0.3, n_estimators=15)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = encoder.inverse_transform(y_pred)
print("XGB hyperparamter selected")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred, average='weighted'))
model.save_model('XGB_model.txt')

# XGB model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = encoder.inverse_transform(y_pred)
print("XGB")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred, average='weighted'))
model.save_model('XGB_model2.txt')
from sklearn import metrics
from wettbewerb import load_references
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import pickle

from dataPrep import extractFeatures

# Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name
ecg_leads,ecg_labels,fs,ecg_names = load_references('./training')                 # Wenn der Ordner mit der Database Training von moodle in einem anderen Ordner liegt, fügen Sie hier den korrigierten Pfad ein 

df = extractFeatures(ecg_leads,fs)

# Setting input/output
df['diagnosis'] = ecg_labels
df.dropna(inplace=True)

X=df[df.columns[:-1]]
y=df['diagnosis']

# Applying oversampling
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# RandomForest model 1
clf = RandomForestClassifier(n_estimators=28,max_depth=25,bootstrap=True,min_samples_leaf=1, class_weight="balanced")
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("RF Hyperparameter selected ")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred, average='weighted'))
pickle.dump(clf, open('RF_model.pkl', 'wb'))

# RandomForest model 2
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("RF")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred, average='weighted'))
pickle.dump(clf, open('RF_model2.pkl', 'wb'))

# XGB model 1
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

# XGB model 2
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = encoder.inverse_transform(y_pred)
print("XGB")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred, average='weighted'))
model.save_model('XGB_model2.txt')


# MLP model
clf = MLPClassifier(hidden_layer_sizes = (140, 150, 140), max_iter = 300, activation = 'tanh', alpha = 0.0005, solver = 'adam')
clf.fit(X, y)
predict_train = clf.predict(X_train)
predict_test = clf.predict(X_test)
pickle.dump(clf, open('NN.pkl', 'wb'))

print("Accuracy:",metrics.accuracy_score(y_test, predict_test))
print("F1:", metrics.f1_score(y_test, predict_test, average = 'weighted'))
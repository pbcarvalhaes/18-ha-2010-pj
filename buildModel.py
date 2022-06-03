from statistics import mode
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/dropNaN.csv')

x=df[df.columns[:-1]]
y=df['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

params = {
    # Parameters that we are going to tune.
    'max_depth':8,
    'eta':.1,
}
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
np.save('encoder.npy', encoder.classes_)


model = xgb.XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
model.save_model("model.txt")
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import collections
import warnings

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template

app = Flask(__name__)

app_train = pd.read_csv("../../csv_file/application_train.csv", index_col=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    app_train_filt = app_train[['TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']]
    individu_test = pd.read_csv('individu.csv')

    le = LabelEncoder()
    le_count = 0

    warnings.filterwarnings('ignore')

    try:
        for col in app_train_filt:
            if app_train_filt[col].dtypes == 'object':
                if len(list(app_train_filt[col].unique())) <= 2:
                    le.fit(app_train_filt[col])
                    app_train_filt[col] = le.transform(app_train_filt[col])
                    individu_test[col] = le.transform(individu_test[col])
                    
                    le_count += 1
        print('test')
    except :
        print('error')


    app_train_filt = pd.get_dummies(app_train_filt)
    individu_test = pd.get_dummies(individu_test)

    train_labels = app_train_filt['TARGET']
    app_train_filt, individu_test = app_train_filt.align(individu_test, join = 'inner', axis = 1)
    app_train_filt['TARGET'] = train_labels

    X = app_train_filt.drop(labels="TARGET", axis=1)
    y = app_train_filt['TARGET']

    X_train_sf, X_test_sf, y_train_sf, y_test_sf = train_test_split(X, y, test_size=0.25, random_state=0)

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train_sf, y_train_sf)

    y_pred_test = classifier.predict(individu_test)

    collections.Counter(y_pred_test)

    for num in range(0, len(y_pred_test)):
        if y_pred_test[num] == 0:
            result = ' est susceptible de rembourser'
        else:
            result = ' présente trop de risque'


    return render_template('index.html', prediction_text='Le client'+result)

if __name__ == "__main__":
    app.run(debug=True)
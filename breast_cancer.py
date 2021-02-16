# -*- coding: utf-8 -*-
"""breast_cancer

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1h3A_ClK5d5Nj-3xyBHpdjLgdLQZKRyS4
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
 

def load_data(dataset_url):
  data = pd.read_csv(dataset_url, header=None)
  dataset = data.values
  X = dataset[:,:-1]
  y = dataset[:,-1]
  X = X.astype(str)
  X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33,random_state=1)
  return X_train, X_test, y_train, y_test

def prep_data(X_train,X_test):
  oe = OrdinalEncoder()
  oe.fit(X_train)
  X_train_encoded = oe.transform(X_train)
  X_test_encoded = oe.transform(X_test)
  return X_train_encoded, X_test_encoded

def prep_labels(y_train, y_test):
  le = LabelEncoder()
  le.fit(y_train)
  y_train_encoded =  le.transform(y_train)
  y_test_encoded = le.transform(y_test)
  return y_train_encoded, y_test_encoded

def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs  


X_train,X_test,y_train,y_test = load_data("/content/drive/MyDrive/A-Kaggle-Datasets/breast-cancer.csv")
X_train_encoded, X_test_encoded = prep_data(X_train,X_test)
y_train_encoded, y_test_encoded = prep_labels(y_train,y_test)
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
print(X_train[:,:10])

for i in range(len(fs.scores_)):
	print(f'Feature {i}: {fs.scores_[i]}')
 
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()





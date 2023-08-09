import pandas as pd
import os
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from tabulate import tabulate
import joblib
import pickle
class_label = ['Non-Default(0)','Default(1)'] # env var

# Load the data
data = pd.read_csv('D:\Barath Suresh Docs\PROGRAMMING\MACHINE LEARNING\credit_card_fraud_detection\creditcard.csv')

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for use with the CNN
X_train = X_train.reshape((-1, X_train.shape[1], 1))
X_test = X_test.reshape((-1, X_test.shape[1], 1))

model_20 = keras.Sequential([
    keras.layers.Conv1D(filters=32,kernel_size=2,activation="relu",input_shape=(X_train.shape[1],1)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv1D(filters=64,kernel_size=2,activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Conv1D(filters=64,kernel_size=2,activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Conv1D(filters=64,kernel_size=2,activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    
    keras.layers.Flatten(),
    keras.layers.Dense(64,activation="relu"),
    keras.layers.Dropout(0.5), 

    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(50,activation="relu"),
    keras.layers.Dense(25,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid"),
])
model_20.summary()

with os.fdopen(os.open("D:\Barath Suresh Docs\PROGRAMMING\MACHINE LEARNING\credit_card_fraud_detection\Saved Models\CNNHistory_20.joblib", os.O_WRONLY | os.O_CREAT)) as file:
    if os.path.getsize("D:\Barath Suresh Docs\PROGRAMMING\MACHINE LEARNING\credit_card_fraud_detection\Saved Models\CNNHistory_20.joblib") == 0:
        model_20.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history_20 = model_20.fit(X_train, Y_train, epochs=100, validation_split=0.2, batch_size=64)
        joblib.dump(history_20,filename="D:\Barath Suresh Docs\PROGRAMMING\MACHINE LEARNING\credit_card_fraud_detection\Saved Models\CNNHistory_20.joblib")
    else:
        history_20 = joblib.load("D:\Barath Suresh Docs\PROGRAMMING\MACHINE LEARNING\credit_card_fraud_detection\Saved Models\CNNHistory_20.joblib")

# Extract the CNN features
cnn_train_features = model_20.predict(X_train)
cnn_test_features = model_20.predict(X_test)

# Train an SVM on the CNN features
svm = SVC(kernel='poly')
svm.fit(cnn_train_features, Y_train)

# Evaluate the SVM model
svm_train_acc = accuracy_score(Y_train, svm.predict(cnn_train_features))
svm_test_acc = accuracy_score(Y_test, svm.predict(cnn_test_features))
print('SVM train accuracy:', svm_train_acc)
print('SVM test accuracy:', svm_test_acc)


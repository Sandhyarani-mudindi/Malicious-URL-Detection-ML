import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance
import lightgbm as lgb

import time

df1 = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", nrows=50000)
df2 = pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", nrows=50000)
df3 = pd.read_csv("Friday-WorkingHours-Morning.pcap_ISCX.csv", nrows=50000)
df4 = pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv", nrows=50000)
df5 = pd.read_csv("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", nrows=50000)
df6 = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", nrows=50000)
df7 = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv")
#df8 = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")

df = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=0)
df.columns = df.columns.str.strip()

print(df.columns)
df.columns = df.columns.str.strip()
df['Label'].value_counts()

#Data Pre_Processing

# Encode the dataset
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])

# address empty values
if df.isnull().values.any() or np.isinf(df).values.any(): # if there is any empty or infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace = True)

# Split the dataset into training and testing
X = df.drop(['Label'],axis=1).values
y = df.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
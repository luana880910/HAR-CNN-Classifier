import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model
# from keras import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#gnuplot 專題通常用這個
from sklearn.preprocessing import StandardScaler, LabelEncoder
import csv
import scipy.stats as stats
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from decimal import Decimal
model = load_model('my_model_avggood.h5')
processedList = []
tempi = 0
with open("testData.csv", "r") as f:
    for line in f:
        sepline = line.split(",")
        sepline[5] = sepline[5].replace("\n","")
        if tempi == 0:
            tempi = 1
            continue
        temp = [sepline[0], sepline[1], sepline[2], sepline[3],sepline[4],sepline[5]]
        processedList.append(temp)
columns = ['time','x', 'y', 'z','total','label']
data = pd.DataFrame(data = processedList, columns = columns)
data['x'] = data['x'].astype('float')
data['y'] = data['y'].astype('float')
data['z'] = data['z'].astype('float')

df = data.drop(['total','time'], axis = 1).copy()
walk = df[df['label']=='walk'].copy()
scooter = df[df['label']=='scooter'].copy()
MRTt = df[df['label']=='MRT'].copy()
balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([walk, scooter, MRTt])
label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['label'])
X = balanced_data[['x', 'y', 'z']]
y = balanced_data['label']
scaler = StandardScaler()
X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
scaled_X['label'] = y.values


def get_frames(df, frame_size, hop_size):
    
    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels
Fs = 100
frame_size = Fs*2
hop_size = Fs*1
scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
scaled_X['label'] = y.values
X, y = get_frames(scaled_X, frame_size, hop_size)
print(X.shape)
X = X.reshape(528, 200, 3, 1)
y_pred = model.predict_classes(X)
mat = confusion_matrix(y, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7))
test_accuracy = 0
for i in range(len(y_pred)):
    if y[i] == y_pred[i]:
        test_accuracy += 1
test_accuracy /= len(y_pred)
test_accuracy *= 100

model.summary()
print("mission success rate = "+str(Decimal(test_accuracy).quantize(Decimal("0.00")))+"%")
#多了成功率

plt.show()
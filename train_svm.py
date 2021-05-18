import pickle
import subprocess
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm

subprocess.run(["rm","predict.txt"],stdout=subprocess.PIPE)

train = pd.read_csv(r"data.txt",sep=',')
test = pd.read_csv(r"data.txt",sep=',')

predictors = ['r', 't', 'B', 'G', 'R']
X = train[predictors].values
X_predict = test[predictors].values
y = train['m'].values
train.head(10)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = StandardScaler()# Ejercicio, no use la escalización de los datos a ver que tal funciona!
scaler.fit(X_train)# el fit de los datos solo se hace con el conjunto de entrenamiento!
X_train = scaler.transform(X_train)
X_predict = scaler.transform(X_predict)
X_test = scaler.transform(X_test)

kernels=['linear', 'poly', 'rbf', 'sigmoid']
#lineal
Kernel=0
msv = svm.SVC(kernel=kernels[Kernel])

#polinomial cuadrático
#Kernel=1
#msv = svm.SVC(kernel=kernels[Kernel],degree=2)

#polinomial cúbico
#Kernel=1
#msv = svm.SVC(kernel=kernels[Kernel],degree=3)
#rbf 
#Kernel=2
#msv = svm.SVC(kernel=kernels[Kernel])
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

msv.fit(X_train, y_train)

with open('model.pkl', 'wb') as file:
    pickle.dump(msv, file)

pred = msv.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
#Score is Mean Accuracy
scikit_score = msv.score(X_test, y_test)
print('Scikit score: ', scikit_score)
MCC = matthews_corrcoef(y_test, pred)
print("MCC = ", MCC)
ACC = accuracy_score(y_test, pred)
print("ACC = ", ACC)

fpr,tpr,thresholds = roc_curve(y_test, pred)
roc_auc=roc_auc_score(y_test, pred)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic. ROC')
plt.legend(loc="lower right")
plt.show()

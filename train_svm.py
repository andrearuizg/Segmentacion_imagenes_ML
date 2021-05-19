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
import multiprocessing


if __name__ == '__main__':
	train = pd.read_csv(r"data.txt",sep=',')

	predictors = ['r', 't', 'B', 'G', 'R']
	X = train[predictors].values
	y = train['m'].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, shuffle=False)

	scaler = StandardScaler()# Ejercicio, no use la escalización de los datos a ver que tal funciona!
	scaler.fit(X_train)# el fit de los datos solo se hace con el conjunto de entrenamiento!
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	kernels = ['linear', 'poly', 'poly', 'rbf', 'sigmoid']
	degree = [3, 2, 3, 3, 3] # 3 es el valor por defecto
	#lineal
	for i in range(5):
		msv = svm.SVC(kernel=kernels[i], degree=degree[i], verbose=True , max_iter=10000)

		print("Entrena el kernel ",kernels[i]," de grado ",degree[i])

		with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
			msv.fit(X_train, y_train)
			
		file_model = 'models_svm/model_%s_%s.pkl' %(kernels[i],degree[i])
		with open(file_model, 'wb') as file:
			pickle.dump(msv, file)
			
		print("Entrenó el kernel ",kernels[i]," de grado ",degree[i])
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

		plt.figure(i)
		lw = 2
		plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic. ROC')
		plt.legend(loc="lower right")
		out = "./ROC/ROC_%s_%s.png" %(kernels[i], degree[i])
		plt.savefig(out)

import os
import subprocess
import pickle
from glob import glob
from tqdm import tqdm
import pandas as pd
import math
import cv2
import numpy as np
import csv
from time import time
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler

def open_csv(data,kernels,degree):
	file_name = 'results/%s_%s/times.csv' %(kernels,degree)
	with open(file_name, 'a') as f:
		writer = csv.writer(f)
		writer.writerow(data)

graficar = 1 # 1 para graficar tetha y radio, 0 para no hacerlo
images = glob(os.path.join("media/color/*"))

train = pd.read_csv(r"data.txt",sep=',')

predictors = ['r', 't', 'B', 'G', 'R']
X = train[predictors].values
scaler = StandardScaler()# Ejercicio, no use la escalización de los datos a ver que tal funciona!
scaler.fit(X)# el fit de los datos solo se hace con el conjunto de entrenamiento!

for i in range(5):

	kernels = ['linear', 'poly', 'poly', 'rbf', 'sigmoid']
	degree = [3, 2, 3, 3, 3] # 3 es el valor por defecto
	file_model = 'models_svm/model_%s_%s.pkl' %(kernels[i],degree[i])
	with open(file_model, 'rb') as file:
		model = pickle.load(file)

	pr = np.zeros((256, 256), np.uint8)
	gg = np.zeros((256, 256), np.uint8)
	gg1 = np.zeros((256, 256), np.uint8)
	gg2 = np.zeros((256, 256), np.uint8)

	for path in range(len(images)):
		print(path+1,'/',len(images))
		image = cv2.imread(images[path], cv2.IMREAD_UNCHANGED)
		image = cv2.resize(image, (256, 256))
		
		for y in tqdm(range(256)):
			for x in range(256):
				r = math.sqrt((((x-256)/1.42)**2)+(((y - 128)/1.42)**2))
				if not r:
					t = 0
				else:
					t = (math.acos(((256-x)/1.42)/(r)) * (180/math.pi)) * 2.8333
					
				gg[x, y] = int(t)
				gg1[x, y] = int(r)
				gg2[x, y] = t
				B, G, R = image[x,y,:]
				fields = np.array([[r,t,B,G,R]])
				fields = scaler.transform(fields)
				t1 = time()
				pr[x, y] = model.predict(fields)
				open_csv([time()-t1],kernels[i],degree[i])

		result2 = pr		
		pr = pr * 255
		result1 = cv2.cvtColor(pr.astype('uint8'), cv2.COLOR_GRAY2BGR)
		result1[..., 0] = np.where(result1[..., 0], 90, 0).astype('uint8')
		result1[..., 1] = 0
		result1 = cv2.addWeighted(image, 0.7, result1, 0.3, 0)
		
		str_file = (images[path].split('/')[-1]).split('.')[0]
		filename = "results/%s_%s/color/%s-predicted.png" % (kernels[i],degree[i],str_file)
		filename1 = "results/%s_%s/mask/%s-mask-predicted.png" % (kernels[i],degree[i],str_file)
		cv2.imwrite(filename, result1)
		cv2.imwrite(filename1, result2)
		
	if graficar:
		cv2.imwrite('tetha.png', gg)
		cv2.imwrite('radius.png', gg1)
		
print('Terminó °_°')

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

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
images = glob(os.path.join("color/*"))

'''masks = pd.read_csv(r"data.txt",sep=',')
predict = pd.read_csv(r"predict.txt",sep=',')


images = glob(os.path.join("color/*"))

fields=['x','y']
mask = masks[fields].values

fields=['m']
predicts = predict[fields].values'''

pr = np.zeros((256, 256), np.uint8)

for path in range(len(images)):
	print(path,'/',len(images))
	image = cv2.imread(images[path], cv2.IMREAD_UNCHANGED)
	image = cv2.resize(image, (256, 256))
	
	for x in tqdm(range(256)):
		for y in range(256):
			r = math.sqrt(((x-128)**2)+((256-y)**2))
			if not r:
				t = 0
			else:
				t = math.acos((x-128)/r)
			B, G, R = image[x,y,:]
			fields = np.array([[r,t,B,G,R]])
			pr[x, y] = model.predict(fields)
	
	result2 = pr		
	pr = pr * 255
	result1 = cv2.cvtColor(pr.astype('uint8'), cv2.COLOR_GRAY2BGR)
	result1[..., 0] = np.where(result1[..., 0], 90, 0).astype('uint8')
	result1[..., 1] = 0
	result1 = cv2.addWeighted(image, 0.7, result1, 0.3, 0)
	str_file = (images[path].split('/')[-1]).split('.')[0]
	filename = "results/color/%s-predicted.png" % (str_file)
	filename1 = "results/mask/%s-mask-predicted.png" % (str_file)
	cv2.imwrite(filename, result1)
	cv2.imwrite(filename1, result2)

	
print('Terminó °_°')

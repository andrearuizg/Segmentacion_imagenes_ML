import os
import subprocess
from glob import glob
from tqdm import tqdm
import math
import cv2
import numpy as np
import csv

subprocess.run(["rm","data.txt"],stdout=subprocess.PIPE)

def open_csv(data):
	with open(r'data.txt', 'a') as f:
		writer = csv.writer(f)
		writer.writerow(data)

images = glob(os.path.join("./media/color/*"))
masks = glob(os.path.join("./media/mask/*"))

fields=['x','y','r','t','m','R','G','B']
open_csv(fields)

for path in range(len(images)):
	print(path+1,'/',len(images))
	image = cv2.imread(images[path], cv2.IMREAD_UNCHANGED)
	image = cv2.resize(image, (256, 256))
	mask = cv2.imread(masks[path], cv2.IMREAD_UNCHANGED)
	mask = cv2.resize(mask, (256, 256))
	
	for y in tqdm(range(256)):
		for x in range(256):
			r = math.sqrt((((x-256)/1.42)**2)+(((y - 128)/1.42)**2))
			if not r:
				t = 0
			else:
				t = (math.acos(((256-x)/1.42)/(r)) * (180/math.pi)) * 2.8333
			m = mask[x,y]
			B, G, R = image[x,y,:]
			fields = [x,y,r,t,m,B,G,R]
			open_csv(fields)

print('Terminó °_°')

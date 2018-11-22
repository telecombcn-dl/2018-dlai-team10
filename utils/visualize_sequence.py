#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
datadir_simplified = 'C:\\Users\\user\\Ponç\\MET\\DLAI\\Project\\data\\simplified\\wheel.csv'

csv = pd.read_csv(datadir_simplified, sep = ',',engine = 'python')

# Header Format
#['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word'] 
# SELECT JUST THE ONES CLASSIFIED CORRECTLY
drawing = csv[csv['recognized']==True]
drawings = csv['drawing']
drawings = drawings.values
x = np.array(json.loads(drawings[101010]))
print(type(x))
print(np.shape(x))
print(x)
img = np.zeros((256, 256))

for elem in x:
	for i in range(0,len(elem[0])-1):
		plt.subplot(211)
		plt.plot([elem[0][i], elem[0][i+1]], [elem[1][i], elem[1][i+1]], marker = 'o')	
	img[elem[1][:], elem[0][:]] = 255

plt.subplot(212)
plt.imshow(img)
plt.show()

# En realitat l'ordre importa, no fa falta que li passis les linies perque la xarxa (en principi una LSTM) hauria de ser capaç 
# d'interpretar com els pixels tot i que no sap que son pixels
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

datadir_simplified = 'C:\\Users\\user\\Ponç\\MET\\DLAI\\Project\\data\\simplified\\ladder.csv'

out_dir = r'C:\Users\user\Ponç\MET\DLAI\Project\data\simplified_strokes_npy\ladder'

class_name = "ladder_"

csv = pd.read_csv(datadir_simplified, sep = ',',engine = 'python')

# Header Format
#['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word'] 
# SELECT ONLY THE ONES CLASSIFIED CORRECTLY
drawing = csv[csv['recognized']==True]
drawings = csv['drawing']
drawings = drawings.values
index = 100000 # This is the good sample to show in presentation
print(np.shape(drawings))

for i in range(0, np.shape(drawings)[0]):
	x = np.array(json.loads(drawings[i]))

	drawing_strokes = []

	for elem in x:
		mat = np.zeros((2, len(elem[0])))
		mat[0, :] = elem[0][:]
		mat[1, :] = elem[1][:]
		drawing_strokes.append(mat)

	aux = np.zeros((2,1)) 
	for stroke in drawing_strokes:
		aux = np.hstack((aux, stroke))
	sample_name = class_name + str(i)
	np.save(os.path.join(out_dir, sample_name), aux[:, 1:])
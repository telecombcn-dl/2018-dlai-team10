import numpy as np
import os

"""
This script takes the raw dataset from a given directory, divides and reshapes it into single images
and places those images in their also created by the script corresponding folders.

The folder structure is the following:
Inside 'dire', there are npy files for each class. This script creates three folders: train, validation and test. 
Inside each of these three folders, the script creates one folder for each class and places there the corresponding
reshaped and divided dateset files. 
"""

class_name = ['apple', 'banana', 'book', 'fork', 'key', 'ladder', 'pizza', 'stop_sign', 'tennis_racquet', 'wheel']
step = ['train', 'validation', 'test']

dire = r'C:\Users\Usuario\Documents\Telecos\Màster\DLAI\Project\Reduced Dataset'+'\\'

max_length = 100000 # Maximum number of files (drawings) per class
percen=[0.6, 0.3, 0.1] # Percentage of training, validation and testing

begin = [0, int(max_length * percen[0]), int(max_length * (percen[0] + percen[1])) + 1]
end = [int(max_length * (percen[0])), int(max_length * (percen[0] + percen[1])) + 1, max_length]

for c in range(0, len(class_name)):
	filename = dire + str(class_name[c]) + '.npy'
	data = np.load(filename)

	for s in range(0, len(step)):
		dire_step=str(dire) + str(step[s])
		if not os.path.exists(dire_step):
			os.makedirs(dire_step)

		for i in range(begin[s], end[s]):
			dire_class=str(dire_step) + '\\' + str(class_name[c])
			if not os.path.exists(dire_class):
				os.makedirs(dire_class)
			# Reshape the raw data into 28x28 images
			data_sample = data[i,:].reshape((28, 28))
			sample_name = class_name[c] + '_' + str(step[s]) + '_' + str(i)
			np.save(os.path.join(dire_class, sample_name), data_sample)
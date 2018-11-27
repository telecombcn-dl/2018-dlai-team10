import os
import numpy as np
import shutil
import random

"""	This script slices the dataset into training, validation and test subsets. The folders in which these subsets will be located must be provided.
	Why are we doing it like this? To load it with a dataloader that expects a dataset inheriting from torchvision.datasets.folder
	train_dir: Directory where the original files to sample are located.
	validation_dir: Directory where validation classes will be located.
	test_dir: Directory where test clases will be located.
	class_name: the class you are going to slice.
	Pre-conditions: You must have your dataset organized in folders with the name of each class (you can do it with save_npy_from_csv.py)
	training ---- apple  --- apple_0npy, apple_1.npy, ..., apple_17273.npy
			 ---- banana --- banana_0.npy, banana_1.npy, ..., banana_20081.npy
			 ---- book ...
			 ...
	
	validation ---- apple --- empty
			   ---- banana--- empty
	 		 ...

	test 	   ---- apple ---empty
			 ...
"""

train_dir = r"C:\Users\user\Ponç\MET\DLAI\Project\data\simplified_strokes_npy\train"
val_dir = r"C:\Users\user\Ponç\MET\DLAI\Project\data\simplified_strokes_npy\validation"
test_dir = r"C:\Users\user\Ponç\MET\DLAI\Project\data\simplified_strokes_npy\test"

class_name = "apple"

files = os.listdir(os.path.join(train_dir, class_name))
number_of_classes_in_original_dir = len(os.listdir(os.path.join(train_dir, class_name)))
print("Number of classes before slicing = " + str(number_of_classes_in_original_dir))
# We agreed to do 60% TRAINING, 30% VALIDATION, 10% TEST
validation_size = int(0.3 * number_of_classes_in_original_dir)
print("Number of classes for the validation set = " + str(validation_size))
# The line below generates validation_size indices between 0 and number_of_classes_in_original_dir
random_indices_validation = random.sample(range(0, number_of_classes_in_original_dir), validation_size)
#Once we have the indices we move the files
for i in random_indices_validation:
	current_path = os.path.join(train_dir, class_name, files[i])
	validation_path = os.path.join(val_dir, class_name, files[i])
	shutil.move(current_path, validation_path)

# Once the files are moved, we count the files again and build the test
files = os.listdir(os.path.join(train_dir, class_name))
number_of_classes_in_original_dir = len(os.listdir(os.path.join(train_dir, class_name)))
test_size = int(0.1 * number_of_classes_in_original_dir)
random_indices_test = random.sample(range(0, number_of_classes_in_original_dir), test_size)
#Once we have the indices we move the files
for i in random_indices_test:
	current_path = os.path.join(train_dir, class_name, files[i])
	validation_path = os.path.join(test_dir, class_name, files[i])
	shutil.move(current_path, validation_path)
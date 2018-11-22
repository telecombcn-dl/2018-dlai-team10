import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
	# This argument should be a .npy file containing drawings
	file_to_load = sys.argv[1]
	
	# Load the reduced images
	numpy_data = np.load(file_to_load)

	# Visualize the images in the npy matrix
	for i in range(0, len(numpy_data[:, 0])):
		# images are 28 x 28
		data_sample = numpy_data[i,:].reshape((28, 28))
		plt.imshow(data_sample)
		plt.show()
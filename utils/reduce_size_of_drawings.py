import numpy as np
import sys

if __name__ == '__main__':
	file_location = sys.argv[1]
	number_of_drawings_to_keep = sys.argv[2]
	outFile = sys.argv[3]
	x = np.load(file_location)
	np.save(outFile, x[:int(number_of_drawings_to_keep), :])

#virtualenv -p python3 env

import os
import numpy as np
import matplotlib.pyplot as plt
training_loss = []
validation_loss = []

training_ex = 6e5
plot_every = 200
#filename = r"C:\Users\user\Ponç\MET\DLAI\Project\training_results\CNN\CNN_model1_Adam_32_0c00001_50.txt"
filename = r"C:\Users\user\Ponç\MET\DLAI\Project\training_results\LSTM\LSTM_model1_Adam_32_0c02_10.txt"
parameters = filename.split("\\")[-1].split("_")
print(parameters)
model_type = parameters[0] # CNN or LSTM or MLP
model_id = parameters[1] 
optimizer = parameters[2]
bs = int(parameters[3]) # Batch size
learning_rate = parameters[4]
epochs = parameters[5]

if("CNN" in model_type):
	CNN = True
else:
	CNN = False


with open(filename) as f:
	for line in f:
		if("Training" in line and CNN):
			training_loss.append(float(str(line).split(":")[1][1:]))
		elif("Validation" in line and CNN):
			validation_loss.append(float(str(line).split(":")[1][1:]))
		elif(not CNN):
			training_loss.append(float(str(line[17:22])))

if(CNN):
	validation_loss = np.asarray(validation_loss)
	training_loss = np.asarray(training_loss)
	x_axis = np.asarray(range(0,len(training_loss)))
	print(validation_loss)
	print(training_loss)
	plt.title("Model_type = " + str(model_type) + " - Model_id = " + str(model_id) + " - Optim = " + str(optimizer) + " - Batch Size = " + str(bs) + " - Learning Rate = " + str(learning_rate))
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.plot(x_axis * bs *plot_every / training_ex, training_loss)
	plt.plot(x_axis * bs *plot_every/ training_ex, validation_loss)
	plt.show()
else:
	training_loss = np.asarray(training_loss)
	x_axis = np.asarray(range(0,len(training_loss)))
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.title("Model_type = " + str(model_type) + " - Model_id = " + str(model_id) + " - Optim = " + str(optimizer) + " - Batch Size = " + str(bs) + " - Learning Rate = " + str(learning_rate))
	plt.plot(x_axis * bs *plot_every/ training_ex, training_loss)
	plt.show()
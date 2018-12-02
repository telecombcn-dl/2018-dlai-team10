import torch
import torchvision
import numpy as np
import torch.utils.data
import torchvision.datasets as datasets
import torch.nn as nn
from torch.nn import functional as F

"""

	train    ---- apple  --- apple_0npy, apple_1.npy, ..., apple_17273.npy
			 ---- banana --- banana_0.npy, banana_1.npy, ..., banana_20081.npy
			 ---- book ...
			 ...
	
	validation ---- apple --- moltes apples...
			   ---- banana--- mltes bananes...
	 		 ...

	test 	   ---- apple ---moltes apples...
			 ...

"""

class Quick_draw_LSTM(nn.Module):
	def __init__(self, lstm_input_size, lstm_units, lstm_hidden_units, batch_size, output_dim):
		super(Quick_draw_LSTM, self).__init__()
		self.lstm_input_size = lstm_input_size
		# Number of stacked lstm that we will have 
		self.lstm_units = lstm_units

		self.lstm_hidden_units = lstm_hidden_units
		self.batch_size = batch_size
		self.output_dim = output_dim
		
		self.__build_model()

	def __build_model(self):
		self.lstm = nn.LSTM(input_size = self.lstm_input_size, hidden_size = self.lstm_hidden_units,num_layers = self.lstm_units)
		self.hidden_to_class = nn.Linear(self.lstm_hidden_units, self.output_dim)

	def init_hidden(self):
		hidden_a = torch.randn(self.lstm_units, self.batch_size, self.lstm_hidden_units)
		hidden_b = torch.randn(self.lstm_units, self.batch_size, self.lstm_hidden_units)
		return (hidden_a, hidden_b)

	def forward(self, X, X_lengths):
		# at the beginning of each sequence we must reset the hidden states
		self.hidden = self.init_hidden()
		seq_len, batch_size, features_size = X.size()
		"""We pack the batch with pack_padded_sequence, this method is useful because the LSTM won't see
		   the padded values. This function expects as arguments: a tensor of (T x B x *)
		"""
		X = nn.utils.rnn.pack_padded_sequence(X, X_lengths)
		X, self.hidden = self.lstm(X, self.hidden)
		X = nn.utils.rnn.pad_packed_sequence(X)
		print("El tipus de X[0] és: " +str(type(X[0])))
		
		X = X[0].contiguous()
		X = X.view(-1, X.shape[2])
		X = self.hidden_to_class(X)
		X = F.log_softmax(X, dim=1)
		X = X.view(seq_len, batch_size, self.output_dim)
		return X

def pad_tensor(my_vec, my_pad, my_dim):
    """
    args:
        my_vec - tensor to my_pad
        my_pad - the size to my_pad to
        my_dim - my_dimension to my_pad

    return:
        a new tensor my_padded to 'my_pad' in my_dimension 'my_dim'
    """
    my_pad_size = list(my_vec.shape)
    my_pad_size[my_dim] = my_pad - my_vec.size(my_dim)
    return torch.cat([my_vec, torch.zeros(*my_pad_size).double()], dim=my_dim)

class PadCollate:
	"""
	a variant of callate_fn that pads according to the longest sequence in
	a batch of sequences
	"""
	def __init__(self, dim=0):
		"""
		args:
		dim - the dimension to be padded (dimension of time in sequences)
		"""
		self.dim = dim


	def pad_collate(self, batch):
		"""
		args:
		batch - list of (tensor, label)
		reutrn:
		    xs - a tensor of all examples in 'batch' after padding
		    ys - a LongTensor of all labels in batch
		"""
		# find longest sequence
		lengths = np.flip(np.sort([x[0].shape[self.dim] for x in batch]), axis = 0)
		print(lengths)
		max_len = max(map(lambda x: x[0].shape[self.dim], batch))
		batch = list(map(lambda x: (pad_tensor(x[0], my_pad=max_len, my_dim=self.dim), x[1]), batch))
		# stack all
		xs = torch.stack(list(map(lambda x: x[0], batch)), dim=1)
		ys = torch.tensor(list(map(lambda x: x[1], batch)))
		return xs, ys, lengths

	def __call__(self, batch):
		return self.pad_collate(batch)


def load_sample(x):
	# We load them in format N x 2 in order to stack them in proper order
	return torch.from_numpy(np.load(x).T).double()

train_dir = r"C:\Users\user\Ponç\MET\DLAI\Project\data\simplified_strokes_npy\train"
val_dir = r"C:\Users\user\Ponç\MET\DLAI\Project\data\simplified_strokes_npy\validation"
test_dir = r"C:\Users\user\Ponç\MET\DLAI\Project\data\simplified_strokes_npy\test"

train_dataset = datasets.DatasetFolder(train_dir, extensions = ['.npy'], loader = load_sample)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 0, collate_fn = PadCollate(dim=0))
data_iter = iter(train_loader)
#data_iter = torch.utils.data.DataLoaderIter(train_loader)
# El que es printa hauria de ser un batch que entra per fer el forward pass i el seu backward pass corresponent quan entrenem
samples, labels, lengths = data_iter.next() # Jo crec que va per aqui 
print("El tamany del batch es: ")
print(samples.size())
print("El tamany de les labels es:")
print(labels.size())

#now sample a batch and forward propagate it through the network to see if it works
model = Quick_draw_LSTM(2, 1, 128, 8, 10)
X = model(samples.float(), lengths)
print(X[:,7,:].size())
print(X[:,7,:])
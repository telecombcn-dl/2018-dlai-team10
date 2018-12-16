# Deep Learning for Artificial Intelligence Project 
*Group 10: ETSETB Master students M. Alonso, M. Busquets, P. Palau and C. Pitarque*
# Index
1. Introduction

      1.1 Project Outline
  
      1.2 Quick, Draw! Doodle Recognition Challenge
      
  
2. Dataset

3. Models

      3.1 Multilayer Perceptron
  
      3.2 Convolutional Neural Network
      
      3.3 Long-Short Term Memory

4. Conclusions

5. Future Work

6. References

# 1. Introduction 

## 1.1  Project Outline

The main objective of this project was for us to deeply understand the concepts and implementations of various Deep Learning models studied in the course. [...]

We have implemented three different models [...]

## 1.1 Quick, Draw! Doodle Recognition Challenge
Our project consists on trying different approaches for *Kaggle's Quick, Draw! Doodle Recognition Challenge*.

Quick, Draw! is a game that was created in 2016 to educate the public in a playful way about how AI works. The basic idea of the game is that it tells the player a simple concept (such as banana, apple...) and he/she has to draw it in a certain amount of time. While the player is drawing, the AI [...]

However, since the training data comes from the game itself, drawings can be incomplete or may not match the label. The challenge consists on building a recognizer that can effectively learn from this **very noisy data** and perform well on a manually-labeled test set from a different distribution.

Competition link: https://www.kaggle.com/c/quickdraw-doodle-recognition


# 2. The Dataset

![clases](https://user-images.githubusercontent.com/43316350/50059288-43b93480-0185-11e9-8d9f-76695e781a8b.JPG) 

![datapercentage](https://user-images.githubusercontent.com/43316350/50059472-8a0f9300-0187-11e9-8ef4-8173e1488041.JPG)




# 3. Models
We decided to evaluate three different approaches of increasing difficulty and performance: a Multilayer Perceptron (MLP), a Convolutional Neural Network (CNN) and a Recurrent Nerual Network (RNN). 

For the first two approaches (MLP and CNN) we used the simplified dataset, in which the simplified drawings have been rendered into a 28x28 grayscale bitmap in numpy .npy format. While for the RNN, [...]

## 3.1 Multilayer Perceptron

To start, the first model to evaluate was a MLP (Multilayer Perceptron). It uses a supervised learning technique (backpropagation) to train a Neural Network. It can be distinguish from the liner perceptron because it uses multiple hidden layers:

![mlp layer](https://user-images.githubusercontent.com/10107933/50059826-1a4fd700-018c-11e9-8cae-279a28b2a5ef.JPG)

A ReLU Activation function and a Cross-Entropy Loss function have been used.

![loss function](https://user-images.githubusercontent.com/10107933/50059922-75ce9480-018d-11e9-98ff-0d0d729c3380.JPG)


A partciular MLP architectures have been evaluated to find the better performance. 

![models table](https://user-images.githubusercontent.com/10107933/50060014-85021200-018e-11e9-95d4-7268a5ce88c3.JPG)

First, we create a basic model with only three layers, and evaluate the dataset. Later on, we increase the complexity of the network trying to obtain a better performance.


We collect all the results for a better comparison:

![loss](https://user-images.githubusercontent.com/10107933/50060253-fa231680-0191-11e9-8718-8fcd8cc05ca7.JPG)


![accuracy](https://user-images.githubusercontent.com/10107933/50060309-9cdb9500-0192-11e9-8dc6-4b7059623b0e.JPG)

With this architectures, the best accuracy we obtained was 85.3% in the **Model 4**, however it could clearly be seen that the network was overfit.
To prevent this, we decided to implement a loss regularization by adding a new parameter in the cross-entropy loss (weight_decay)

![overfiting_formula](https://user-images.githubusercontent.com/10107933/50060387-7c600a80-0193-11e9-8955-5747c5705d66.JPG)

The regularization does not provide the desired results, but it does improve the previous results a little bit

![overfiting](https://user-images.githubusercontent.com/10107933/50060611-13c65d00-0196-11e9-87a6-331eb31168c0.JPG)


## 3.2 Convolutional Neural Network

Our motivation to tackle this problem of image classification using a CNN (Convolutional Neural Network) is quite obvious, because it is a specialized kind of neural network for processing data that has a known grid-like topology that leverages the ideas of local connectivity, parameter sharing and pooling/subsampling hidden units. *The basic idea behind a CNN is that the network learns hierarchical representations of the data with increasing levels of abstraction.*

We started creating the following basic CNN architecture and testing how it performed but, as it was very shallow it gave very poor results. In fact, most of the times it got stuck very soon in a local minimum, so the results were awful. 
![arquitecturacnn1](https://user-images.githubusercontent.com/43316350/50046296-bcdf5b80-00a1-11e9-8afe-7441718d35d3.JPG) 
With the purpose of improving the performance of the CNN, we deepened the network, so the probability of finding a *bad* local minimum decreased. We came up with the following structure that resulted to be excellent in terms of performance. This final architecture, which will be followingly explained, consists basically on alternating 5 convolutional layers (followed by a non-linearity and a batch normalization layer) with 2 max-pooling layers and, ending with 3 fully connected layers also followed by non-linearity. 
![arquitecturacnn3](https://user-images.githubusercontent.com/43316350/50046302-c963b400-00a1-11e9-90e4-769db06d6ec9.JPG)
The **Convolutional Layers** transform 3D input volume to a 3D output volume of neuron activations performing convolutions on a 2D grid. For the final architecture we have used 5 convolutional with a kernel size of 3x3 and of stride=1 each. They differ in the number of filters though, passing from 6 filters in the first layers to 16 and ending with 32 filters. These last characteristics (filter spatial extent, stride and number of filters) have been set as hyperparameters, which means that they their value is the one that has proven to give a better performance to the network after trying different ones. 

The **Non-linearity Layers** that we have used are ReLU (Rectified Linear Unit) Layers, which can be seen as simple range transforms that perform a simple pixel-based mapping that sets the negative values of the image to zero. 

We also introduced **Batch Normalization** layers (normalize the activations of each channel by subtracting the mean and dividing by the standard deviation), with the objective of simplifying, speeding up the training and reducing the sensitivity to network initialization. 

The network also contains two **Pooling Layers**, which are in charge of the down-sampling of the image and therefore reducing the number of activations, as well as providing invariance to small local changes. Four our architecture we have chosen to get the maximum values of 2x2 pixel rectangular regions around the input locations (that is, Max-Pooling with stride 2,2). It must be noted that we have just used two of these layers because the original size of our input data was already quite small (28x28 pixel images), so if we wanted a deep network, we could not afford adding pooling layers after each convolutional because we would have lost too much information about precise position of things. 

The **Fully-connected Layers** are the classic layers in which every neuron in the previous layer is connected to every neuron in the next layer and activation is computed as matrix multiplication plus bias. Here, the output of the last convolutional layer is flattened to a single vector which is input to a fully connected layer.

With this architecture, we obtained an accuracy on the test set of 89.4%, however in the training and validation plot of the losses and the accuracy, it could clearly be seen that the network was overfit.

![overfitting](https://user-images.githubusercontent.com/43316350/50059060-6b5acd80-0182-11e9-922b-3742113d2218.JPG)

To prevent this overfitting, we decided to implement **loss regularization**, though we could have used many other techniques such as early stopping, dropout, or data augmentation among others. We decided to add the L2 Regularization (or weight decay) to our cross-entropy loss. The L2 penalizes the complexity of the classifier by measuring the number of zeros in the weight vector. The resulting total loss is the following. 

![loss](https://user-images.githubusercontent.com/43316350/50059425-ee7e2280-0186-11e9-8973-6bcbf4670a88.JPG) 

Where *lambda* is the regularization hyperparameter (experimentally decided value).

Using this technique, we were able to obtain an accuracy value on the test set of 91.2%, where although some overfitting occurs, it is not as relevant as before. The results were the following:


## 3.3 LSTM (Long-Short Term Memory)

The model that we are going to train is an LSTM (Long-Short Term Memory Network). We selected this kind of model because we wanted to exploit the temporal information contained in the data.

First of all, we must consider the sizes of the tensors that the network is going to take as input. Our input are variable sized sequences with the format **L x 2**, where L is the length of the sequence, which is variable, and 2 is given by the keypoints in the drawing, which have a range between (0, 0) and (255, 255). We can represent the tensor for each sequence in a drawing:

![just_one_sequence_edited](https://user-images.githubusercontent.com/29488113/50059015-9395fc80-0181-11e9-8384-e37877491e28.jpg)

If we take a mini-batch of these sequences, we have a set of sequences of different length, as depicted in the following picture.

![batch_without padding_edited](https://user-images.githubusercontent.com/29488113/50059096-baa0fe00-0182-11e9-9c43-3259137fe03c.jpg)

Unfortunately, PyTorch can't work with batches of variable lengths. One option we could try is **working with a single sequence in each forward pass**, but that **is a bad idea because we will have a very poor gradient estimate and the training time would last forever**.
Luckily, PyTorch provides a solution which helps us feeding **zero-padded** mini-batches to our networks. That means that we will have good gradient estimations, which means shorter training time and better accuracy. Using these solutions, we can use mini-batch optimization algorithms. We pad each sequence with zeros according to the longest sequence length. Thus, we end with a batch of padded sequences that will have the size: **LONGEST_LENGTH x BATCH_SIZE x 2**. Our batches will look like:

![batch_padded_edited](https://user-images.githubusercontent.com/29488113/50059278-271cfc80-0185-11e9-87f0-e88c1240e109.jpg)

Having explained the input of our network, let us move on to the LSTM networks that we built. 

### LSTM Models

Since we are solving a classification problem, we will need a fully connected layer with softmax activation on top of the LSTM unit **in order to classify the extracted features coming from the LSTM hidden layer**. In this project, we built and trained 3 LSTM models, each one increasing the capacity of the previous one.





# 4. Conclusions

In this project, we have tackled for the first time a Deep Learning problem. We have created self-contained and detailed explainend notebooks, where all the pipeline characteristic of these kind of challenges is implemented **from scratch** (DL settings, data download and manipulation, architecture definition, training steps, validation and testing computation...). Each one of us has addressed the problem with a different approach, studying this way 3 different kinds of deep learning models as a team. We have faced typical deep learning problems such as overfitting, hyperparameter tuning and so on. 

As conclusions, we have found that the model that gives the best performance is the CNN (with an accuracy of 90%), while 



To conclude with, we would like to highlight that we have learned a lot with this project. 

# 5. Future Work

Many different adaptations, tests, and experiments have been left for the future due to lack of time. Followingly, we will briefly define in which directions these future work strands should go:

- **Time Optimization:** Although it has been a very helpful tool, during the implementation of this project, we have wasted a lot of time with Google Colab, due to the fact that execution times are restarded every 12h and then all progress is lost. Additionaly, very often, for unknown reasons the framework suddently disconected and we had to start over (set again the notebook, download the data, restructure it...). If we were to continue with this project, the first thing we would do would be to migrate all the content to Google Cloud. 

- **Challenge Adaptation**: If in the future it was intended to compete in the Kaggle competition (or just to compare the results with the competition's leaderboard - currently with a score of 0.95), many things should be changed. To begin with, the evaluation metrich should be changed to the Mean Average Precision at 3 (which is the one the competition performs) instead of the general accuracy. Furthermore, all the data provided by the competition should be used instead of a reduced version of just 10 of the more than 300 classes available. Moreover, we assume that to obtain competitive results, other models and architectures should also be considered.  

- **Deeper and enhanced analysis of the implemented models:** The Notebooks created could be enhanced by creating better tools to analyse the obtained results such as real-time losses and accuracy plots, computation of a confusion matrix and so on. 

- **Extracted features thorough study:** Deep analysis of how the format of the input data affects the extracted features of each model implementing an encoder/decoder (PONÇ). ![attention](https://user-images.githubusercontent.com/43316350/50055928-92e77100-0155-11e9-9939-533159151bc3.JPG)

# 6. References

•	ADAM Optimizer:  D. P. Kingma, J. L. Ba, *'ADAM: A Method For Stochastic Optimization'*. 

•	Training a classifier: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py 

•	Understanding the effect of the Batch Normalization layers: https://papers.nips.cc/paper/7996-understanding-batch-normalization.pdf

•	Understanding LSTM: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

•	Variable sized mini-batches: https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e 

•	Automatically load variable sized batches: https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8



--------------------------------------------------------------------------------------------------------------------------------

This project has been developed in Python 3.6.0 and using Google Colab Notebooks. It has been implemented in PyTorch 0.4.1

![logos](https://user-images.githubusercontent.com/43316350/50045436-ee9cf600-0092-11e9-8bdd-5f78347ec975.JPG) 

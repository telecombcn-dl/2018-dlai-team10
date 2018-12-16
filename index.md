# Deep Learning for Artifitial Intelligence Project 
*Group 10: ETSETB Master students M. Alonso, M. Busquets, P. Palau and C. Pitarque*
# Index
1. Introduction

      1.1 Project Outline
  
      1.2 Quick, Draw! Doodle Recognition Challenge
      
      1.3 Pipeline
  
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

## 1.3  Pipeline

The main objective of this project was for us to deeply understand the concepts and implementations of various Deep Learning models studied in the course. [...]

We have implemented three different models [...]

# 2. The Dataset



# 3. Models
We decided to evaluate three different approaches of increasing difficulty and performance: a Multilayer Perceptron (MLP), a Convolutional Neural Network (CNN) and a Recurrent Nerual Network (RNN). 

For the first two approaches (MLP and CNN) we used the simplified dataset, in which the simplified drawings have been rendered into a 28x28 grayscale bitmap in numpy .npy format. While for the RNN, [...]

## 3.1 Multilayer Perceptron

## 3.2 Convolutional Neural Network

Our motivation to tackle this problem of image classification using a CNN (Convolutional Neural Network) is quite obvious, because it is a specialized kind of neural network for processing data that has a known grid-like topology that leverages the ideas of local connectivity, parameter sharing and pooling/subsampling hidden units. 

*The basic idea behind a CNN is that the network learns hierarchical representations  of the data with increasing levels of abstraciton. *

We tried different shallow network architectures but the following deeper network resulted to be the best one in terms of performance.


![arquitecturacnn3](https://user-images.githubusercontent.com/43316350/50046302-c963b400-00a1-11e9-90e4-769db06d6ec9.JPG)


This final architecture, which will be followingly explained, consists basically on alternating 5 convolutional layers followed by a non-linearity with 2 max-pooling layers, ending with 3 fully connected layers also followed by non-linearity. 

The **Convolutional Layers**  transform 3D input volume to a 3D output volume of neuron activations performing convolutions on a 2D grid. For the final architecture we have used 5 convolutional with a kernel size of 3x3 and of stride=1 each. They differ in the number of filters though, passing from 6 filters in the first layers to 16 and ending with 32 filters. These last characteristics (filter spatial extent, stride and number of filters) have been set as hyperparameters, which means that they their value is the one that has proven to give a better performance to the network after trying different ones. 

The **Non-liniarity Layers** that we have used are ReLU (Rectified Linear Unit) Layers, which can be seen as simple range transforms that perform a simple pixel-based mapping that sets the negative values of the image to zero. 

The network also contains two **Pooling Layers**, which are in charge of the down-sampling of the image and therefore reducing the number of activations, as well as providing invariance to small local changes. Four our architecture we have chosen to get the maximum values of 2x2 pixel rectangular regions around the input locations (that is, Max-Pooling with stride 2,2). It must be noted that we have just used two of this layer because the original size of our input data was already quite small (28x28 pixel images), so if we wanted a deep network, we could not afford adding pooling layers after each convolutional because we would have lost too much information about precise position of things. 

The **Fully-connected Layers** are the classic layers in which every neuron in the previous layer is connected to every neuron in the next layer and activation is computed as matrix multiplication plus bias. Here, the output of the last convolutional layer is flattened to a single vector which is input to a fully connected layer. 

At some point, we decided to introduce **Batch Normalization** Layer (normalize the activations of each channel by subbstracting the mean and dividing by the standard deviation), with the objective of simplifying, speeding up the training and reducing the sensitivity to network initialization. However, this approached resulted to give a worse performance than the one obtained without adding those layers, so we decided to remove them. 


- **Stochastic Gradient Descent (SGD)**, which blabla


- **Adaptive Moments (ADAM)**, which is an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. 

### Architecture 1
I used different...
![arquitecturacnn1](https://user-images.githubusercontent.com/43316350/50046296-bcdf5b80-00a1-11e9-8afe-7441718d35d3.JPG)

### Architecture 2
![cnn model 3](https://user-images.githubusercontent.com/43316350/50045304-daf09000-0090-11e9-9cd8-61c0230a3f39.JPG)
)

![captura](https://user-images.githubusercontent.com/43316350/50052992-9535d500-012d-11e9-8f46-88ca463bbd49.JPG)

## Experiments

![loss accuracy_bones](https://user-images.githubusercontent.com/43316350/50053154-17bf9400-0130-11e9-96bd-fc6d5ef3294f.JPG)

## 3.3 LSTM (Long-Short Term Memory)

The model that we are going to train is an LSTM (Long-Short Term Memory Network). We selected this kind of model because we wanted to exploit the temporal information contained in the data.

First of all, we must consider the sizes of the tensors that the network is going to take as input. Our input are variable sized sequences with the format **L x 2**, where L is the length of the sequence, which is variable, and 2 is given by the keypoints in the drawing, which have a range between (0, 0) and (255, 255). We can represent the tensor for each sequence in a drawing:

![just_one_sequence_edited](https://user-images.githubusercontent.com/29488113/50059015-9395fc80-0181-11e9-8384-e37877491e28.jpg)

If we take a mini-batch of these sequences, we have a set of sequences of different length, as depicted in the following picture.

![batch_without padding_edited](https://user-images.githubusercontent.com/29488113/50059096-baa0fe00-0182-11e9-9c43-3259137fe03c.jpg)

Unfortunately, PyTorch can't work with batches of variable lengths. One option we could try is **working with a single sequence in each forward pass**, but that **is a bad idea because we will have a very poor gradient estimate and the training time would last forever**.
However, PyTorch provides a solution which helps us feeding **zero-padded** mini-batches to our networks.

We pad this sequences with zeros according to the longest sequence length. Thus, we end with a batch of padded sequence that will have the size: LONGEST_LENGTH x BATCH_SIZE x 2
Having explained the input to our network, we have to build our LSTM network. Since we are solving a classification problem, we will need a fully connected layer on top of the LSTM in order to classify the extracted features coming from the LSTM hidden layer.





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

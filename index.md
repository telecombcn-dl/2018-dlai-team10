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

The optimizer blabla. We tried two different optimizers:


- **Stochastic Gradient Descent (SGD)**, which blabla


- **Adaptive Moments (ADAM)**, which is an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. 

### Architecture 1
I used different...
![arquitecturacnn1](https://user-images.githubusercontent.com/43316350/50046296-bcdf5b80-00a1-11e9-8afe-7441718d35d3.JPG)

### Architecture 2
![cnn model 3](https://user-images.githubusercontent.com/43316350/50045304-daf09000-0090-11e9-9cd8-61c0230a3f39.JPG)
)
### Architecture 3
![arquitecturacnn3](https://user-images.githubusercontent.com/43316350/50046302-c963b400-00a1-11e9-90e4-769db06d6ec9.JPG)

![captura](https://user-images.githubusercontent.com/43316350/50052992-9535d500-012d-11e9-8f46-88ca463bbd49.JPG)

## Experiments

![loss accuracy_bones](https://user-images.githubusercontent.com/43316350/50053154-17bf9400-0130-11e9-96bd-fc6d5ef3294f.JPG)

## 3.1 Recurrent Neutal Network

# 4. Conclusions
Results [...]

The evaluation of the challenge is performed according to the Mean Average Precision @ 3 (MAP@3): MAP@3=1U∑u=1U∑k=1min(n,3)P(k)
where U is the number of scored drawings in the test data, P(k) is the precision at cutoff k, and n is the number predictions per drawing.

The current leader of the competition has a score of 0.95480. However, our results can not be compared to this because we have not implemented our models neither used all the data that the challenge provided. 

# 5. Future Work

Many different adaptations, tests, and experiments have been left for the future due to lack of time. Followingly, we will briefly define in which directions these future work strands should go:

- **Time Optimization:** 

- **Challenge Adaptation**: If in the future it was intended to compete in the Kaggle competition (or just to compare the results with the competition's leaderboard), many things should be changed. To begin with, the evaluation metrich should be changed to the Mean Average Precision at 3 (which is the one the competition performs) instead of the general accuracy. Furthermore, all the data provided by the competition should be used instead of a reduced version of just 10 of the more than 300 classes available. Moreover, we assume that to obtain competitive results, other models and architectures should also be considered.  

- **Deeper and enhanced analysis of the implemented models:** The Notebooks created could be enhanced by creating better tools to analyse the obtained results such as real-time losses and accuracy plots, computation of a confusion matrix and so on. 

- **Extracted features thorough study:** Deep analysis of how the format of the input data affects the extracted features of each model implementing an encoder/decoder (PONÇ). 

# 6. References

•	ADAM Optimizer:  D. P. Kingma, J. L. Ba, *'ADAM: A Method For Stochastic Optimization'*. 

•	Training a classifier: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py 



--------------------------------------------------------------------------------------------------------------------------------

This project has been developed in Python 3.6.0 and using Google Colab Notebooks. It has been implemented in PyTorch 0.4.1

![logos](https://user-images.githubusercontent.com/43316350/50045436-ee9cf600-0092-11e9-8bdd-5f78347ec975.JPG) 

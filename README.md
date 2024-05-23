# Fashion MNIST Classifier with Feed-Forward Neural Network

In this practical, we implement and train a feed-forward neural network (also known as a "Multi-Layer Perceptron" or MLP) on a dataset called "Fashion MNIST", consisting of small greyscale images of items of fashion.

## Learning Objectives
- Understand how to use Keras Layers to build a neural network architecture
- Understand how a model is trained and evaluated
- Understand the concept of train/validation/test split and why it's useful

## Project Overview
The objective of this project is to train a neural network to classify images of fashion items into one of the predefined categories. We will be using the Fashion MNIST dataset, which consists of 70,000 greyscale images divided into 60,000 training images and 10,000 test images. The goal is to train a model on the training set and evaluate its performance on the test set.

## Dataset
The Fashion MNIST dataset includes images of 10 different categories of fashion items:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

Each image is 28x28 pixels in greyscale.

## Project Structure
The main steps of the project are:
1. **Load and preprocess the data**
2. **Build the neural network architecture**
3. **Compile the model**
4. **Train the model**
5. **Evaluate the model**
6. **Visualize training results**

## Code Implementation

```python
from __future__ import print_function

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## Explanation
1. **Load and preprocess the data**: We start by loading the Fashion MNIST dataset and reshaping the data into vectors of size 784 (28x28) and normalizing the pixel values to the range [0, 1].
2. **Build the neural network architecture**: The network consists of an input layer, two hidden layers with 512 neurons each and ReLU activation, and a dropout layer to prevent overfitting. The output layer has 10 neurons with softmax activation to classify the images into one of the 10 categories.
3. **Compile the model**: We use categorical cross-entropy as the loss function, RMSprop as the optimizer, and accuracy as the evaluation metric.
4. **Train the model**: The model is trained on the training data with a validation split of 20%, for 20 epochs with a batch size of 128.
5. **Evaluate the model**: The model's performance is evaluated on the test set.
6. **Visualize training results**: We plot the training and validation accuracy and loss to visualize the model's performance over epochs.

## Conclusion
By following the steps above, we have built and trained a neural network to classify images from the Fashion MNIST dataset. This project demonstrated the use of Keras to build neural network architectures, train models, and evaluate their performance.



## License
This project is licensed under the MIT License.

## Acknowledgements
- [Keras Documentation](https://keras.io/)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)

# MNIST TensorFlow

This is a collection of Tensflow scripts for implementing neural networks to
recognize hand-written digits using the MNIST dataset for training and testing.

### `nn_digits_simple_tf2.py`
This is a TensorFlow 2.0 compatible script using Keras. It creates a simple 3-layer
neural network using stochastic gradient descent as the optimizer for the mean-squared-error
loss function. Activation functions for each layer is the sigmoid function.

It performs poorly, reaching an accuracy of ~60% at best.

### `nn_digits_adam_tf2.py`
This is a more robust version of the previous script that uses 'adam', a much
more efficient optimizer function, along with cross-entropy for the loss
function. An additional dropout layer is also incorporated. This version also
does not require the image label values to be one-hot encoded, allowing for
runs to be quicker.  It achieves an accuracy of 97% using half as many training
epochs as the simpler model.

### `nn_digits_cnn_tf2.py`
This is the best performing neural network for the MNIST dataset. It deploys
a 2D convolutional neural network for the first section, image feature
extraction. It then uses a final single 128-node hidden layer to process
the image data, before outputting to the 10-node output layer. Best accuracy 
so far is 98%.

### `test_models.py`
A script to load saved, trained models from the previous scripts and test the
neural networks against our own set of created hand-written digits.

The images were created with an image program using a digital pen/tablet and
saved as 28x28 PNG files. They are in the `images` directory.

### `nn_digits_tf1.py`
This is an original code written for Tensorflow 1 and is only included here for
demonstration purposes. 

It is adapted from the excellent tutorial presented at the [Katakoda
site](https://www.katacoda.com/basiafusinska/courses/tensorflow-getting-started/tensorflow-mnist-beginner)

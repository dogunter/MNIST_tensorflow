'''
A TensorFlow 2.0 script to train a neural network to recognize
hand-written digits using the MNIST dataset.

This makes use of the Keras deep learning library 
https://keras.io/
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os

'''
Create an mnist object
'''
mnist = tf.keras.datasets.mnist

'''
Load the MNSIT dataset, separating out the training and testing data
'''
(train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()

'''
Remember that the images from the MNIST dataset are grayscale pixel values in
the range 0-255. Divide all the image values by 255.0 in order to create a new
range 0-1
'''
train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0

'''
Create a keras model

A 'model' in keras is a way to organize the layers of a neural network.
Up to now, we have only dealt with neural networks whose layers are
treated sequentially, that is, one after another. We apply inputs via
an input layer and then progress through one or more hidden layers,
and on to an output layer. Backpropagation occurs sequentially as well.

So for now, we will create a sequential keras model for our image
recognition neural network

We will tell keras to create a sequential model and then specify the size
of each layer. We can also specify the activation function to use for each
layer. The first layer will not have an activation function since it is just the 
inputs being fed in.
   Layer 1 (input layer) will be 28x28 or 784 nodes
   Layer 2 (hidden layer) is 128 nodes and uses the relu activation function
   Layer 3 (hidden layer) is a dropout layer - we haven't covered this yet
   Layer 4 is our 10-node output layer with 'softmax' as the activation function

For more on the concept of dropout, read
https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5

Our dropout threshold is 0.2, meaning that after the first hidden layer, we create a second hidden layer
where each node of the first is kept with a probability of 0.8 (1.0 - 0.2).
'''
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

'''
Compile the model defined above. We specified the structure of our neural
network but now we need to define how the weights and biases will be evaluated.

We previously studied stochastic gradient descent (SGD) as a way to optimize
our error function. There are many other optimization models for use with
neural networks, one of which has been named 'adam'. It is described in detail
in this paper, https://arxiv.org/abs/1412.6980v8

'adam' is based on SGD but is highly computationally efficient with low
memory overhead. 

Now that we have specified the optimizer to use, we need to define what our
error function will be. Remember that this is also called a loss function and
"loss function" is the preferred term in neural network research. We used "mean
squared error" in our simply classroom exercises when we studied SGD.  There
are other ways to define loss functions and "categorical crossentropy" is one
of them. Coss-entropy is taken from information theory and it too complex to go
into here, but it is more robust and quicker to implement.

Finally, specify the metrics that will be used to adjust our weights and biases. We
want the best accuracy and we simply specify 'accuracy' for our metrics.
'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''
Everything is set to go. We only have to feed training data into
our neural network (model) and specify the number of training epochs
'''
n_epochs = 1
mini_batch_size = 32
model.fit(train_imgs, train_labels, epochs=n_epochs, batch_size=mini_batch_size)

'''
Now that the neural network has been trained, test it with
the testing dataset.
'''
model.evaluate(test_imgs, test_labels, verbose=2)

'''
Save the trained model so we can load it and use it later without having to
retrain.
'''
export_path = "."
export_dir = os.path.dirname(export_path)
model_name = 'nn_digits_adam.h5'
model.save(export_dir+model_name)
print( "Model saved to " + export_dir + model_name)
model.summary()

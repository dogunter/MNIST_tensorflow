import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation

'''
Create an mnist object
'''
mnist = keras.datasets.mnist

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
The images stored in the data files have been flattened and stored
as a continues stream of numbers. The following lines of code will
restore them back to 28x28 pixel images.
'''
train_imgs = train_imgs.reshape(-1, 28, 28, 1)
test_imgs = test_imgs.reshape(-1, 28, 28, 1)

'''
Recall how we vectorized the labels? That is, if a label was '3' then the
vectorized label was [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].

This is called one-hot encoding. Keras provides a function to achieve this,
the 'to_categorical()' function. It takes an array of digits and hot-encodes
them to arrays of n-digits, 10 in our case.
'''
train_labels_cat = keras.utils.np_utils.to_categorical(train_labels)
test_labels_cat = keras.utils.np_utils.to_categorical(test_labels)

'''
Create a keras model

A 'model' in keras is a way to organize the layers of a neural network.
Up to now, we have only dealt with neural networks whose layers are
treated sequentially, that is, one after another. We apply inputs via
an input layer and then progress through one or more hidden layers,
and on to an output layer. Backpropagation occurs sequentially as well.

Convolutional Neural Networks are also sequential networks.
So for now, we will create a sequential keras model for our image
recognition CNN.

We will tell keras to create a sequential model.
Next, we'll add the layers the model one by one.
We can also specify the activation function to use for each
layer. 

   Layer 1: A Conv2D (convolutional 2D layer) consisting of 32 filters, each
            of size 3x3, which will take 28x28 b&w images as input.
   Layer 2: Afer convolution, apply pooling with a pool size of 2. 
            Add the relu activation function to the output of the pooling layer.
   Layer 3: Flatten the data and connect it to the remaining fully-connected
            section of the neural network.
   Layer 4: Next hidden layer of 128 nodes using the relu activation function.
   Layer 5: Our 10-node output layer with 'softmax' as the activation function
'''
model = keras.models.Sequential()
# Layer 1
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
# Layer 2
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Activation('relu'))
# Layer 3
model.add(Flatten())
# Layer 4
model.add(Dense(128, activation='relu'))
# Layer 5
model.add(Dense(10, activation='softmax'))

'''
Compile the model defined above. We specified the structure of our neural
network but now we need to define how the weights and biases will be evaluated.

We are once again using the 'adam' optimizer, so this is nothing new.
For the loss function, we are also re-deploying the 'categorical_crossentropy'
method.

Finally, specify the metrics that will be used to adjust our weights and
biases. We want the best accuracy and we simply specify 'accuracy' for our
metrics.
'''
model.compile(optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
'''
Feed in the data for fitting (training) the model
'''
n_epochs = 2
mini_batch_size = 32
model.fit(train_imgs, train_labels_cat, epochs=n_epochs, batch_size=mini_batch_size, 
        verbose=1, validation_split=0.3)

'''
Test out the model with the test set of data
'''
print("Evaluating the model.")
model.evaluate(test_imgs, test_labels_cat, verbose=2)

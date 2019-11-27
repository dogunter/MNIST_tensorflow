'''
Compare the TensorFlow 2.0 models created by the other scripts.
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
# Since we are using keras as the main component to a TF backend
import keras

'''
Load our previously saved models
'''
model_name = 'nn_digits_adam.h5'
sgd_model = tf.keras.models.load_model('nn_digits_sgd.h5')
adam_model = tf.keras.models.load_model('nn_digits_adam.h5')
cnn_model = keras.models.load_model('nn_digits_cnn.h5')

'''
 Test the system on personal images, also of size 28 x 28, saved as PNG files.

 Use the convert function with the L parameter to reduce the 4D RGBA
 representation to one grayscale color channel. We store this as a numpy array
 and invert it using np.invert, because the current matrix represents black as
 0 and white as 255, whereas we need the opposite.
'''
digits = np.arange(10)
correct = 0
new_test_imgs_unconverted = []
new_test_imgs = []
new_test_labels = []
for i in digits:
   img_name = "images/david-"+str(i)+".png"
   test_img = Image.open(img_name).convert('L')
   new_test_imgs_unconverted.append(test_img)
   test_img = np.invert(test_img)
   new_test_imgs.append(test_img)
   new_test_labels.append(i)

new_test_imgs = np.asarray(new_test_imgs)/255.0
new_test_labels = np.asarray(new_test_labels, dtype=int)
new_hot_test_labels = tf.keras.utils.to_categorical(new_test_labels, 10)
cnn_test_imgs = new_test_imgs.reshape(-1, 28, 28, 1)

# Generate a dispaly of all the test images
fig = plt.figure(figsize=(8, 3))
for i in range(10):
    fig.add_subplot(1,10, i+1)
    plt.imshow(new_test_imgs_unconverted[i], cmap='gray')
    plt.axis('off')
plt.show()

print("SGD model predictions:")
predictions = sgd_model(new_test_imgs)
for i, logits in enumerate(predictions):
    predicted = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[predicted]
    print("Digit {},  prediction: {} ({:4.1f}%)".format(i, predicted, 100*p))

loss, accuracy = sgd_model.evaluate(new_test_imgs, new_hot_test_labels, verbose=2)
print('Restored SGD model, accuracy: {:5.2f}%'.format(100 * accuracy) )

print("adam model predictions:")
predictions = adam_model(new_test_imgs)
for i, logits in enumerate(predictions):
    predicted = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[predicted]
    print("Digit {},  prediction: {} ({:4.1f}%)".format(i, predicted, 100*p))

loss, accuracy = adam_model.evaluate(new_test_imgs, new_test_labels, verbose=2)
print('Restored adam model, accuracy: {:5.2f}%'.format(100 * accuracy) )

print("CNN model predictions:")
predictions = cnn_model.predict(cnn_test_imgs)
for i, logits in enumerate(predictions):
    predicted = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[predicted]
    print("Digit {},  prediction: {} ({:4.1f}%)".format(i, predicted, 100*p))

loss, accuracy = cnn_model.evaluate(cnn_test_imgs, new_hot_test_labels, verbose=2)
print('Restored CNN model, accuracy: {:5.2f}%'.format(100 * accuracy) )


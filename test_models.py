'''
Test the TensorFlow 2.0 models created by the other scripts.
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from PIL import Image
import numpy as np
import os

'''
Load a model
'''
model_name = 'nn_digits_adam.h5'
adam_model = tf.keras.models.load_model(model_name)

'''
 Test the system on personal images, also of size 28 x 28, saved as PNG files.

 Use the convert function with the L parameter to reduce the 4D RGBA
 representation to one grayscale color channel. We store this as a numpy array
 and invert it using np.invert, because the current matrix represents black as
 0 and white as 255, whereas we need the opposite.
'''
digits = np.arange(10)
correct = 0
new_test_imgs = []
new_test_labels = []
for i in digits:
   img_name = "images/david-"+str(i)+".png"
   test_img = np.invert(Image.open(img_name).convert('L'))
   new_test_imgs.append(test_img)
   new_test_labels.append(i)

new_test_imgs = np.asarray(new_test_imgs)/255.0
new_test_labels = np.asarray(new_test_labels, dtype=int)

loss, accuracy = adam_model.evaluate(new_test_imgs, new_test_labels, verbose=2)
print('Restored adam model, accuracy: {:5.2f}%'.format(100 * accuracy) )

# intro to tensorflow

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# mnist data is split into 3 parts 
# 55k datapoints of training data as mnist.train
# 10k datapoints of testing data as mnist.test
# 5k datapoints of validation data as mnist.validation

# each mnist datapoint ahs two parts, an image of the handwritten digit (x) and the corresponding label (y)
# both training and test sets contain images(as mnist.train.images) and corresponding labels (mnist.train.labels) 
# each image is 28x28 pixels => 784 datapoints interpreted as a 28x28 matrix
# softmax flattens the array to 784x1 matrix (list), :(
# but for math lets say the 55k mnist.train.images set is 55000x784 matrix (where the numbers are lines up side by side and each corresponding image has a label from 0-9 repsenting the image)
# in this tutorial / our labels are now called "one hot vectors" where one hot means that the vector is 0 in most dimensions and 1 in a single dimension  

# SOFTMAX REGRESSIONS
# since we know that every image in MNIST is of a digit 0-9 there are 10 possible outcomes.
# our goal is to look at an image and give probability fo it being each digit.
# soft max gives us a list of values between 0 and 1that ad up to 1. this is always teh final layer in more sophisticated neural networks
# softmax step 1: add up evidence of our input being in certain classes
#     in this case, to tally up that a given image is in a particular class, do a weighted sum of the pixel intesnsityies.
#     if the weight is negative, then the pixel having a high intensity is evidence aginst the image being in that class, (and vice versa)
# softmax step 2: convert that evidence into probabilities.
# biases: extra evidenece to say that some things are more likely independent of the input, these biases correspond to the different outputs 

# Implementing the Regression
# to do efficient numerical computing, tensorflow opertes outside of python, using NumPy you can do expensive cmatrix multiplication outside of Python,

import tensorflow as tf 

# describing symbolic variables x, a place holder that we'll input when we as TensorFlow to run a computation.
x = tf.placeholder(tf.float32, [None,784]) 

# assigning weights and biases for our model. these are 'like additional inputs'. TF handles it as a variable == modifiable tensor thatlives in TF's graph of intearcting operations. 
W = tf.Variable(tf.zeros([784,10])) #784, for flattened datapoints, 10 for 10 outcomes
b = tf.Variable(tf.zeros([10])) # ten biases for 10 possible outcomes

#implementing the model
y = tf.nn.softmax(tf.matmul(x,W) +b)

# TRAINING
# In order to train the model we need to define what it means to be good.
# typically in ML we define what it means to be bad rather than good, we try to minimize the error margin, make the model more precise (at the cost of computing costs)
# a common function that describes the loss of a model, Cross Entroy, it describes how inneficient our predictions are for describing the truth.

# to implement cross entropy we need to first add a new placeholder to input the correct answers:
y_ = tf.placeholder(tf.float32, [None, 10])

# implementing cross entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# first tf.low(y) took the log of y values,
# second tf.reduce sum(y_ * tf.log(y), red_ind=[1]) added the elements in the second dimension of y due to red_ind=[1]
# third tf.reduce_mean computes the mean over all teh examples in the batch

# tensor flow automatically knows the length of your computation graph due to back propagation algorithms,
# implementing the training step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#definte the operation but does not run it yet
init = tf.initialize_all_variables()

# launches model in a Session
sess = tf.Session()
sess.run(init)

# runs train_step 1000 times
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# for each i in the loop we run 100 pairs of x's and y's, to replace the placesholders defined earlier
# using small batches of random data is called STOCHASTI TRAINING, in this case we are doing STOCHASTIC GRADIENT DESCENT
# ideally. IDEALLY, we should use all data, but computation costs are too high

# EVALUATING OUR MODEL
# first, figure out where we predicted the correct label using tf.argmax, which yields the index of the high entry in a tensor along some axis
#     tf.argmax(y,1) is the label our model thinks is most likely for each input
#     tf.argmax(y_,1) is the correct label. 
#     tf.equal checks if the prediction matches the truth

# implementation of model evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# this gives us a list of booleans(remember the one_hot concept) where if the numbers y_ and y match (within error) the value at the index is set to 1 for true 0 for false.
# now compare which ones are correct we cast to floats and then take the mean [0,0,1,1] = 0.50
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels}))


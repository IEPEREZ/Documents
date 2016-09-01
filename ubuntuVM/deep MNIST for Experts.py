# deep MNIST for Experts
# load mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# here mnist is a lightweigt class which stores the trainins, validation and tests sets as NumPy arrays.
# it also provides a function for iterating through data minibatches

# Start TensorFlow Interactive Session
# TensorFlow uses a highly efficient C++ backend to do its computation.
# Tensfor flow creates teh computation Graph and then launches it in a session

import tensorflow as tf
sess = tf.InteractiveSession()


# softmax regression model

# placeholders where x = is a unique pixel in the 28x28 image, and y_ is the label to which it corresponds to
x = tf.placeholder(tf.float32, shape=[None, 784])  
y_ = tf.placeholder(tf.float32, shape=[None, 10]) 
# shpe argument to a placeholder is option but it allows tensorflow to automatically catch bugs stemming from inconsistent tensor shapes

# Variables
# defining weights and biases
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# initializing session and all variables
sess.run(tf.initialize_all_variables())

# Predicted class and Loss Function
# compute the softmax probabilities that are assigned to each class
y = tf.nn.softmax(tf.matmul(x,W) +b)

# computation for cross entropyusing bayes theorem
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# note: tf.reduce_sum  sums across all classes and tf.reduce_mean takes the average of those sums

# Train the Model
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# Evaluating the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#Evaluating the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# train step iteration
#for i in range(1000):
#	batch = mnist.train.next_batch(100)
#	train_step.run(feed_dict={x: batch[0], y_: batch[1]})


# building a multilayer convolutional Network

# Weight initialization
# we generate weights with a small amount noise to break symmtry, and prevent 0 gradients, 
# using ReLU neurons, the convention is to initialize them with a slightly positive inital bias to avoid dead neuron

# weight variable generation
def wegiht_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# bias variable generation
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Convolution and Pooling
# TensorFlow also gives us a lot of flexiblity in convolution and pooling operations.
# This handles concepts of stride size and boundaries???
# convolutions us stride of 1 and are zero padded to ensure that the len(output) == len(input)
# Pooling is plain old max pooling over 2x2 blocks ??? 

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# First Convolutional Layer
# this first layer will consist of convolution followed by max pooling.
# the convolution will compute 32 feature for each 5x5 patch, it's weight tensor will have a shape of [5, 5, 1, 32]
# where the first two dimensions are patch size(5x5), (1) is no. of input channels, and (32) is no of output channels  "I should make a physical map of my neural network ideas"

# setting convolution layer variables
W_conv1 = wegiht_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# to apply the layer, we first reshape x to a 4d tensor(it was 1d when using softmax), where second and third dimensions correspond to image width and height, and the finial dimension correspond to the number of color channels.

# reshaping x's
x_image = tf.reshape(x, [-1,28,28,1])

# convolve x_image with the weight tensor, add the bias and apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
# this second layer is the defining factor that will build a deep network, we stack several layers of this type.
# The second layer will have 64 featres for each 5x5 patch

# setting second convolution layer variables
W_conv2 = wegiht_variable([5,5,32,64])
b_conv2 = bias_variable([64])

# convolving and pooling (see line 96)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
# by doing these 2x2 max pool operations in the previous two layers we have effectively reduced the image to 7x7 pixels
# adding 1024 neurons to allow processing of the entire image
# we reshape the tensor from the pooling layer into a batch of vectors, mulltiply by a weight matrix, add a bias, and apply the ReLU

W_fc1 = wegiht_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout 
# to reduce overfitting we must apply a dropout before getting to the readout layer,
# create a placeholder for the probability that a neuron's output is kept during dropout.
# look more into dropout :(

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
# finally add the sacred softmax layer
W_fc2 = wegiht_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and Evaluate the Model
# train using new code, key differences:
#     replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer
#     includes additional parameter keep_prob in fee_dict to control the dropout rate
#     we will add logging to every 100th interation in the training process
for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i%100 ==0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("step%d, training, accuracy %g" %(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob: 0.5})
print("test accuracy %g" %accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.laels, keep_prob: 1.0}))
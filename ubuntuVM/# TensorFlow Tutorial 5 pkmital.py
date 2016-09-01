# TensorFlow Tutorial 5 pkmital.py

# imports 
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# defining the necessary functions from libs.utils in pkmitals repos
# montage 
def montage(W):
	# draw all filters (n_input * n_output filters) as a montage image separated by 1 pixel borders
	W = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
	n_plots =int(np.ceil(np.sqrt(W.shape[-1])))
	m = np.ones(
		(W.shape[0] * n_plots + n_plots +1,
		 W.shape[1] * n_plots + n_plots +1)) * 0.5
	for i in range(n_plots):
		for j in range(n_plots):
			this_filter = i * n_plots + j
			if this_filter < W.shape[-1]:
				m[1 + i + i * W.shape[0]:1 + i + (i+1) * W.shape[0], 
				1 + j + j * W.shape[1]:1 + j + (j+1) * W.shape[1]] = (np.squeeze(W[:, :, :, this_filter]))
	return m

# weight variable ()
def weight_variable(shape):
	# helper function to create a weight variable initialized w/ a normal distribution
	# params shape : list --> size of weight variable
	initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
	return tf.Variable(initial)

# bias variable ()
def bias_variable(shape):
	# helper function to create a bias variable initialized with a constant value.
	# params = shape : list --> size of ?? weight ?? variable
	initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
	return tf.Variable(initial)


# setup input to the network and true out label
# these are placeholders which we weill fill in later
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# since x is currently a 2D tensor as [batch, height*width], we need to reshape to a 4D tensor
# convolutional graphs need a 4D tensor. If one component of 'shape' is the speachial value -1
# the size of that dimension is computed so that the total size remains constant. 
# since we haven't defined the batch dimension's shape yet, we use -1 to denote that this dimension should not change size.
x_tensor = tf.reshape(x,[-1,28,28,1])

# SETTING UP TH EFIRST CONVOLUTIONAL LAYER
# weight matrix is [heigh x width input_channels x output channels]
filter_size = 5 
n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

# bias is [output_channels]
b_conv1 = bias_variable([n_filters_1])

# now we can build a graph which does the first layer of convolution:
# define out stride as batch x height x width x channels
# instead of pooling, we use stride of 2 and more layers with smaller filters.
h_conv1 = tf.nn.relu(
	tf.nn.conv2d(input=x_tensor,
		filter=W_conv1,
		strides=[1,2,2,1],
		padding='SAME') +
	b_conv1)

# and just like the first layer, add additional lyers to creater the deepnet 
n_filters_2 = 16
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
h_conv2 = tf.nn.relu(
	tf.nn.conv2d(input = h_conv1,
		filter = W_conv2,
		strides=[1,2,2,1],
		padding='SAME')+ b_conv2)

# We can not reshape so that we can connect to a fully connected layer:
h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * n_filters_2])

# creating a fully conected layer 
n_fc = 1024
W_fc1 = weight_variable([7 * 7 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

# adding a drop a layer to regularize and reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# always add a smooth layer of softmax:
W_fc2 = weight_variable([n_fc, 10])
b_fc2 = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# define loss / training functions
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

# monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# creating the new session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# train in minibatches and report accuracy:
batch_size = 100
n_epochs = 5
for epoch_i in range(n_epochs):
	for batch_i in range( mnist.train.num_examples // batch_size):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
	print(sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1.0}))

# now lets take a look at the kernels we've learned 
W = sess.run(W_conv1)
plt.imshow(montage(W / np.max(W)), cmap='coolwarm')
plt.show(block=True)


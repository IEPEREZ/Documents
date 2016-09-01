# TensorFlow Tutorial 2 pkmital
# imports 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# creating toy data
plt.ion()
n_observations = 100
fig, ax = plt.subplots(1,1)
xs = np.linspace(-3,3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
#ax.scatter(xs, ys)
#plt.plot()
#plt.show(block=True)

# tf.placeholders for teh input and output of the network,
# placeholders are variales which we need to fill in when we are ready to compute the graph
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# We will try to optimize min_(W, b) \\ (X*w +b) - y ||^2 ???
# the 'Variable()' constructor requires an initial value for teh variable,
# which can be a tensor of any type and shape. The initial Value defines teh type and shape of the variable.
# After construction the type and shape of the variable are fixed. 
# The Value can be changed using one of the assign methods
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
Y_pred = tf.add(tf.mul(X, W), b)

# loss function will measure the distance between out observations and predictions and average them over time. 
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2))/ (n_observations -1)
# ff we wanted to add regularization, we could add other terms to the cost (e.g., ridge regress has a paramter controlling the amount of 'shrinkage' over the norm of activations)
# the larger the shrinkage, the more robust to collinearity ???
##cost = tf.add(cost, tf.mul(1e-6, tf.global_norm([W])))

# using gradient descent to optimize W,b 
# performs a single step in teh negative gradient 
learning_rate = 0.01 # tf is this?
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# creating a session to use the graph
n_epochs = 1000
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	# fit all training data
	prev_training_cost = 0.0
	for epoch_i in range(n_epochs):
		for (x, y) in zip(xs, ys):
			sess.run(optimizer, feed_dict={X: x, Y:y})

		training_cost = sess.run(cost, feed_dict={X: x, Y:y})
		print(training_cost)

		if epoch_i % 200 == 0:
			ax.plot(xs, Y_pred.eval(feed_dict={X: xs}, session=sess), 'k', alpha=(epoch_i / n_epochs))
			fig.show()
			plt.draw()
		# allow the trainin to quich if we've reach a min
		if np.abs(prev_training_cost - training_cost) < 0.000001:
			break
		prev_training_cost = training_cost
plt.show()
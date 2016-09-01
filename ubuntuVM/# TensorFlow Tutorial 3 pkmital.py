# TensorFlow Tutorial 3 pkmital.py 

# %% imports 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# pick up toy data
plt.ion()
n_observations = 100
#fig, ax = plt.subplots(1, 1)
xs = np.linspace(-3,3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
#ax.scatter(xs, ys)
#plt.plot()
#plt.show(block=True)

# Set placeholders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# instead of single factors and biases we'll create a polynomial function
# of different polynomial degrees, we ill learn the influence that each
# degree of inputs (X^[0,1,2,3]) has on the final output Y
Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
for pow_i in range(1,5):
	W = tf.Variable(tf.random_normal([1]), name='weight_%d' %pow_i)
	Y_pred = tf.add(tf.mul(tf.pow(X, pow_i), W), Y_pred)

# Loss func will measure the distance between out observations and prediction and average over them
cost = tf.reduce_sum(tf.pow(Y_pred -Y, 2))/(n_observations -1)

# To add regularization, we could add other terms to the cost see the other tutorial

# gradient descent to optimize W, b
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# we create a session to use the computation graph:
n_epochs = 1000
tc_xs = []
tc_ys = []
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	prev_training_cost = 0.0
	for epoch_i in range(n_epochs):
		for (x, y) in zip(xs, ys):
			sess.run(optimizer, feed_dict={X: x, Y: y})

		training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})
		print(training_cost)
		tc_xs.append(epoch_i)
		tc_ys.append(training_cost)

		#	ax.plot(xs, Y_pred.eval(feed_dict={X: xs}, session=sess), 'k', alpha=epoch_i/n_epochs)
		#	plt.plot()
		#	plt.show(block=True)
		if np.abs(prev_training_cost - training_cost) < 0.00001:
			break
		prev_training_cost = training_cost
plt.scatter(tc_xs, tc_ys)
plt.plot()
plt.show(block = True)




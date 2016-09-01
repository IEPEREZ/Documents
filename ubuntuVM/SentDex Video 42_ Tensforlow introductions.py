#SentDex video 43 neural networks 

#x1 = tf.constant(5)
#x2 = tf.constant(6)

#result = tf.mul(x1,x2)
#print (result)

#with tf.Session() as sess:
#	print(sess.run(result))

"""
input > wgt > hid layer 1 (activation function) > wgts > hid layer 2 
(activation function) > weights > output layer 

compare output to intended output > cost or loss function
optimization function (optimizer) > minimize  cost (adamoptimize, SGD, Adagrad)

backpropagation 

feed forward + backprog = epos

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

# for matrices height x width 
x = tf.placeholder('float',[None, 784]) 
y = tf.placeholder('float', )

def neural_network_model(data):
	# (input_data * weights) + biases 
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal(n_classes))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases']) # (input_data * weights) + biases
	l1 = tf.nn.relu(l1) # threshold function

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases']) # (input_data * weights) + biases
	l2 = tf.nn.relu(l2) # threshold function

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases']) # (input_data * weights) + biases
	l3 = tf.nn.relu(l3) # threshold function

	output_layer = tf.matmul((l3, output_layer['weights']) + output_layer['biases'])

	return output_layer


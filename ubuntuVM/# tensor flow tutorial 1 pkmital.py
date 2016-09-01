# tensor flow tutorial 1 pkmital

# importing tensor flow and pyplot
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')

# tf. graph represents a collection of tf.operations 
# operations are created by writing out equations.
# by default, there is a graph: tf.get_default_graph()
# and new operations are added to this graph the result of a tf.Operation is atf.Tensor, which holds the values

# first a tfTensor
n_values = 32
x = tf.linspace(-3.0, 3.0, n_values)

# construct a tf.session to execute a graph
sess = tf.Session()
result = sess.run(x)

# Alt. pass a session to the eval func == fn:
x.eval(session=sess)
# note that x.eval() doesn't work b/c it must evaulate on an open session (question, would it work if using a giant with statement for with sess.tf.Session() as sess: state)

# to close a session
sess.close()

# to start an interative session
sess = tf.InteractiveSession()

# not you can use x.eval()
#print x.eval()

################################################################################

# running operations 
# use tf.Operawtion to create a gaussian ditribution with a sigma of 1 and a mean of 0

sigma = 1.0
mean = 0.0
z = (tf.exp(tf.neg(tf.pow(x-mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))) * (1.0/ (sigma * tf.sqrt(2.0 * 3.1415))))

# by default, new operation are added to the default graph asser z.graph = tf.get_default_graph
assert z.graph is tf.get_default_graph()

#execute the graph and plot the result
#plt.plot(z.eval())
#plt.show()

#find out the shape of a tensor
#print (z.get_shape())

#print(z.get_shape().as_list())

# sometimes we may not know the shape of atensor until it is computed in the graph. 
# In that case we should use teh tf.Shape fn, which will return a "tensor" which can be eval'd
# rather than a "discrete value" of tf.Dimension

# we can combine tensors like so:
#print(tf.pack([tf.shape(x), tf.shape(z), [3], [4]]).eval())

# lets multiply the two to get a 2D gaussian
z_2d = tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))

# Execute the graph and store the value that out represent in result???
#plt.imshow(z_2d.eval())
#plt.show()

# We can also list the operations in a graph
#ops = tf.get_default_graph().get_operations()
#print([op.name for op in ops])

# lets try creating a generic function for computing teh same thing:
def gabor(n_values=32, sigma=1.0, mean=0.0):
	x = tf.linspace(-3.0, 3.0, n_values)
	z = (tf.exp(tf.neg(tf.pow(x-mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))) * (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
	gauss_kernel = tf.matmul(
		tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))
	x = tf.reshape(tf.sin(tf.linspace(-3.0,3.0, n_values)), [n_values,1])
	y = tf.reshape(tf.ones_like(x), [1, n_values])
	gabor_kernel = tf.mul(tf.matmul(x, y), gauss_kernel)
	return gabor_kernel

# confirm this does somethin:
# plt.imshow(gabor().eval())
# plt.show()

# and another function which we can convolve
def convolve(img, W):
	# the W matrix is 2D 
	# but conv2d will need a tensor which is 4d :(
	# height, width, n input, n output
	if len(W.get_shape()) == 2:
		dims = W.get_shape().as_list() + [1,1]
		W = tf.reshape(W, dims)

	if len(img.get_shape()) == 2:
		# num, height, width, channels
		dims = [1] + img.get_shape().as_list() + [1]
		img = tf.reshape(img, dims)

	elif len(img.get_shape()) == 3:
		dims = [1] + img.get_shape().as_list()
		img = tf.reshape(img, dims)
		# if the image is 3 channels, then out convolution kernel needs to be repeated for each input channel
		W = tf.concat(2, [W,W,W])

	# Stride is how many values to skiip for teh dimension of num, height, width channels
	convolved = tf.nn.conv2d(img, W, strides=[1,1,1,1], padding='SAME')
	return convolved

# loading image:
from skimage import data
img = data.camera()
#plt.imshow(img)
#plt.show()
#print(img.shape)

# create a placeholder for our graph which can store any input:
x = tf.placeholder(tf.float32, shape=img.shape)

# create a graph which can convolve out image with a gabor
out = convolve(x, gabor())

#not send the image into the graph and compute the result
result = tf.squeeze(out).eval(feed_dict={x:img})
plt.imshow(result)
plt.show()

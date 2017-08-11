"""
Starter code for logistic regression model to solve OCR task 
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/

Accuracy: 0.944800019264
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import time

def multilayer_perceptron_3layers(x, weights, biases):
	layer_1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)

	output_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
	return output_layer

mnist = mnist_data.read_data_sets("/home/marchelo/CesarBragagnini/tf012_py27/Assigment01/data", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

x = tf.placeholder(tf.float32, shape=[None, n_input], name="placeholder_x")
y = tf.placeholder(tf.float32, shape=[None, n_classes], name="placeholder_y")

weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="h1"),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="h2"),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="w_out")	
}

biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="b1"),
	'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="b2"),
	'out': tf.Variable(tf.random_normal([n_classes]), name="b_out")
}

pred = multilayer_perceptron_3layers(x, weights, biases)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, name="entropy")
loss = tf.reduce_mean(entropy, name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())

	writer = tf.summary.FileWriter("/tmp/test", graph = sess.graph)
	for epoch in range(training_epochs):
		total_loss = 0 
		total_batch = int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y:batch_y})
			total_loss += c
		if epoch % display_step == 0:
			print "Average loss epoch {0}: {1}".format(epoch+1, total_loss/total_batch)
	print 'Total time: {0} seconds'.format(time.time()-start_time)
	print "Optimization Finished!"

	correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print "Accuracy: {0}".format(accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

	writer.close()




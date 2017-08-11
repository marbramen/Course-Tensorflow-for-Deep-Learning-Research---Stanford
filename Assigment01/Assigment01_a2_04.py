"""
Starter code for logistic regression model to solve OCR task 
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/

Accuracy: 0.982500016689
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import time
import math

def multilayer_perceptron_5layers(x, weights, biases):
	layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer1 = tf.nn.relu(layer1)

	layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
	layer2 = tf.nn.relu(layer2)

	layer3 = tf.add(tf.matmul(layer2, weights['h3']), biases['b3'])
	layer3 = tf.nn.relu(layer3)		

	layer4 = tf.add(tf.matmul(layer3, weights['h4']), biases['b4'])
	layer4 = tf.nn.relu(layer4)

	output_layer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
	return output_layer

mnist = mnist_data.read_data_sets("/home/marchelo/CesarBragagnini/tf012_py27/Assigment01/data", one_hot=True)

lr = tf.placeholder(tf.float32, name="placeholder_learning_rate")
min_learning_rate = 0.003
max_learning_rate = 0.0001
decay_speed = 2000


training_epochs = 1500
batch_size = 100
display_step = 1

n_input = 784
n_classes = 10
n_hidden_1 = 200
n_hidden_2 = 100
n_hidden_3 = 60
n_hidden_4 = 30

x = tf.placeholder(tf.float32, shape=[None, n_input], name="placeholder_x")
y = tf.placeholder(tf.float32, shape=[None, n_classes], name="placeholder_y")

weights = {
	'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1), name="var_h1") ,
	'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1), name="var_h2"),
	'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1), name="var_h3"),
	'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], stddev=0.1), name="var_h4"),
	'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_classes], stddev=0.1), name="var_wout") 
}

biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1], name="var_b1")), 
	'b2': tf.Variable(tf.random_normal([n_hidden_2], name="var_b2")),
	'b3': tf.Variable(tf.random_normal([n_hidden_3], name="var_b3")),
	'b4': tf.Variable(tf.random_normal([n_hidden_4], name="var_b4")),
	'out': tf.Variable(tf.random_normal([n_classes], name="var_wbias"))
}

pred = multilayer_perceptron_5layers(x, weights, biases)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y)
loss = tf.reduce_mean(entropy, name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)

with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())

	writer = tf.summary.FileWriter("/tmp/test", graph=sess.graph)
	for epoch in range(training_epochs):		
		total_loss = 0
		total_batch = int(mnist.train.num_examples/batch_size)
		learning_rate = min_learning_rate + (max_learning_rate-min_learning_rate)*math.exp(-epoch/decay_speed)

		for i in range(total_batch):
			x_batch, y_batch = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch, lr:learning_rate})
			total_loss += c
		if epoch % display_step == 0:
			print "Average loss epoch {0}: {1}".format(epoch+1, total_loss/total_batch)
	print "Total time {0} seconds".format(time.time() - start_time)
	print "Optimizatin finished"

	correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print "Accuracy: {0}".format(accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

	writer.close()




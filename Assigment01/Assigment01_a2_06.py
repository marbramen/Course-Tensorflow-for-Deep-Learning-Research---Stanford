"""
Starter code for logistic regression model to solve OCR task 
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/

Accuracy: 0.990000009537
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import time
import math
	
def multilayer_perceptron_5layers(x, weights, biases, size_fully_1, pkeep):
	stride = 1	
	layer1 = tf.nn.conv2d(x, weights['h1'], strides=[1,stride,stride,1], padding='SAME') + biases['b1']
	layer1 = tf.nn.relu(layer1)

	stride = 2
	layer2 = tf.nn.conv2d(layer1, weights['h2'], strides=[1,stride,stride,1], padding='SAME') + biases['b2']
	layer2 = tf.nn.relu(layer2)

	stride = 2
	layer3 = tf.nn.conv2d(layer2, weights['h3'], strides=[1,stride,stride,1], padding='SAME') + biases['b3']
	layer3 = tf.nn.relu(layer3)
	layer3 = tf.reshape(layer3, shape=[-1,7*7*size_fully_1])

	layer4 = tf.add(tf.matmul(layer3, weights['h4']), biases['b4'])
	layer4 = tf.nn.relu(layer4)
	layer4 = tf.nn.dropout(layer4, pkeep)

	output_layer = tf.add(tf.matmul(layer4, weights['out']), biases['out'])
	return output_layer

mnist = mnist_data.read_data_sets("/home/marchelo/CesarBragagnini/tf012_py27/Assigment01/data", one_hot=True, reshape=False, validation_size=0)

lr = tf.placeholder(tf.float32, name="placeholder_learning_rate")
pkeep = tf.placeholder(tf.float32, name="placeholder_learning_rate")

min_learning_rate = 0.003
max_learning_rate = 0.0001
decay_speed = 2000

training_epochs = 150
batch_size = 100
display_step = 1

size_image = 28
n_classes = 10
depth_layer_conv_1 = 4
depth_layer_conv_2 = 8
depth_layer_conv_3 = 12
layer_fully_conn = 200

x = tf.placeholder(tf.float32, shape=[None, size_image, size_image, 1], name="placeholder_x")
y = tf.placeholder(tf.float32, shape=[None, n_classes], name="placeholder_y")

weights = {
	'h1': tf.Variable(tf.truncated_normal([5,5,1,depth_layer_conv_1], stddev=0.1), name="var_h1") ,
	'h2': tf.Variable(tf.truncated_normal([4,4,depth_layer_conv_1,depth_layer_conv_2], stddev=0.1), name="var_h2"),
	'h3': tf.Variable(tf.truncated_normal([4,4,depth_layer_conv_2, depth_layer_conv_3], stddev=0.1), name="var_h3"),
	'h4': tf.Variable(tf.truncated_normal([7*7*depth_layer_conv_3,layer_fully_conn], stddev=0.1), name="var_h4"),
	'out': tf.Variable(tf.truncated_normal([layer_fully_conn, n_classes], stddev=0.1), name="var_wout") 
}

biases = {
	'b1': tf.Variable(tf.ones([depth_layer_conv_1])/10, name="var_b1"), 
	'b2': tf.Variable(tf.ones([depth_layer_conv_2])/10, name="var_b2"),
	'b3': tf.Variable(tf.ones([depth_layer_conv_3])/10, name="var_b3"),
	'b4': tf.Variable(tf.ones([layer_fully_conn])/10, name="var_b4"),
	'out': tf.Variable(tf.ones([n_classes])/10, name="var_wbias")
}

pred = multilayer_perceptron_5layers(x, weights, biases, depth_layer_conv_3, pkeep)

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
			_, c = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch, lr:learning_rate, pkeep: 0.75})
			total_loss += c
		if epoch % display_step == 0:
			print "Average loss epoch {0}: {1}".format(epoch+1, total_loss/total_batch)
	print "Total time {0} seconds".format(time.time() - start_time)
	print "Optimizatin finished"

	correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print "Accuracy: {0}".format(accuracy.eval({x: mnist.test.images, y:mnist.test.labels, pkeep:1.00}))

	writer.close()




"""
Starter code for logistic regression model to solve OCR task 
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/

Accuracy: 0.859099984169
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 15

mnist = input_data.read_data_sets('/home/marchelo/CesarBragagnini/tf012_py27/Assigment01/data', one_hot=True) 

X = tf.placeholder(tf.float32, shape=[None, 784], name="placeholder_X")
Y = tf.placeholder(tf.float32, shape=[None, 10], name="placeholder_Y")

W = tf.Variable(tf.zeros([784,10]), name="weight_1")
b = tf.Variable(tf.zeros([10]), name="bias")

logits = tf.add(tf.matmul(X,W), b, name="logits")
entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y, name = "entropy")
loss = tf.reduce_mean(entropy, name = "loss")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size)


	print logits

	writer = tf.summary.FileWriter("/tmp/test", graph = sess.graph)
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)			
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch}) 

			total_loss += loss_batch
		print 'Average loss epoch {0}: {1}'.format(i+1, total_loss/n_batches)

	print 'Total time: {0} seconds'.format(time.time() - start_time)
	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	pred = tf.nn.softmax(logits)
	correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print 'Accuracy {0}'.format(accuracy.eval({X:mnist.test.images, Y:mnist.test.labels}))	

	writer.close()

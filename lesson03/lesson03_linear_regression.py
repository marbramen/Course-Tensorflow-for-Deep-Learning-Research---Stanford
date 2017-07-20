"""
Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

tf.InteractiveSession()

DATA_FILE = '/home/marchelo/CesarBragagnini/tf012_py27/tf-stanford-tutorials/data/fire_theft.xls'

# Phase 1: Assemble the graph
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name="placeholder_X")
Y = tf.placeholder(tf.float32, name="placeholder_Y")

# Step 3: create weight and bias, initialized to 0
# name your variables w and b
w = tf.Variable(1.0, name="variable_w1")
b = tf.Variable(1.0, name="variable_bias")
#u = tf.Variable(1.0, name="variable_w2")

# Step 4: predict Y (number of theft) from the number of fire
# name your variable Y_predicted
#Y_predicted = tf.add_n([tf.mul(tf.mul(X,X),w),tf.mul(X,u), b])
Y_predicted = tf.add(tf.mul(X,w),b, name="Y_predicted")

# Step 5: use the square error as the loss function
# name your variable loss
loss = tf.square(Y - Y_predicted, name="variable_loss")

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
 
# Phase 2: Train our model
with tf.Session() as sess:
	# Step 7: initialize the necessary variables, in this case, w and b
	# TO - DO	
	
	##### one way to initialize
	#sess.run(w.initializer)
	#sess.run(b.initializer)
	##### second way to initialize
	sess.run(tf.global_variables_initializer())
	##### third way to initialize
	#sess.run(tf.variables_initializer([w,b]))
	writer = tf.summary.FileWriter("/tmp/test", graph=sess.graph)
	# Step 8: train the model
	for i in range(100): # run 100 epochs
		total_loss = 0
		for x, y in data:
			# Session runs optimizer to minimize loss and fetch the value of loss
			# TO DO: write sess.run()
			_, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
			total_loss += l
		print 'Epoch {0}: {1}'.format(i, total_loss/n_samples)
	w,b = sess.run([w,b])
	writer.close()

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()

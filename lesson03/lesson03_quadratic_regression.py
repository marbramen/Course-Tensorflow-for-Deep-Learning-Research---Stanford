import tensorflow as tf
import matplotlib.pyplot as plt
import xlrd
import numpy as np

DATA_FILE = '/home/marchelo/CesarBragagnini/tf012_py27/tf-stanford-tutorials/data/fire_theft.xls'

# go to construct: Y_pred = k1*X*X + k2*X + k3

# Phase 1: Assemble the graph
# Step 1: read the data form .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

#Step 2: Create placeHolders
X = tf.placeholder(tf.float32, name="placeholder_X")
Y = tf.placeholder(tf.float32, name="placeholder_Y")

#Step 2: Create Variables
k1 = tf.Variable(0.0, name="variable_k1")	
k2 = tf.Variable(1.0, name="variable_k2")
k3 = tf.Variable(0.0, dtype=tf.float32, name="variable_k3")

#Step 3: Create Y_pred
#Y_pred = X * X * k1 + X * k2 + k3
Y_pred = X * X * k1 + X * k2 + k3

#Step 4: calculate Loss
loss = tf.square(Y - Y_pred, name="variable_loss1")

#Step 5: optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.variables_initializer([k1,k2,k3]))
	print sess.run(k3)
	writer = tf.summary.FileWriter("/tmp/test", graph = sess.graph)	
	for i in range(10):
		total_loss = 0	
		for x, y in data:			
			_, l, y_pred1 = sess.run([optimizer, loss, Y_pred], feed_dict={X: x, Y: y})			
			print sess.run(Y_pred, {X:x, Y:y})
			print i
			total_loss += l

		print "Epoch {0}: {1}".format(i, total_loss/n_samples)	
	k1,k2,k3 =	sess.run([k1,k2,k3])
	writer.close()

# plot the results
X,Y = data.T[0], data.T[1]
plt.plot(X,Y, 'bo', label='Real data')
plt.plot(X, k1*X*X + k2*X +k3, 'r', label='Predicted data')
plt.legend()
plt.show()











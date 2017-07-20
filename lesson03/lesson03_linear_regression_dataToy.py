import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = tf.placeholder(tf.float32, name="placeholder_X")
Y = tf.placeholder(tf.float32, name="placeholder_Y")

W = tf.Variable(0.0, name="weight_1")
b = tf.Variable(0.0, name="bias")

Y_pred = X * W + b

loss = tf.square(Y - Y_pred, name="var_loss")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

num_samples = 100
x_input = np.linspace(-1,1,num_samples)
y_input = x_input * 3 + np.random.random(x_input.shape[0])*0.5

with tf.Session() as sess:
	sess.run(tf.variables_initializer([W,b]))
	for ii in range(250):
		total_loss = 0.0
		for i in range(num_samples):
			_, l = sess.run([optimizer, loss], feed_dict={X: x_input[i-1], Y: y_input[i-1]})
			total_loss += l
		print 'Epoch {0}: {1}'.format(ii, total_loss / num_samples)
	[W,b] = sess.run([W,b])


plt.plot(x_input, y_input, 'bo', label='Real data')
plt.plot(x_input, x_input * W + b, 'r', label='Predicted data')
plt.legend()
plt.show()

import tensorflow as tf

a = tf.constant(2, name="a")
b = tf.constant(2, name="b")
x = tf.mul(a,b, name="mul")

with tf.Session() as sess:
	writer = tf.summary.FileWriter('/tmp/test', sess.graph)
	print sess.run(a)
	print sess.run(x)

sess.close()	


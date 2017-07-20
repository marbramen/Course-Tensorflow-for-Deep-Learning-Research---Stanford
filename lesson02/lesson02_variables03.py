import tensorflow as tf

a = tf.Variable(2, name="scalar")
b = tf.Variable([2,3], name="vector")
c = tf.Variable([[1,2],[3,4]], name="matrix")

with tf.Session() as sess:
	sess.run(c.initializer)
	print c
	print sess.run(c)
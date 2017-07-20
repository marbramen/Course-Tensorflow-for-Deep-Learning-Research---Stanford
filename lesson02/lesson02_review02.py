import tensorflow as tf

a = tf.Variable(tf.truncated_normal([7]), name="vector")
b = tf.Variable([[1,2],[3,4]], name="matrix")
c = tf.Variable(1, name="scalar")

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print sess.run(a)
	print sess.run(b)
	print sess.run(c)

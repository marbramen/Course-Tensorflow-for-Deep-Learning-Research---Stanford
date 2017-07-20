import tensorflow as tf

a = tf.Variable(2, name="scalar")
b = tf.Variable([2,3], name="vector")
c = tf.Variable([[1,2],[3,4]], name="matrix")

init_ab = tf.variables_initializer([a,b], name="init_ab")

with tf.Session() as sess:
	sess.run(init_ab)
	print sess.run(a)
	print sess.run(b)		
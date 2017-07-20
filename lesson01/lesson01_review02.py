import tensorflow as tf

op1 = tf.add(2,3)
op2 = tf.mul(8, op1)
op3 = tf.pow(op2,op1)

with tf.Session() as sess:
	print sess.run(op2)

tf.summary.FileWriter('/tmp/test', sess.graph)
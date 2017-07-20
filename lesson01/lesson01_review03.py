import tensorflow as tf

g1 = tf.Graph()
g2 = tf.get_default_graph()


with tf.Session(graph = g1) as sess1:
	op1 = tf.add(2.0,3)
	op2 = tf.add(2.0,3.0)
	op3 = tf.mul(op1, op2)
	print sess1.run(op3)

with tf.Session(graph = g2) as sess2:
	op4 = tf.add(3,2)
	op5 = tf.pow(op4, op4)
	print sess2.run(op5)

tf.summary.FileWriter('/tmp/test', sess1.graph)	
import tensorflow as tf
x = 2
y = 3

g1 = tf.get_default_graph()
g2 = tf.Graph()

op1 = tf.add(x,y)
useless = tf.add(op1,x)
op2 = tf.mul(x,y)
op3 = tf.pow(op1,op2)

with g2.as_default():
	op4 = tf.add(x,y)
	op5 = tf.add(op4, x)

with tf.Session(graph=g2) as sess:
	print sess.run(op4)

write = tf.summary.FileWriter("/tmp/test", sess.graph)

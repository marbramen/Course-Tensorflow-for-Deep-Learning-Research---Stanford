import tensorflow as tf 
g1 = tf.get_default_graph()
g2 = tf.Graph()

with g1.as_default():
	a = tf.constant(3)

with g2.as_default():
	b = tf.constant(5)

with tf.Session(graph=g1) as sess:
	print sess.run(a)

with tf.Session(graph=g2) as sess:
	print sess.run(b)	


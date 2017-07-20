import tensorflow as tf

a = tf.Variable(tf.truncated_normal([7]), name="vector")
b = tf.Variable([[1,2],[3,4]], name="matrix")
c = tf.Variable(1, name="scalar")

g1 = tf.get_default_graph()
g2 = tf.get_default_graph()

init_ab = tf.variables_initializer([a,b], name="init_ab")
init_c = tf.variables_initializer([c], name="init_c")

with tf.Session(graph = g1) as sess1:
	sess1.run(init_ab)	
	print sess1.run(a)
	print sess1.run(b)

with tf.Session(graph = g2)	as sess2:
	sess2.run(init_c)
	print sess2.run(c)
	



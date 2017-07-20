import tensorflow as tf

a = tf.Variable(2, name="scalar")
b = tf.Variable([2,3], name="vector")
c = tf.Variable([[1,2],[3,4]], name="matrix")
w1 = tf.Variable(tf.truncated_normal([14,10]))
w2 = tf.Variable(100)
w3 = tf.Variable(200)
w4 = w3.assign(20)

g1 = tf.get_default_graph()
g2 = tf.get_default_graph()

init_ab = tf.variables_initializer([a,b,w1,w2,w3], name="init_a")

with tf.Session(graph=g1) as sess1:
	sess1.run(c.initializer)	
	print sess1.run(c)

with tf.Session(graph=g2) as sess2:
	sess2.run(init_ab)
	sess2.run(w2.assign(10))
	print sess2.run(a)
	print sess2.run(b)
	print sess2.run(w1)
	print sess2.run(w2)
	print sess2.run(w3)
	print sess2.run(w4)

write = tf.summary.FileWriter("/tmp/test", graph=g1)	
write.close()


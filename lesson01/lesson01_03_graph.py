import tensorflow as tf 
g = tf.Graph()

with g.as_default():
	a = 3
	b = 5
	x = tf.add(a,b)

print x
sess = tf.Session(graph = g)
print sess.run(x)
sess.close()

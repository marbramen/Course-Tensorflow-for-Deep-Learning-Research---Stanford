import tensorflow as tf

#tf.zeros(shape, dtype=tf.float32, name=None)
a = tf.zeros([2,3], dtype=tf.int32)

#tf.zeros(shape, dtype=tf.float32, name=None, optimize=True)
b = tf.zeros_like(a)

#tf.ones(shape, dtype=tf.float32, name=None)
c = tf.ones([3,3,3], dtype=tf.float32)

#tf.ones_like(shape, dtype=tf.float32, name=None, optimize=True)
d = tf.ones_like(c, dtype=tf.float32)

#tf.fill(shape, value, name=None)
e = tf.fill([2,3], 69)

#tf.linspace(star, stop, num, name=None)
f = tf.linspace(10.0, 13.0, 10, name="linspace")

#tf.range(start, limit=None, delta=1, dtype=None, name='range')
g1 = tf.range(7,20,3)
g2 = tf.range(20,7,-3)
g3 = tf.range(5)

with tf.Session() as sess:
	print "zeros():"
	print sess.run(a)
	print "zeros_like():"
	print sess.run(b)
	print "ones():"
	print sess.run(c)
	print "ones_like():"
	print sess.run(d)
	print "fill():"
	print sess.run(e)
	print "linspace():"
	print sess.run(f)
	print "range():"
	print sess.run(g1)
	print sess.run(g2)
	print sess.run(g3)
	writer = tf.summary.FileWriter("/tmp/test", sess.graph)


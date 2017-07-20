import tensorflow as tf

# tf.constant(value, shape=None, name=None, verify_shape=False)
a_const = tf.constant([24,69], shape=[2], name="a_const", verify_shape=True)
a_const_error = tf.constant(2, shape=[2,2], name="b_const", verify_shape=False)

# operations
# tf.zeros(shape, dtype=tf.float32, name=None)
# tf.zeros_like(shape, dtype=tf.float32, name=None, optimize=True)
# tf.ones(shape, dtype=tf.float32, name=None)
# tf.ones_like(shape, dtype=tf.float32, name=None, optimize=True)
# tf.fill(shape, value, name=None)
# tf.linspace(star, stop, num, name=None)
# tf.range(star, limit=None, delta=1, dtype=None, name="range")

zeros_a = tf.zeros([2,3], name="zeros_a")
zeros_b = tf.zeros_like(zeros_a, name="zeros_b")
ones_a = tf.ones([7], name="ones_a")
ones_b = tf.ones_like(ones_a)
linspace_a = tf.linspace(10.0,20.0,5,name="linspace_a")
range_a = tf.range(10, 20, 0.1, dtype=tf.float32, name="range_a")

with tf.Session() as sess:
	print sess.run(a_const)
	print sess.run(a_const_error)
	print sess.run(zeros_a)
	print sess.run(zeros_b)
	print sess.run(ones_a)
	print sess.run(ones_b)
	print sess.run(linspace_a)
	print sess.run(range_a)

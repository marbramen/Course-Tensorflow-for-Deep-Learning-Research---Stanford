import tensorflow as tf

# tf.placeholder(dtype, shape=None, name=None)

a = tf.placeholder(dtype=tf.float32, shape=[3], name="a_ph")
dict_a = {a:[1.0,4.2,4.5]}
b_constant = tf.constant([1.0,2.0,3.0], shape=[3], name="b_constant")
c = a + b_constant
list_value = [[1,2,3],[4,5,6],[7,8,9]]

with tf.Session() as sess:
	print sess.run(c,{a:[6.0,7.0,8.0]})
	for a_value in list_value:
		print sess.run(c, {a:a_value})
		print sess.run(c, feed_dict=dict_a)


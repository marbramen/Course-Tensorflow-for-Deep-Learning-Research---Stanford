import tensorflow as tf

# tf.placeholder(dytpe, shape=None, name=None)

a = tf.placeholder(tf.int32, shape=[3])
b = tf.constant([1,2,3], dtype=tf.int32)
c = a + b
list_values = [[10,10,10],[20,20,20],[30,30,30]]

# considerar que seteamos el valor del placeholder usando un dictonary donde el key puede ser un placeholder

with tf.Session() as sess:
	print sess.run(c, {a:[3,4,5]})
	dict_a = {a:[100,100,100]}
	for a_values in list_values:

		print sess.run(c, {a:a_values})
		print sess.run(c, feed_dict=dict_a)

writer = tf.summary.FileWriter("/tmp/test", sess.graph)	
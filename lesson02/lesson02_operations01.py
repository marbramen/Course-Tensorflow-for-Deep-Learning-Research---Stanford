import tensorflow as tf

a = tf.constant([3,6])
b = tf.constant([7,4])
c = tf.add_n([a,b,a])

d = tf.matmul(tf.reshape(a, [1,2]), tf.reshape(b, [2,1]))

with tf.Session() as sess:
	print sess.run(c)
	print sess.run(d)




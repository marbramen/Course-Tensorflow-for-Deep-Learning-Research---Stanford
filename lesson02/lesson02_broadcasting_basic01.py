import tensorflow as tf

x = tf.constant([2,1], shape=[2], name="constX")
y = tf.constant([[3,4],[5,6]], shape=[2,2], verify_shape=True, name="constY")
z = tf.add(x,y, name="sum_X_Y")

with tf.Session() as sess:
	print sess.run(x)
	print sess.run(y)
	print sess.run(z)
	writer = tf.summary.FileWriter("/tmp/test", sess.graph)

writer.close()


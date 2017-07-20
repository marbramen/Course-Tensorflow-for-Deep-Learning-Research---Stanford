import tensorflow as tf 

a = tf.Variable(2, name="scalar")
b = tf.Variable([2,3], name="array")
c = tf.Variable([[0,1], [2,3], [4,5]], name="matrix")

# create a variable as tensor of 784x10, filled with zeros
W = tf.Variable(tf.zeros([784,10]))

init = tf.global_variables_initializer()


with tf.Session() as sess:		
	sess.run(init)
	print "Scalar:"
	print a
	print sess.run(a)
	print "matrix:"
	print c
	print c.eval()
	print sess.run(c)
	tf.summary.FileWriter("/tmp/test", sess.graph)



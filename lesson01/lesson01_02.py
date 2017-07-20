import tensorflow as tf
x = 2
y = 3
op1 = tf.add(x,y)
op2 = tf.mul(x,y)
op3 = tf.pow(op1,op2)
with tf.Session() as sess:
	print sess.run(op1)
	print sess.run(op2)
	print sess.run(op3)

tf.train.write_graph(sess.graph_def, '/home/marchelo/CesarBragagnini/tf012_py27/lesson01/fileLesson01', 'graph.pbtxt')



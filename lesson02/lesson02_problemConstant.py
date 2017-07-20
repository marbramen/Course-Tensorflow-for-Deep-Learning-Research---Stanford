import tensorflow as tf

myConst = tf.constant([1.0,2.0], name="myConst")
myTensor = tf.zeros([2,2], dtype=tf.float32)
myLinspce = tf.linspace(10.0, 20.0, 5)

# to print graph, print all content of the constant 
# so this may be expensive
with tf.Session() as sess:
	print sess.graph.as_graph_def()

import tensorflow as tf

a = tf.Variable(2, name="var_a")
b = a.assign(a * 3)

g1 = tf.get_default_graph()

# no es necesario initializer la b 
init = tf.variables_initializer([a], name="init_ab")

# si se inicializa la b , genera error
# return control_flow_ops.group(*[v.initializer for v in var_list], name=name)
#init = tf.variables_initializer([a,b], name="init_ab")

with tf.Session(graph = g1) as sess:
	sess.run(init)
	print sess.run(a)
	print sess.run(b)
	print sess.run(b)
	tf.summary.FileWriter("/tmp/test", graph = g1)

import tensorflow as tf

"""
#Normal Loading
a_var = tf.Variable(tf.truncated_normal([3]))
b_var = tf.Variable(tf.truncated_normal([3]))
c_var = a_var + b_var

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		print sess.run(c_var)	
	print tf.get_default_graph().as_graph_def()		
"""

#Lazy Loading
a_var = tf.Variable(tf.truncated_normal([3]))
b_var = tf.Variable(tf.truncated_normal([3]))

init_ab = tf.variables_initializer([a_var,b_var],name="init_ab")

with tf.Session() as sess:
	sess.run(init_ab)
	for _ in range(3):
		print sess.run(a_var + b_var)
	print tf.get_default_graph().as_graph_def()


""" 
There are two ways to avoid this bug. First, always separate the definition of ops and their
execution when you can. But when it is not possible because you want to group related ops into
classes, you can use Python property to ensure that your function is only loaded once when it's
first called. This is not a Python course so I won't dig into how to do it. But if you want to know,
check out this wonderful blog post by Danijar Hafner ( http://danijar.com/structuring-your-tensorflow-models/ )
session vs graph: https://danijar.com/what-is-a-tensorflow-session/
"""

import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()  #define a session
# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x = 2
y = 3
op1 = tf.add(x,y)
op2 = tf.mul(x,y)
op3 = tf.pow(op1,op2)

# Before starting, initialize the variables.  We will 'run' this first.
# Launch the graph.
sess = tf.Session()
sess.run(op3)
sess.close()
	
#### ----> ADD THIS LINE <---- ####
writer = tf.train.SummaryWriter("/tmp/test", sess.graph)




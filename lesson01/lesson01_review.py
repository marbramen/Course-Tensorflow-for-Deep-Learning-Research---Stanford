import tensorflow as tf

x = 2 
y = 3
op1 = tf.add(x,y)
op2 = tf.mul(op1, op1)

sess = tf.Session()
print sess.run(op2)
sess.close()

file = tf.summary.FileWriter('/tmp/test', sess.graph)

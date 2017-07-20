import tensorflow as tf
x = 2
y = 3

op1 = tf.add(x,y)
useless = tf.add(op1,x)
op2 = tf.mul(x,y)
op3 = tf.pow(op1,op2)

write = tf.summary.FileWriter("/tmp/test", tf.Session().graph)

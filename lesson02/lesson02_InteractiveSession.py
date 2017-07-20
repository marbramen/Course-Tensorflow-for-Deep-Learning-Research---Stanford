import tensorflow as tf

a = tf.constant(10)
b = tf.constant(2)
c = a + b

sess = tf.InteractiveSession()
print c.eval()
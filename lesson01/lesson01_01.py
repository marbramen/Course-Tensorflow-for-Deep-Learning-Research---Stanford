import tensorflow as tf
a = tf.add(2,3)
print a
sess = tf.Session()
print sess.run(a)
sess.close()
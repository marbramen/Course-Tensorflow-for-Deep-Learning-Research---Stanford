import tensorflow as tf 

sess = tf.InteractiveSession()

#tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

# esta linea bota error por que verifica el shape
#a = tf.constant(2, shape=[2,2], verify_shape=True)

# no bota error porque no esta verificando el shape
a =  tf.constant(2, shape=[2,2], verify_shape=False)
print sess.run(a)
print a.eval()

# cumple el shape
b = tf.constant([[3,3],[3,3]], shape=[2,2], verify_shape=True, name="constB")
print b.eval()
print sess.run(b)



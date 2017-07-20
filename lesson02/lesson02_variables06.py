import tensorflow as tf

W = tf.Variable(10)
V = W.assign_add(10*W)

sess1 = tf.Session()
sess2 = tf.Session()

init1 = sess1.run(W.initializer)
init2 = sess2.run(W.initializer)

print "sess1"
print "sess1.run(V): %d"%sess1.run(V)
print "sess1.run(W.assign_add(10)): %d"%sess1.run(W.assign_add(10))
print "sess1.run(V): %d"%sess1.run(V)
print "sess2.run(W.assign_sub(2)): %d"%sess1.run(W.assign_sub(2))
print "sess1.run(V): %d"%sess1.run(V)

print "sess2"
print "sess1.run(V): %d"%sess2.run(V)
print "sess1.run(W.assign_add(30)): %d"%sess2.run(W.assign_add(30))
print "sess1.run(V): %d"%sess2.run(V)
print "sess2.run(W.assign_sub(5)): %d"%sess2.run(W.assign_sub(5))
print "sess1.run(V): %d"%sess2.run(V)

sess1.close()
sess2.close()
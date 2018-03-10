import tensorflow as tf
node_a = tf.constant(5.9)
node_b = tf.constant(6.0)

node = (node_a * node_b)//5

se = tf.Session()

print(se.run(node))

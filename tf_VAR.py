import tensorflow as tf


m = tf.Variable([-2.0],tf.float32)
c = tf.Variable([2.1],tf.float32)
x = tf.placeholder(tf.float32)

Y = m*x + c

y = tf.placeholder(tf.float32)


sq_diff = tf.square(Y-y)
loss = tf.reduce_sum(sq_diff)


st = tf.global_variables_initializer()

se = tf.Session()

se.run(st)

print(se.run(loss,{x:[1,2,3,4,5,6],y:[-1,-2,-3,-6,-7,-10]}))

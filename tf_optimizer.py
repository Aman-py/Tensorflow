import tensorflow as tf


m = tf.Variable([-2.0],tf.float32)
c = tf.Variable([2.1],tf.float32)
x = tf.placeholder(tf.float32)

Y = m*x + c

y = tf.placeholder(tf.float32)


sq_diff = tf.square(Y-y)
loss = tf.reduce_sum(sq_diff)

opt = tf.train.GradientDescentOptimizer(0.01)
trn = opt.minimize(loss)

st = tf.global_variables_initializer()

se = tf.Session()

se.run(st)

for i in range(100):
    se.run(trn,{x:[1,2,3,4,5,6],y:[-1,-2,-3,-6,-7,-10]})
    

print(se.run([m,c]))

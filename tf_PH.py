import tensorflow as tf
import numpy as np

node_a = tf.placeholder(tf.int64)
node_b = tf.placeholder(tf.int64)

node = (node_a * node_b) + (node_a) + (node_b)

se = tf.Session()

show = se.run(node,{node_a:[4,6],node_b:[198,182]})

print(show)

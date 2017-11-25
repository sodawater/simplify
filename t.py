import tensorflow as tf
import numpy as np
u = tf.placeholder(tf.int32,shape=[None])
k = tf.shape(u)[0]
print(k)
with tf.Session() as s:
    s.run(tf.initialize_all_variables())
    print(s.run(k,feed_dict={u:[1,2,3]}))

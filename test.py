import tensorflow as tf
import numpy as np

x = tf.placeholder(dtype=tf.int32, shape=[2,4])
a = tf.constant([0] * 2,dtype=tf.int32)
a = tf.reshape(a,[2,1])
y = tf.concat([a,x],1)
sess = tf.Session()
print(list(sess.run(y,feed_dict={x:[[1,2,3,4],[5,2,3,4]]})))



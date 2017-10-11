import tensorflow as tf
import numpy as np
buckets = [(10, 5), (15, 10), (25, 20), (50, 40)]
encoder_inputs = []
for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
    encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                              name="encoder{0}".format(i)))
print(encoder_inputs)
#a = tf.Variable(tf.random_uniform([5],-1.0,1.0))
#sess = tf.Session()
print([[1] * 3] * 2)
for i in range(40):
    a = np.random.rand(2,3) * 2 - [[1] * 3] * 2
    print(a)
#    print(sess.run(a))

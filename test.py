import tensorflow as tf
import numpy as np

class A():
    def __init__(self):
        with tf.variable_scope("test") as test_scope:
            self.a = tf.placeholder(dtype=tf.int32)
            self.b = tf.placeholder(dtype=tf.int32)
            self.c = self.a + self.b

a1 = A()
a2 = A()
a3 = A()
sess = tf.Session()
print(sess.run([a1.c],feed_dict={a1.a:1,a1.b:2}))
print(a3.a)
print(a1.a)


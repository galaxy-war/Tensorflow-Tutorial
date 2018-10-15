"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
"""
import tensorflow as tf

x1 = tf.placeholder(dtype=tf.float32, shape=None)
y1 = tf.placeholder(dtype=tf.float32, shape=None)
z1 = x1 + y1

x2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1, 2])
z2 = tf.matmul(x2, y2)

with tf.Session() as sess:
    # when only one operation to run
    # z1_value = sess.run(z1, feed_dict={x1: 1, y1: 2})
    z1_value = sess.run(z1, feed_dict={x1: 1, y1: 2})
    print("#1 z1_value=",  z1_value)
    # when run multiple operations
    z1_value, z2_value = sess.run([z1, z2],       # run them together
        feed_dict={
            x1: 1, y1: 2,
            x2: [[2], [2]], y2: [[3, 3]]
        })
    print("#2 z1_value=", z1_value)
    print("#2 z2_value=", z2_value)
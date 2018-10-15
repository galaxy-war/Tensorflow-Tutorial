"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

with tf.name_scope("my_vars"):
 var = tf.Variable(0)    # our first variable in the "global_variable" set
print("# Pre 1:", var)
add_operation = tf.add(var, 1)
update_operation = tf.assign(var, add_operation)
print("# Pre 2:", var)

with tf.Session() as sess:
    # once define variables, you have to initialize them by doing this
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        sess.run(update_operation)
        print(sess.run(var))
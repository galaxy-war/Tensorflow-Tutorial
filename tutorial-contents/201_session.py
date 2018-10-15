"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
"""
import tensorflow as tf

m1 = tf.constant([[2, 2]]) # Tensor("Const:0", shape=(1, 2), dtype=int32)
m2 = tf.constant([[3],
                  [3]])    # Tensor("Const_1:0", shape=(2, 1), dtype=int32)
dot_operation = tf.matmul(m1, m2)

print(dot_operation)  # wrong! no result
print(m1)
print(m2)

# method1 use session
sess = tf.Session()

result = sess.run(dot_operation)
# print(result)
sess.close()

# method2 use session
with tf.Session() as sess:
    result_ = sess.run(dot_operation)
    #print(result_)
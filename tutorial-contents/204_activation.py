"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置TF的CPU和线程资源 
tfConfig = tf.ConfigProto(device_count={"CPU": 1}, # limit to num_cpu_core CPU usage
                inter_op_parallelism_threads = 1,
                intra_op_parallelism_threads = 2,
                log_device_placement=True
              )

# fake data
x = np.linspace(-5, 5, 200)     # x data, shape=(100, 1)

# following are popular activation functions
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)
y_softmax = tf.nn.softmax(x)  # softmax is a special kind of activation function, it is about probability

with tf.Session(config = tfConfig) as sess:
 y_relu, y_sigmoid, y_tanh, y_softplus, y_softmax = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus, y_softmax])

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(321)
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.subplot(322)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.subplot(323)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.subplot(324)
plt.plot(x, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.subplot(325)
plt.plot(x, y_softmax, c='red', label='softmax')
# plt.ylim((-0.1, 6))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.show()
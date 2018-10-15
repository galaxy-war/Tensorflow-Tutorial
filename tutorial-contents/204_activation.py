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
# （1，x）分布函数
y_relu = tf.nn.relu(x)

# 输出归一化函数
y_sigmoid = tf.nn.sigmoid(x)

# 反正切函数，可以将函数的输出限定在（0，1）
y_tanh = tf.nn.tanh(x)

# 和 ReLU，sigmoid 类似，使函数输出变得平滑，连续可导
y_softplus = tf.nn.softplus(x)

# 在求具体函数值比较困难的时候，转为求该函数的最大可能概率，进而求得最可能相关参数及函数值
y_softmax = tf.nn.softmax(x*0.5)  # softmax is a special kind of activation function, it is about probability

# y = ln x
y_ln = tf.log(x)
y_exp = tf.exp(x)

# y 为 标准正态分布的随机数, 即 服从均值为0，标准方差为1的标准正态分布
y_stand_normalize = tf.random_normal(x.shape)

with tf.Session(config = tfConfig) as sess:
 y_relu, y_sigmoid, y_tanh, y_softplus, y_softmax, y_ln, y_exp, y_stand_normalize = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus, y_softmax, y_ln, y_exp, y_stand_normalize])

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(331)
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.subplot(332)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.subplot(333)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.subplot(334)
plt.plot(x, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.subplot(335)
plt.plot(x, y_softmax, c='red', label='softmax')
# 取消Y轴限制
# plt.ylim((-0.1, 6))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.subplot(336)
# the couple f'(x)
plt.plot(x, y_exp, c='green', label='exp')
plt.plot(x, y_ln, c='red', label='ln')
# 取消Y轴限制
plt.ylim((-10, 10))
plt.xlim((-10, 10))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.subplot(337)
# the couple f'(x)
plt.plot(x, y_stand_normalize, c='green', label='random_normal')
# 取消Y轴限制
#plt.ylim((-10, 10))
#plt.xlim((-10, 10))
plt.axvline()
plt.axhline()
plt.legend(loc='best')

plt.show()
"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# fake data
# np.newaxis 增加一个维度，在这里 原来只是一维线性
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
print("# def x=", x)
noise = np.random.normal(0, 0.1, size=x.shape)
print("# def noise=", noise)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

# plot data
plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 50, tf.nn.relu)          # hidden layer
l2 = tf.layers.dense(l1, 30, tf.nn.relu)          # hidden layer
l3 = tf.layers.dense(l2, 10, tf.nn.relu)          # hidden layer
output = tf.layers.dense(l3, 1)                     # output layer

loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
#train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
# train_op = tf.train.AdadeltaOptimizer().minimize(loss)
# train_op = tf.train.AdamOptimizer().minimize(loss)
# train_op = tf.train.AdagradOptimizer().minimize(loss)
#
# global_step = tf.Variable(0, name='global_step', trainable=False)
# train_op = optimizer.minimize(loss, global_step=global_step)
# train_op = tf.train.AdagradDAOptimizer(learning_rate=0.2, global_step=output.shape).minimize(loss)
train_op = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(loss)
print(train_op)


"""
sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph


plt.ion()   # something about plotting

for step in range(100):
    # train and net output
    _, l, pred = sess.run([train_op, loss, output], feed_dict={tf_x: x, tf_y: y})

    if step % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _, l, pred = sess.run([train_op, loss, output], feed_dict={tf_x: x, tf_y: y})
    print("# the first l is:", l)
    step = 1
    plt.ion()
    while l > 1e-3:
        _, l, pred = sess.run([train_op, loss, output], feed_dict={tf_x: x, tf_y: y})
        print("# The Step: %d and loss is:%-4f" % (step, l))
        step = step + 1
        if step % 1000 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x, y)
            plt.plot(x, pred, 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
    plt.ioff()
    plt.show()
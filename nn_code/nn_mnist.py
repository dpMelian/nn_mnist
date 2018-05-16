import gzip
import cPickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y.astype(int), 10)
valid_y = one_hot(valid_y.astype(int), 10)
test_y = one_hot(test_y.astype(int), 10)


# ---------------- Visualizing some element of the MNIST dataset --------------

# import matplotlib.cm as cm
# import matplotlib.pyplot as plt

# plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print train_y[57]


# TODO: the neural net!!

x = tf.placeholder("float", [None, 784])    # 784 = 28x28px
y_ = tf.placeholder("float", [None, 10])

n_neurons = 150

W1 = tf.Variable(np.float32(np.random.rand(784, n_neurons)) * 0.1)    # weight; second parameter is the number of neurons
b1 = tf.Variable(np.float32(np.random.rand(n_neurons)) * 0.1)         # bias

W2 = tf.Variable(np.float32(np.random.rand(n_neurons, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

# loss = tf.reduce_sum(tf.square(y_ - y))
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))     # cross entropy

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20

epoch = 0
loss_val = 1
array_valid_loss = []
array_loss = []
current_val_error = 1000
previous_val_error = 0


def percentage(whole):
    return (2 * whole) / 100.0


while abs(current_val_error - previous_val_error) > percentage(previous_val_error):   # current val - previous val < 2%
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        # using two parameters to print loss value
        _, loss_val = sess.run([train, loss], feed_dict={x: batch_xs, y_: batch_ys})

    epoch += 1
    previous_val_error = current_val_error
    current_val_error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    array_valid_loss.append(current_val_error)
    array_loss.append(loss_val)

    print "Validation:"
    print "Epoch #", epoch
    print "loss value:", loss_val
    print "Validation error:", current_val_error
    print "****************************"


print "*****TEST*****"

success_num = 0
failure_num = 0

result = sess.run(y, feed_dict={x: test_x})

# zipped list

for i, j in zip(test_y, result):
    # checks whether the predicted (rounded) result is correct from test set
    if np.array_equal(i, np.round(j, 0)):
        success_num += 1
    else:
        failure_num += 1

print "Test results:", success_num, "successes and", failure_num, "failures"
print "Accuracy:", ((success_num*100)/10000), "%"

error_training, = plt.plot(array_loss, label='Error training')
error_validacion, = plt.plot(array_valid_loss, label='Error validacion', linestyle='--')

plt.legend(handles=[error_training, error_validacion])
plt.xlabel("epoch")
plt.ylabel("error")
plt.show()


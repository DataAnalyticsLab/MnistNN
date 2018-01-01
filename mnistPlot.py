import tensorflow as tf
import numpy as np
import math
from matplotlib import pyplot as plt
from collections import Counter

def plotGraph(x,y,title,xAxisTitle,yAxisTitle):
    plt.scatter(x,y)
    plt.title(title)
    plt.ylabel(yAxisTitle)
    plt.xlabel(xAxisTitle)
    plt.show()


def weightCount(x):
    a = x.ravel()
   ## myRoundedList=a
    myRoundedList = [ round(elem,4) for elem in a]
    d = Counter(myRoundedList)
    return d








from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

randomSession = tf.Session()

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100
hidden_layer_size = 500
dropout_rate = 1.0

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, hidden_layer_size], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([hidden_layer_size]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([hidden_layer_size, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

#
dropout_variable = tf.placeholder(tf.float32)


# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)
hidden_out_after_drop = tf.nn.dropout(hidden_out, dropout_variable)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
y_after_dropout = tf.nn.dropout(y_, dropout_variable)
y_=y_after_dropout

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)+ (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.AdamOptimizer().minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



weight1=None
weight2=None

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _,c = sess.run([optimiser, cross_entropy],feed_dict={x: batch_x, y: batch_y, dropout_variable:dropout_rate})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels,dropout_variable:1.0}))
   weight1=sess.run(W1)
   weight2=sess.run(W2)


weight1Counter=weightCount(weight1)
weight2Counter=weightCount(weight2)



plotGraph(weight1Counter.keys(),weight1Counter.values(),"Weight Distribution of Layer 1","Value of Weights","Number of Edges")
plotGraph(weight2Counter.keys(),weight2Counter.values(),"Weight Distribution of Layer 2","Value of Weights","Number of Edges")


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
from matplotlib import pyplot as plt
from collections import Counter

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def plotGraph(x,y,title,xAxisTitle,yAxisTitle):
    plt.scatter(x,y)
    plt.title(title)
    plt.ylabel(yAxisTitle)
    plt.xlabel(xAxisTitle)
    plt.show()


def weightCount(x):
    a = x.ravel()
   ## myRoundedList=a
    myRoundedList = [ round(elem,2) for elem in a]
    d = Counter(myRoundedList)
    return d


numberOfNodes = 500

n_nodes_hl1 = 2000
n_nodes_hl2 = 2000
n_nodes_hl3 = 2000

n_classes = 10
batch_size = 100
keep_prob = 1.0

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')
dropout_variable = tf.placeholder('float')



hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

l1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
l1 = tf.nn.relu(l1)
l1d = l1


l2 = tf.add(tf.matmul(l1d, hidden_2_layer['weights']), hidden_2_layer['biases'])
l2 = tf.nn.relu(l2)
l2d = tf.nn.dropout(l2, dropout_variable)

l3 = tf.add(tf.matmul(l2d, hidden_3_layer['weights']), hidden_3_layer['biases'])
l3 = tf.nn.relu(l3)
l3d = tf.nn.dropout(l3, dropout_variable)

output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']


prediction = output
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

hm_epochs = 10
with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
    sess.run(tf.global_variables_initializer())

    for epoch in range(hm_epochs):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y,dropout_variable:keep_prob})
            epoch_loss += c

        print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels,dropout_variable:1.0}))

    layer1 = sess.run(hidden_1_layer['weights'])
    weight1Counter = weightCount(layer1)

    layer2 = sess.run(hidden_2_layer['weights'])
    weight2Counter = weightCount(layer2)

    layer3 = sess.run(hidden_3_layer['weights'])
    weight3Counter = weightCount(layer3)
    plotGraph(weight1Counter.keys(), weight1Counter.values(), "Weight Distribution of Layer 1", "Value of Weights","Number of Edges")
    plotGraph(weight2Counter.keys(), weight2Counter.values(), "Weight Distribution of Layer 2", "Value of Weights","Number of Edges")
    plotGraph(weight3Counter.keys(),weight3Counter.values(),"Weight Distribution of Layer 3","Value of Weights","Number of Edges")

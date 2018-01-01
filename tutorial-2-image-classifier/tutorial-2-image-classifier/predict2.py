import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import matplotlib.pyplot as plt


weights = []
biases = []



def get_weights(layer):
    len_of_layer = len(layer)
    for i in range(len_of_layer):
        # print(layer[i])
        temp_sub_layer = layer[i]
        len_of_sub_layer = len(layer[i])
        for j in range(len_of_sub_layer):
            # print(temp_sub_layer[j] , " ", end='', flush=True)
            weights.append(temp_sub_layer[j])
            # print()
    # print(layer)
    # print(len_of_layer)
    # print(layer[0])
    #print("weights ",len(weights))

def print_layer(layer):
    print(layer)

def length_of_array(layer):
    len_of_layer = len(layer)
    return_len = 0
    #print(len_of_layer)
    for i in range(len_of_layer):
        # print(layer[i])
        temp_sub_layer = layer[i]
        len_of_sub_layer = len(layer[i])
        #print(len_of_sub_layer)
        return_len+= len_of_sub_layer
    return return_len

def length_of_conv(layer):
    len_conv = 0
    first_len = len(layer)
    for i in range(first_len):
        sec_len = len(layer[i])
        # print(sec_len)
        for j in range(sec_len):
            third_len = len(layer[i][j])
            # print(third_len, " ", end='', flush=True)
            for k in range(third_len):
                fourth_len = len(layer[i][j][k])
                # print(fourth_len)
                for l in range(fourth_len):
                    # print(layer[i][j][k][l])
                    # weights.append(layer[i][j][k][l])
                    len_conv += 1
                    # print()
    return len_conv

def get_weights_conv(layer):

    first_len = len(layer)
    for i in range(first_len):
        sec_len = len(layer[i])
        # print(sec_len)
        for j in range(sec_len):
            third_len = len(layer[i][j])
            # print(third_len, " ", end='', flush=True)
            for k in range(third_len):
                fourth_len = len(layer[i][j][k])
                # print(fourth_len)
                for l in range(fourth_len):
                    # print(layer[i][j][k][l])
                    weights.append(layer[i][j][k][l])

                    # print()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('dogs-cats-model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))


    layer  = sess.run('fcW1:0')
    get_weights(layer)
    print(len(weights))

    layer = sess.run('fcW2:0')
    #print_layer(layer)
    #print(length_of_array(layer))
    get_weights(layer)
    print(len(weights))

    layer = sess.run('convW1:0')
    get_weights_conv(layer)
    print(len(weights))
    #print_layer(layer)

    layer = sess.run('convW2:0')
    get_weights_conv(layer)
    print(len(weights))

    #print(length_of_conv(layer))
    #print_layer(layer)

    layer = sess.run('convW3:0')
    get_weights_conv(layer)
    print(len(weights))
    #print(length_of_conv(layer))
    #print_layer(layer)


    #print(weights)
    thefile = open('weight.txt', 'w')

    for item in weights:
        thefile.write("%s\n" % item)



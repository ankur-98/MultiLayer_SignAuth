from model_funcs import * 
from tf_utils import *

import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import sys

def model(X_train, Y_train, X_val, Y_val, layer_dims, activations, learning_rate = 0.001, num_epochs = 1500, minibatch_size = 32, print_cost = True):

    ops.reset_default_graph()
    
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    
    costs = []
    
    X, Y = create_placeholders(n_x, n_y)
    
    parameters = initialize_parameters(layer_dims)
    
    Z = forward_propagation(X, parameters, activations)
    
    cost = cost_func(Z, Y)

    optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    ##############################################

    # device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
    # shape = (int(sys.argv[2]), int(sys.argv[2]))
    # if device_name == "gpu":
    #     device_name = "/gpu:0"
    # else:
    #     device_name = "/cpu:0"
    #
    # print("Shape:", shape, "Device:", device_name)
    with tf.device("/gpu:0"):
    ###############################################


     with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in tqdm.tqdm(range(num_epochs)):
#
           epoch_cost = 0
           num_minibatches = int(m/minibatch_size)
           minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

           for minibatch in minibatches:
               (minibatch_X, minibatch_Y) = minibatch
               _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
               epoch_cost += minibatch_cost/num_minibatches

           if print_cost == True and epoch % 1 == 0:
               print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
           if print_cost == True and epoch % 5 == 0:
               costs.append(epoch_cost)
        
            # _, c = sess.run([optimizer,cost], feed_dict={X:X_train, Y:Y_train})
            # epoch_cost += c
            # if print_cost == True and epoch % 100 == 0:
            #     print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            # if print_cost == True and epoch % 5 == 0:
            #     costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens) ')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()
        
        parameters = sess.run(parameters)
        
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
        
        # Calculate accuracy on the val set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_val, Y: Y_val}))



        return parameters

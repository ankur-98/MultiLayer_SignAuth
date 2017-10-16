import tensorflow as tf
import numpy as np

def create_placeholders(n_x,n_y):
    
    """
    n_x : size of input image
    n_y : no. of classes
    X   : placeholder for input
    Y   : plceholder for output
    
    """
    
    X = tf.placeholder(tf.float32,[n_x,None],name = "X")
    Y = tf.placeholder(tf.float32,[n_y,None],name = "Y")
    
    return X,Y


def initialize_parameters(layer_dims):
    
    """
    layer_dims : [Lin,L1,L2...,Lout] Defines the architecture of the model
    parameter  : contains the weights and bias of the model
    
    """
    
    L = len(layer_dims)
    parameters = {}
    
    for l in range(1,L):
        parameters['W' + str(l)] = tf.get_variable('W' + str(l), [layer_dims[l],layer_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layer_dims[l],1], initializer = tf.ones_initializer())
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters


def activation_func(activation,Z):
    
    if(activation == "relu"):
        A = tf.nn.relu(Z)
    elif(activation == "tanh"):
        A = tf.nn.tanh(Z)
    elif(activation == "sigmoid"):
        A = tf.nn.sigmoid(Z)
    
    return A
    

def forward_propagation(X, parameters, activation):
    
    """
    returns the Lout layer logit 
    
    """
    
    A_prev = X
    # A_prev = tf.transpose(X)
    L = int(len(parameters)/2)
    Z = {}
    
    for l in range(1,L):
        Z["Z" + str(l)] = tf.add(tf.matmul(parameters["W" + str(l)],A_prev),parameters["b" + str(l)])
        A_prev = activation_func(activation[l-1],Z["Z" + str(l)])
        
    Z["Z" + str(L)] = tf.add(tf.matmul(parameters["W" + str(L)],A_prev),parameters["b" + str(L)])
    
    return Z["Z" + str(L)]    


def cost_func(Z,Y):
    
    """
    logits  : Lout layer output with activation function as linear activation function - Z
    labels  : Expected Output - Y
    cost    : The mean difference between target and actual output
    
    """
    
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z,labels=Y))
    
    return cost


#Packages
import numpy as np

#function to get size of datasets
def get_m(train,test):
    m_train = train.shape[0]     #Number of training examples
    m_test = test.shape[0]       #Number of testing examples
    num_px = train.shape[1]      #Height/Width of each image
    #Each image is of size: (300, 300, 3)
    return m_train,m_test,num_px

#function to return flattened image dataset
def get_flatten(train,test):
    train_flatten = train.reshape(train.shape[0],-1).T
    test_flatten = test.reshape(test.shape[0],-1).T
    return train_flatten,test_flatten

#function to standardize dataset
def standardize(train_f,test_f):
    train = train_f/255
    test = test_f/255
    return train,test
                                            
train_set_x_orig, test_set_x_orig, train_set_y, test_set_y = np.load('data_set.npy')
m_train,m_test,num_px = get_m(train_set_x_orig,test_set_x_orig)
train_set_x_flatten,test_set_x_flatten = get_flatten(train_set_x_orig,test_set_x_orig)

train_set_x,test_set_x = standardize(train_set_x_flatten,test_set_x_flatten)
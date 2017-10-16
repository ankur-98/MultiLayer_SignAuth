from proc import *
from std_dataset import *
from arrange import *
from tensorflow.python.framework import ops

import numpy as np

ops.reset_default_graph()

# n = eval(input("\n\nEnter number of iterations: ")) #1000
# l = eval(input("Enter learning rate: ")) #0.001
# p = bool(eval(input("Enter 1 to display cost else 0: "))) #False

# d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = n, learning_rate = l, print_cost = p)

n = 600
l = 0.0002
b = 239

d = model(train_set_x, train_set_y, test_set_x, test_set_y, [train_set_x.shape[0],30000,30000,10000,2000,2000,500,train_set_y.shape[0]], ["tanh","tanh","tanh","tanh","relu","relu"], l, n, b, True)
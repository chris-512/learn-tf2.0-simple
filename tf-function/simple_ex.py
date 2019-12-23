import tensorflow as tf 

# https://www.tensorflow.org/guide/function
# RFC: https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md

"""The tf.function decorator

When you annotate a function with `tf.function`, you can still call it like any other function.
But it will be compiled into a graph, which means you get the benefits of faster execution, running on 
GPU or TPU, or exporting to SavedModel.
"""

@tf.function 
def simple_nn_layer(x, y):
    return tf.nn.relu(tf.matmul(x, y))

x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))
print(simple_nn_layer(x, y))

print(simple_nn_layer)

"""
If your code uses multiple functions, you don't need to annotate them all
-- any functions called from an annotated function will also run in graph mode.
"""

def linear_layer(x):
    return 2 * x + 1

@tf.function
def deep_net(x):
    return tf.nn.relu(linear_layer(x))

print(deep_net(tf.constant((1, 2, 3))))

import timeit 
conv_layer = tf.keras.layers.Conv2D(100, 3)

@tf.function 
def conv_fn(image):
    return conv_layer(image)

image = tf.zeros([1, 200, 200, 100])
# warm up
conv_layer(image); conv_fn(image)
print('Eager conv: ', timeit.timeit(lambda: conv_layer(image), number=10))
print('Function conv: ', timeit.timeit(lambda: conv_fn(image), number=10))
print('Note how there\'s not much differene in performance for convolutions')

lstm_cell = tf.keras.layers.LSTMCell(10)

@tf.function
def lstm_fn(input, state):
  return lstm_cell(input, state)

input = tf.zeros([10, 10])
state = [tf.zeros([10, 10])] * 2
# warm up
lstm_cell(input, state); lstm_fn(input, state)
print("eager lstm:", timeit.timeit(lambda: lstm_cell(input, state), number=10))
print("function lstm:", timeit.timeit(lambda: lstm_fn(input, state), number=10))

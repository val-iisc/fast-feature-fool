'''
wrapper functions for tensorflow layers
'''
import tensorflow as tf

def conv_layer(bottom, weight, bias=None, s=1, padding='SAME', relu=True, group=1):
    if group==1:
        conv = tf.nn.conv2d(bottom, weight, [1, s, s, 1], padding=padding)
    else:
        input_split = tf.split(bottom, group, 3)
        weight_split = tf.split(weight, group, 3)
        conv_1 = tf.nn.conv2d(input_split[0], weight_split[0], [1, s, s, 1], padding=padding)
        conv_2 = tf.nn.conv2d(input_split[1], weight_split[1], [1, s, s, 1], padding=padding)
        conv = tf.concat([conv_1, conv_2], 3)
    if bias is None:
        if relu:
            return tf.nn.relu(conv)
        else:
            return conv
    else:
        bias = tf.nn.bias_add(conv, bias)
        if relu:
            return tf.nn.relu(bias)
        else:
            return bias

def max_pool(bottom, k=3, s=1, padding='SAME'):
     return tf.nn.max_pool(bottom, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)

def fully_connected(bottom, weight, bias):
    fc = tf.nn.bias_add(tf.matmul(bottom, weight), bias)
    return fc

def inception_block(bottom, name, weights, biases):
    block = {}
    with tf.name_scope(name+'1x1'):
        block['branch_1'] = conv_layer(bottom, weights['inception_'+name+'_1x1'], biases['inception_'+name+'_1x1'])

    with tf.name_scope(name+'3x3'):
        block['branch_2_r'] = conv_layer(bottom, weights['inception_'+name+'_3x3_reduce'], biases['inception_'+name+'_3x3_reduce'])
        block['branch_2'] = conv_layer(block['branch_2_r'], weights['inception_'+name+'_3x3'], biases['inception_'+name+'_3x3'])

    with tf.name_scope(name+'5x5'):
        block['branch_3_r'] = conv_layer(bottom, weights['inception_'+name+'_5x5_reduce'], biases['inception_'+name+'_5x5_reduce'])
        block['branch_3'] = conv_layer(block['branch_3_r'], weights['inception_'+name+'_5x5'], biases['inception_'+name+'_5x5'])

    with tf.name_scope(name+'pool'):
        block['branch_4_p'] = max_pool(bottom)
        block['branch_4'] = conv_layer(block['branch_4_p'], weights['inception_'+name+'_pool_proj'], biases['inception_'+name+'_pool_proj'])

    block['concat'] = tf.concat(axis=3, values=[block['branch_1'], block['branch_2'], block['branch_3'], block['branch_4']])

    return block

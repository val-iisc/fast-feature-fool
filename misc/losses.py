#definition of different custom loss functions
import tensorflow as tf

#to maximise activations
def activations(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if j == 'concat':
                        loss += tf.log(tf.reduce_mean(tf.abs(network[i][j]))) #total blob activations
            except:
                loss += tf.log(tf.reduce_mean(tf.abs(network[i]))) #total blob activations
    return loss

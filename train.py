'''
This code generated a universal adversarial network for a given network
'''

from nets.vgg_f import vggf
from nets.caffenet import caffenet
from nets.vgg_16 import vgg16
from nets.vgg_19 import vgg19
from nets.googlenet import googlenet
from urllib import urlretrieve
from misc.losses import *
import tensorflow as tf
import numpy as np
import argparse
import time
import os

def validate_arguments(args):
    nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet']
    if not(args.network in nets):
        print ('invalid network')
        exit (-1)

def choose_net(network):
    MAP = {
        'vggf'     : vggf,
        'caffenet' : caffenet,
        'vgg16'    : vgg16,
        'vgg19'    : vgg19,
        'googlenet': googlenet
    }
    if network == 'caffenet':
        size = 227
    else:
        size = 224
    # placeholder to pass image
    input_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='input_image')
    # initializing adversarial image
    adv_image = tf.Variable(tf.random_uniform([1,size,size,3],minval=-10,maxval=10), name='noise_image', dtype='float32')
    # clipping for imperceptibility constraint
    adv_image = tf.clip_by_value(adv_image,-10,10)
    input_batch = tf.concat([input_image, tf.add(input_image,adv_image)], 0)

    return MAP[network](adv_image), MAP[network](input_batch), input_image, adv_image

def not_optim_layers(network):
# Layers at which are excluded from optimization
    if network == 'vggf':
        return ['norm1', 'pool1', 'norm2', 'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
    if network == 'caffenet':
        return ['norm1', 'pool1', 'norm2', 'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
    elif network == 'vgg16':
        return ['norm1', 'pool1', 'norm2', 'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
    elif network == 'vgg19':
        return ['norm1', 'pool1', 'norm2', 'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
    else:
        return ['pool1_3x3_s2', 'pool1_norm1', 'conv2_norm2', 'pool2_3x3_s2', 'pool3_3x3_s2', 'pool4_3x3_s2', 'pool5_7x7_s1', 'loss3_classifier', 'prob']

def train(adv_net, net, in_im, ad_im, opt_layers, net_name):
    losses = {'activation': -activations(adv_net, opt_layers)}
    cost = losses['activation']
    tvars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    grads = optimizer.compute_gradients(cost,tvars)
    update = optimizer.apply_gradients(grads)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    if net_name == 'caffenet':
        data_path = os.path.join('data', 'preprocessed_images_227.npy')
        download_link = "https://www.dropbox.com/s/0v8pnnumbytb378/preprocessed_images_227.npy?raw=1"
    else:
        data_path = os.path.join('data', 'preprocessed_images_224.npy')
        download_link = "https://www.dropbox.com/s/k4tamvdjndyvgws/preprocessed_images_224.npy?raw=1"
    if os.path.isfile(data_path) == 0:
        print('Downloading validation data (1K images)...')
        urlretrieve (download_link, data_path)
    imgs = np.load(data_path)

    fool_rate = 0 # current fooling rate
    check = 200 # iterations to check fooling rate
    stopping = 0 # early stopping condition
    t_s = time.time()
    print "Starting {:} training...".format(net_name)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            sess.run(update)
            if i%300 == 0 and i!=0:
                assign_op = tvars[0].assign(tf.divide(tvars[0],2.0)) #for scaling adv_input to lesser range
                sess.run(assign_op)
            if i == check:
                check += 300
                temp = 0
                for j in range(1000):
                    softmax_scores = sess.run(net['prob'], feed_dict={in_im: imgs[j:j+1]})
                    if np.argmax(softmax_scores[0])!=np.argmax(softmax_scores[1]):
                        temp += 1
                loss, im = sess.run([cost, ad_im])
                print('iter: {:5d}\tloss: {:14.8f}\tfooling_rate: {:.2f}'.format(i, loss, temp/10.0))
                if temp > fool_rate:
                    stopping = 0
                    fool_rate = temp
                    np.save('perturbations/perturbation_'+net_name,im)
                else:
                    stopping += 1
                if stopping == 5:
                    print 'Optimization finished!'
                    break
        print 'Time taken: {:.2f}s'.format(time.time()-t_s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='googlenet', help='The network eg. googlenet')
    args = parser.parse_args()
    validate_arguments(args)
    adv_net, net, inp_im, ad_im  = choose_net(args.network)
    opt_layers = not_optim_layers(args.network)
    train(adv_net, net, inp_im, ad_im, opt_layers, args.network)

if __name__ == '__main__':
    main()

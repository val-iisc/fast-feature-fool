'''
The code takes in the image list and generated perturbation and calculates the
fooling rate and classification accuracy on the ILSVRC validation set (50K images)
'''

from nets.vgg_f import vggf
from nets.caffenet import caffenet
from nets.vgg_16 import vgg16
from nets.vgg_19 import vgg19
from nets.googlenet import googlenet
from misc.utils import img_preprocess, upsample, downsample
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import argparse

def validate_arguments(args):
    nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet']
    if not(args.network in nets):
        print ('invalid network')
        exit (-1)
    if args.adv_im is None:
        print ('no path to perturbation')
        exit (-1)
    if args.img_list is None or args.gt_labels is None:
        print ('provide image list and labels')
        exit (-1)

def choose_net(network, adv_image):
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
    # loading the perturbation
    pert_load = np.load(adv_image)
    # preprocessing if necessary
    if (pert_load.shape[1] == 224 and size == 227):
	pert_load = upsample(np.squeeze(pert_load))
    elif (pert_load.shape[1] == 227 and size == 224):
	pert_load = downsample(np.squeeze(pert_load))
    elif (pert_load.shape[1] not in [224, 227]):
	raise Exception("Invalid size of input perturbation")
    adv_image = tf.constant(pert_load, dtype='float32')
    # placeholder to pass image
    input_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='input_image')
    input_batch = tf.concat([input_image, tf.add(input_image,adv_image)], 0)

    return MAP[network](input_batch), input_image

def classify(net, in_im, net_name, im_list, gt_labels):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    imgs = open(im_list).readlines()
    gt_labels = open(gt_labels).readlines()
    fool_rate = 0
    top_1 = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i,name in enumerate(imgs):
            if net_name == 'caffenet':
                im = img_preprocess(name.strip(), size=227)
            else:
                im = img_preprocess(name.strip())
            softmax_scores = sess.run(net['prob'], feed_dict={in_im: im})
            if i!=0 and i%1000 == 0:
                print 'iter: {:5d}\ttop-1: {:04.2f}\tfooling-rate: {:04.2f}'.format(i, (top_1/float(i))*100, (fool_rate)/float(i)*100)
            if np.argmax(softmax_scores[0]) == int(gt_labels[i].strip()):
                top_1 += 1
            if np.argmax(softmax_scores[0]) != np.argmax(softmax_scores[1]):
                fool_rate += 1
    print 'Top-1 Accuracy = {:.2f}'.format(top_1/500.0)
    print 'Fooling Rate = {:.2f}'.format(fool_rate/500.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='googlenet', help='The network eg. googlenet')
    parser.add_argument('--adv_im', help='Path to the perturbation image')
    parser.add_argument('--img_list',  help='Path to the validation image list')
    parser.add_argument('--gt_labels', help='Path to the ground truth validation labels')
    args = parser.parse_args()
    validate_arguments(args)
    net, inp_im  = choose_net(args.network, args.adv_im)
    classify(net, inp_im, args.network, args.img_list, args.gt_labels)

if __name__ == '__main__':
    main()

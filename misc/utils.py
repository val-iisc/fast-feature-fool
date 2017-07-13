import numpy as np
from skimage.io import imread
from skimage.transform import resize

#utilities
def img_preprocess(img_path, size=224):
    mean = [103.939, 116.779, 123.68]
    img = imread(img_path)
    img = resize(img, (size, size))*255.0
    if len(img.shape) == 2:
        img = np.dstack([img,img,img])
    img[:,:,0] -= mean[2]
    img[:,:,1] -= mean[1]
    img[:,:,2] -= mean[0]
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = np.reshape(img,[1,size,size,3])
    return img


def downsample(inp):
        return np.reshape(inp[1:-2,1:-2,:], [1,224,224,3])

def upsample(inp):
        out = np.zeros([227,227,3])
        out[1:-2,1:-2,:] = inp
        out[0,1:-2,:] = inp[0,:,:]
        out[-2,1:-2,:] = inp[-1,:,:]
        out[-1,1:-2,:] = inp[-1,:,:]
        out[:,0,:] = out[:,1,:]
        out[:,-2,:] = out[:,-3,:]
        out[:,-1,:] = out[:,-3,:]
        return np.reshape(out,[1,227,227,3])


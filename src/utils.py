from urllib import request
import skimage.transform
import numpy as np
import cv2 
import os
from sklearn import utils
from sklearn.cross_validation import train_test_split

MEAN_VALUES = np.array([103.939, 116.779, 123.68]).reshape((3,1,1))

def download_file(url,fname = None):
    '''
        http://stackoverflow.com/a/22776
    '''
    if fname == None:
        fname = os.path.join('..','models',url.split('/')[-1])
    conn = request.urlopen(url)
    with open(fname, 'ab+') as f:
        fsize = int(conn.headers['content-length'])
        print ("Downloading: %s Bytes: %s" % (fname, fsize))
        fsize_dl = 0
        block_sz = 8192*1024*2
        while True:
            buff = conn.read(block_sz)
            if not buff:
                break
            fsize_dl += len(buff)
            f.write(buff)
            status = r"%10d  [%3.2f%%]" % (fsize_dl, fsize_dl * 100. / fsize)
            status = status + chr(8)*(len(status)+1)
            print (status)

def process_image_vgg(im,IMAGE_W,IMAGE_H):
    
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    im = cv2.resize(im,(IMAGE_W,IMAGE_H))
    rawim = np.copy(im).astype('uint8')
    h, w, _ = im.shape
    im = im.transpose(2,0,1).reshape(3, h, w)
    im = im - MEAN_VALUES
    return rawim, np.float32(im[np.newaxis])

    
def imread(path_to_im):
    return cv2.imread(path_to_im)
       
def train_test_split(X,y,random_state = 42):
    return train_test_split( X, y, test_size=0.1,random_state = random_state)
    
def transpose(im):
    if len(im.shape) == 2:
        H,W = im.shape
        C = 1
    else:
        H,W,C = im.shape
    im = np.asarray(im, dtype = np.float32) / 255.0
    if len(im.shape) == 2:
        features = im.reshape(C,H,W)
    else:
        features = im.transpose(2,0,1).reshape(C, H, W)
        return features 

def resize(im,shape = (64, 32)):
    try:
        return cv2.resize(im, shape)
    except:
        print('Bad Data')
        return None
    
def shuffle(X,y):
    return utils.shuffle(X,y,random_state=42)

def imgrayscale(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
def deprocess_image_vgg(im):
    im = np.copy(im[0])
    im += MEAN_VALUES
    im = im[::-1]
    im = np.swapaxes(np.swapaxes(im, 0, 1), 1, 2)
    im = np.clip(im, 0, 255).astype('uint8')
    return im
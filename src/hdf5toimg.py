import numpy as np
import h5py
import imageio
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
from time import time


if __name__=="__main__":
    with h5py.File('./msh/sphere/sphere.hdf5', 'r') as f:
        dset = f['moxel']
        img = np.array(dset[:,:,0])*255
        t0 = time()
        for k in np.arange(1002,1052):
            fname = './msh/sphere/sphere{:04d}.png'.format(k)
            print(fname)
            img = np.array(dset[:,:,k])*255
            cv2.imwrite(fname, img)
        print(time()-t0)

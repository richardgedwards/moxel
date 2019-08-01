import numpy as np
import h5py
import imageio

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

if __name__=="__main__":
    with h5py.File('cavity.hdf5', 'r') as f:
        dset = f['mesh']
        for k in np.arange(165):
            fname = './cavity/cavity{:04d}.png'.format(k)
            print(fname)
            img = np.array(dset[:,:,k])*255
            imageio.imwrite(fname, img.T)

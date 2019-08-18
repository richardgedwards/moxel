import numpy as np
import h5py
import imageio
from PhotonFile import PhotonFile

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

if __name__=="__main__":
    with h5py.File('./msh/cavity.hdf5', 'r') as f:
        dset = f['moxel']
        # for k in np.arange(175,180):
        #     fname = './msh/cavity_low/cavity{:04d}.png'.format(k)
        #     print(fname)
        #     img = np.array(dset[:,:,k])*255
        #     imageio.imwrite(fname, img.T)

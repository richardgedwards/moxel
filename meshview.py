import numpy as np
import matplotlib.pyplot as plt
import h5py
import imageio
import time

def previous_slice(ax):
    with h5py.File('sphere.hdf5', 'r') as f:
        ax.index += 1
        volume = f['mesh']
        ax.imshow(volume[:,:,ax.index].T)
        print(ax.index)


def next_slice(ax):
     with h5py.File('sphere.hdf5', 'r') as f:
        ax.index -= 1
        volume = f['mesh']
        ax.imshow(volume[:,:,ax.index].T)
        print(ax.index)
    

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


if __name__=="__main__":
    with h5py.File('sphere.hdf5', 'r') as f:
        volume = f['mesh']
        # remove_keymap_conflicts({'j', 'k'})
        # fig, ax = plt.subplots()
        # ax.index = 0
        # ax.imshow(volume[:,:,ax.index].T)
        # fig.canvas.mpl_connect('key_press_event', process_key)

        for k in np.arange(100):
            plt.imshow(volume[:,:,k])
            print(k)
            plt.pause(0.0001)
        plt.show()


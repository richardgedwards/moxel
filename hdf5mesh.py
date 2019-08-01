import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import decimate

import meshio
import h5py
# import imageio

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys


class MeshSlice:
    def __init__(self, infilename, outfilename=None, edgetype=0, R=Rotation([0,0,0,1]), t=np.array((0,0,0)), 
                 d=np.array([47.25, 47.25, 50])*1e-6, imageshape=(1440, 2560), zmax=155e-3, q=4):

        self.infilename = infilename
        self.outfilename = outfilename
        self.mesh = meshio.read(infilename)
        self.R = R
        self.t = t
        self.d = d
        self.imageshape = imageshape
        self.zmax = zmax
        self.filled = set()
        self.edges = set()
        self.q = 1  # anti-aliasing factor
        if edgetype == 0:
            # find set of mesh edges
            for cell in self.mesh.cells['tetra']:
                self.edges.add(frozenset((cell[0], cell[1])))
                self.edges.add(frozenset((cell[1], cell[2])))
                self.edges.add(frozenset((cell[2], cell[0])))
                self.edges.add(frozenset((cell[0], cell[3])))
                self.edges.add(frozenset((cell[1], cell[3])))
                self.edges.add(frozenset((cell[2], cell[3])))
        elif edgetype == 1:            
            # find set of outer edges
            for cell in self.mesh.cells['triangle']:
                self.edges.add(frozenset((cell[0], cell[1])))
                self.edges.add(frozenset((cell[1], cell[2])))
                self.edges.add(frozenset((cell[2], cell[0])))


    def __repr__(self):
        np.set_printoptions(precision=4)
        extents = self.extents
        imgextents = np.c_[self.l2i(extents[:,0]), self.l2i(extents[:,1])]
        repr = "MeshSlice Characteristics:\n"
        repr += 'No. Edges: {:}\n'.format(len(self.edges))
        repr += 'No. Vertices: {:}\n'.format(len(self.mesh.points))
        repr += 'shape: {:}\n'.format(self.shape)
        repr += 'resolution: {:}\n'.format(self.d)
        repr += 'spatial size: {:}\n'.format(self.dim)
        repr += 'mesh size: {:}\n'.format(extents[:,1]-extents[:,0])
        repr += 'mesh occupancy: {:}\n'.format((extents[:,1]-extents[:,0])/self.dim)
        repr += 'mesh extents: \n{:}\n'.format(extents)
        repr += 'image extents: \n{:}\n'.format(imgextents)
        repr += 'local coordinate location: {:}\n'.format(self.t)
        repr += 'orientation local basis vectors: \n{:}\n'.format(self.R.apply(np.identity(3)))
        return repr


    @property
    def shape(self):
        return np.asarray((self.imageshape[0], self.imageshape[1], int(np.round(self.zmax/self.d[2]))))


    @property
    def dim(self):
        return self.shape*self.d


    @property
    def extents(self):
        s = self.g2l(self.mesh.points)
        return np.c_[np.min(s,axis=0), np.max(s,axis=0)]


    @property
    def imgextents(self):
        return self.l2i(self.extents)


    # coordinate transformations
    def l2g(self, s):
        ''' transform from local to global coordinates '''
        return self.R.apply(s)+self.t


    def g2l(self, r):
        ''' transform from global to local coordinates '''
        return self.R.apply(r-self.t, inverse=True)        


    def l2i(self, s):
        ''' transform from local to image coordinates '''
        a = np.asarray(np.round(s/self.d*self.q-0.5), dtype=int)
        a[a<0] = 0
        b = a>=self.shape
        a[b] = self.shape[b]
        return a


    def i2l(self, p):
        ''' transform from image to local coordinates '''
        return self.d/self.q*(np.asarray(p)+0.5)


    def g2i(self, r):
        ''' transform from global to image coordinates '''
        a = np.asarray(np.round(self.R.apply(r-self.t, inverse=True)/self.d*self.q-0.5), dtype=int)
        a[a<0] = 0
        b = a>=self.shape
        a[b] = self.shape[b]
        return a


    def i2g(self, p):
        ''' transform from image to global coordinates '''
        return self.R.apply(self.d/self.q*(np.asarray(p)+0.5))+self.t


    def locate(self, offset=np.array([0, 0, 0])):
        ''' locate the mesh in local space where elements of offset range [0..1] '''
        s = self.R.apply(self.mesh.points, inverse=True)
        mn, mx = np.min(s, axis=0), np.max(s, axis=0)
        self.t = self.R.apply(mn-offset*(self.dim-mx+mn))
        return self.t


    def arePointsInCylinder(self, r, p1, p2, radius):
        ''' determine if points r are inside a cylinder of given radius and endponts p1 and p2 '''
        p = p2-p1
        a = r[0]-p1[0], r[1]-p1[1], r[2]-p1[2]
        b = r[0]-p2[0], r[1]-p2[1], r[2]-p2[2]
        c = a[1]*p[2]-a[2]*p[1], a[2]*p[0]-a[0]*p[2], a[0]*p[1]-a[1]*p[0]
        d = radius*np.linalg.norm(p)
        b0 = (c[0]**2 + c[1]**2 + c[2]**2) <= d**2
        b1 = (a[0]*p[0] + a[1]*p[1] + a[2]*p[2]) >= 0
        b2 = (b[0]*p[0] + b[1]*p[1] + b[2]*p[2]) <= 0
        return np.logical_and(np.logical_and(b0,b1),b2)


    #http://www.iquilezles.org/www/articles/diskbbox/diskbbox.htm
    def evalDiskBB(self, c, n, r):
        ''' bounding box for a disk defined by (c)enter, (n)ormal, (r)adius ''' 
        e = r*np.sqrt(1-n**2 )
        return np.c_[c-e, c+e].T


    def evalCylinderBB(self, pa, pb, r):
        ''' bounding box for a cylinder defined by points pa and pb, and a radius ra '''
        a = pb-pa
        e = r*np.sqrt(1-a**2/np.dot(a,a))
        return np.c_[np.minimum(pa-e, pb-e), np.maximum(pa+e, pb+e)].T


    def evalConeBB(self, pa, pb, ra, rb):
        ''' bounding box for a cone defined by points pa and pb, and radii ra and rb '''
        a = pb-pa
        e = np.sqrt(1-a**2/np.dot(a,a))
        return np.c_[np.minimum(pa-e*ra, pb-e*rb), np.maximum(pa+e*ra, pb+e*rb)].T


    def evalSphereBB(self, p, r):
        ''' bounding box for a sphere defined by center point p and radius r '''
        return np.c_[p-r, p+r].T


    def basis(self,w):
        ''' find orthonormal basis given a normal vector w (orthoganlization) '''
        w /= np.linalg.norm(w)
        wn = np.sqrt(1-w[0]**2)
        return np.array([[(1-w[0]**2)/wn,        0,  w[0]],
                         [ -w[0]*w[1]/wn,  w[2]/wn,  w[1]],
                         [ -w[0]*w[2]/wn, -w[1]/wn,  w[2]]])


    def apply(self, rs=0.001, rc=0.001):
        if self.outfilename is None:
            self.outfilename, ext = os.path.splitext(self.infilename)
        else:
            self.outfilename, ext = os.path.splitext(self.outfilename)
        self.outfilename += '.hdf5'
        with h5py.File(self.outfilename, 'w') as f:
            dset = f.create_dataset('mesh', self.shape, dtype=np.uint8, compression="gzip", compression_opts=7)
            vtx = self.mesh.points
            ii=0
            for p0 in vtx:
                print('vertex {:}/{:}'.format(ii, len(self.mesh.points)))
                p = self.g2l(p0)
                (i1,j1,k1), (i2,j2,k2) = self.l2i(p-rs), self.l2i(p+rs)
                i,j,k = np.ogrid[i1:i2, j1:j2, k1:k2]
                u,v,w = self.d[0]/self.q*(i+0.5), self.d[1]/self.q*(j+0.5), self.d[2]/self.q*(k+0.5)
                pv = (u-p[0])**2 + (v-p[1])**2 + (w-p[2])**2 <= rs**2
                # anti-aliasing filter
                # if self.q > 1:
                #     pv = decimate(pv, q=self.q//2, axis=2, n=1)
                #     pv = decimate(pv[::-1,::-1,::-1], q=self.q//2, axis=2, n=1)
                #     pv = decimate(pv, q=self.q//2, axis=1, n=1)
                #     pv = decimate(pv[::-1,::-1,::-1], q=self.q//2, axis=1, n=1)
                #     pv = decimate(pv, q=self.q//2, axis=0, n=1)
                #     pv = decimate(pv[::-1,::-1,::-1], q=self.q//2, axis=0, n=1)
                dset[i1:i2,j1:j2,k1:k2] = np.logical_or(dset[i1:i2,j1:j2,k1:k2], pv)
                ii += 1

            edges = [list(e) for e in self.edges]
            bbcyl = np.asarray([self.evalCylinderBB(pa=self.g2l(vtx[e[0]]), pb=self.g2l(vtx[e[1]]), r=rc) for e in edges])
            ii = 0
            for e,b in zip(edges, bbcyl):
                print('edge {:}/{:} ({:.2f}MB)'.format(ii, len(self.edges), os.path.getsize(self.outfilename)/1e6))
                (i1,j1,k1),(i2,j2,k2) = self.l2i(b[0]), self.l2i(b[1])
                i,j,k = np.ogrid[i1:i2, j1:j2, k1:k2]
                u,v,w = self.d[0]/self.q*(i+0.5), self.d[1]/self.q*(j+0.5), self.d[2]/self.q*(k+0.5)
                p1,p2 = self.g2l(vtx[e[0]]), self.g2l(vtx[e[1]])
                ev = self.arePointsInCylinder(r=(u,v,w), p1=p1, p2=p2, radius=rc)
                dset[i1:i2,j1:j2,k1:k2] = np.logical_or(dset[i1:i2,j1:j2,k1:k2], ev)
                ii += 1



if __name__ == "__main__":
    scale = 1
    meshslice = MeshSlice('cavity.msh', edgetype=0)
    meshslice.mesh.points *= scale
    meshslice.locate([0.5, 0.5, 0])
    meshslice.apply(rs=0.00025, rc=0.00025)
    print(meshslice)

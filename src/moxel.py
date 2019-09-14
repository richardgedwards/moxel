import os, sys
import numpy as np
from scipy.spatial.transform import Rotation
import meshio
import h5py
#from scipy.signal import decimate
# import imageio

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SLAParams:
    def __init__(self, dx=47.25e-6, dy=47.25e-6, dz=50e-6, imageshape=(2560,1440), zmax=155e-3):
        self.d = (dx,dy,dz)
        self.imageshape = imageshape
        self.zmax = zmax


class Moxel:
    def __init__(self, infilename, outfilename=None, R=None, t=None, radius=0, q=4, sla=None):
        self.R = Rotation([0,0,0,1]) if R is None else R
        self.t = np.array([0,0,0]) if t is None else t
        self.sla = SLAParams() if sla is None else sla
        self.infilename = infilename
        self.outfilename = outfilename
        self.mesh = meshio.read(infilename)
        self.edges = set()
        self.q = 1  # anti-aliasing factor (not implemented)
        self.radius = radius

        for cell in self.mesh.cells['tetra']:
            self.edges.add(frozenset((cell[0], cell[1])))
            self.edges.add(frozenset((cell[0], cell[2])))
            self.edges.add(frozenset((cell[0], cell[3])))
            self.edges.add(frozenset((cell[1], cell[2])))
            self.edges.add(frozenset((cell[1], cell[3])))
            self.edges.add(frozenset((cell[2], cell[3])))

        # for cell in self.mesh.cells['tetra10']:
        #     self.edges.add(frozenset((cell[0], cell[7])))
        #     self.edges.add(frozenset((cell[7], cell[3])))
        #     self.edges.add(frozenset((cell[3], cell[8])))
        #     self.edges.add(frozenset((cell[8], cell[1])))
        #     self.edges.add(frozenset((cell[1], cell[4])))
        #     self.edges.add(frozenset((cell[4], cell[0])))
        #     self.edges.add(frozenset((cell[1], cell[5])))
        #     self.edges.add(frozenset((cell[5], cell[2])))
        #     self.edges.add(frozenset((cell[0], cell[6])))
        #     self.edges.add(frozenset((cell[6], cell[2])))
        #     self.edges.add(frozenset((cell[3], cell[9])))
        #     self.edges.add(frozenset((cell[9], cell[2])))
    
        self.ui = np.unique([[p for p in edge] for edge in self.edges])


    def __repr__(self):
        np.set_printoptions(precision=5)
        extents = self.extents
        imgextents = np.c_[self.l2i(extents[:,0]), self.l2i(extents[:,1])]
        repr = "<Mesh Voxelization (Moxel) Object>:\n"
        repr += 'No. Edges: {:}\n'.format(len(self.edges))
        repr += 'No. Vertices: {:}\n'.format(len(self.mesh.points))
        repr += 'No. Unique edge vertices: {:}\n'.format(len(self.ui))
        repr += 'shape: {:}\n'.format(self.shape)
        repr += 'resolution: {:}\n'.format(self.sla.d)
        repr += 'build volume: {:}\n'.format(self.dim)
        repr += 'mesh size: {:}\n'.format(extents[:,1]-extents[:,0])
        repr += 'mesh occupancy: {:}\n'.format((extents[:,1]-extents[:,0])/self.dim)
        repr += 'mesh extents: \n{:}\n'.format(extents)
        repr += 'image extents: \n{:}\n'.format(imgextents)
        repr += 'image size: {:}\n'.format(1+imgextents[:,1]-imgextents[:,0])
        repr += 'local coordinate location: {:}\n'.format(self.t)
        repr += 'orientation of local basis vectors: \n{:}\n'.format(self.R.apply(np.identity(3)))
        return repr


    @property
    def shape(self):
        return np.asarray((self.sla.imageshape[0], self.sla.imageshape[1], int(np.round(self.sla.zmax/self.sla.d[2]))))


    @property
    def dim(self):
        return self.shape*self.sla.d


    @property
    def extents(self):
        s = self.g2l(self.mesh.points)
        return np.c_[np.min(s,axis=0)-self.radius, np.max(s,axis=0)+self.radius]


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
        a = np.asarray(np.round(s/self.sla.d*self.q-0.5), dtype=int)
        a[a<0] = 0
        b = a>=self.shape
        a[b] = self.shape[b]
        return a


    def i2l(self, p):
        ''' transform from image to local coordinates '''
        return self.sla.d/self.q*(np.asarray(p)+0.5)


    def g2i(self, r):
        ''' transform from global to image coordinates '''
        a = np.asarray(np.round(self.R.apply(r-self.t, inverse=True)/self.sla.d*self.q-0.5), dtype=int)
        a[a<0] = 0
        b = a>=self.shape
        a[b] = self.shape[b]
        return a


    def i2g(self, p):
        ''' transform from image to global coordinates '''
        return self.R.apply(np.asarray(self.sla.d)/self.q*(np.asarray(p)+0.5))+self.t


    def locate(self, offset=np.array([0, 0, 0])):
        ''' locate the mesh in local space with normalized offset [0..1] '''
        s = self.R.apply(self.mesh.points, inverse=True)
        mn, mx = np.min(s, axis=0)-self.radius, np.max(s, axis=0)+self.radius
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


    # Bounding boxes of various geometries 
    # ref: http://www.iquilezles.org/www/articles/diskbbox/diskbbox.htm
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


    def evalMaxElementDims(self):
            points = self.mesh.points
            mxp = 1+np.max(np.array([self.g2i(p+self.radius)-self.g2i(p-self.radius) for p in points[self.ui]]), axis=0)

            mxc = np.zeros([3,], dtype=int)
            for edge in self.edges:
                e = list(edge)
                (p0,p1) = self.evalCylinderBB(pa=points[e[0]], pb=points[e[1]], r=self.radius)
                d = 1+self.g2i(p1)-self.g2i(p0)
                idx = d>mxc
                mxc[idx] = d[idx]
            return np.max(np.c_[mxc,mxp], axis=1)


    def eval(self):
        ''' evaluate the voxelization of the given mesh '''
        if self.outfilename is None:
            self.outfilename, ext = os.path.splitext(self.infilename)
        else:
            self.outfilename, ext = os.path.splitext(self.outfilename)
        self.outfilename += '.hdf5'
        with h5py.File(self.outfilename, 'w') as f:
            dset = f.create_dataset('moxel', self.shape, dtype=np.uint8, compression="gzip", compression_opts=7)
            r = self.mesh.points

            # need to iterate over the unique set of edge points 
            # the number of vertices can exceed the number of edge points depending on the meshing algorithm
            points = r[self.ui]
            ii=0
            for p0 in points:
                print('vertex {:}/{:} ({:.2f}MB)'.format(ii, len(points), os.path.getsize(self.outfilename)/1e6))
                p = self.g2l(p0)
                (i1,j1,k1), (i2,j2,k2) = self.l2i(p-self.radius), self.l2i(p+self.radius)
                i,j,k = np.ogrid[i1:i2, j1:j2, k1:k2]
                u,v,w = self.sla.d[0]/self.q*(i+0.5), self.sla.d[1]/self.q*(j+0.5), self.sla.d[2]/self.q*(k+0.5)
                pv = (u-p[0])**2 + (v-p[1])**2 + (w-p[2])**2 <= self.radius**2
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
            bbcyl = np.asarray([self.evalCylinderBB(pa=self.g2l(r[e[0]]), pb=self.g2l(r[e[1]]), r=self.radius) for e in edges])
            ii = 0
            for e,b in zip(edges, bbcyl):
                print('edge {:}/{:} ({:.2f}MB)'.format(ii, len(self.edges), os.path.getsize(self.outfilename)/1e6))
                (i1,j1,k1),(i2,j2,k2) = self.l2i(b[0]), self.l2i(b[1])
                i,j,k = np.ogrid[i1:i2, j1:j2, k1:k2]
                u,v,w = self.sla.d[0]/self.q*(i+0.5), self.sla.d[1]/self.q*(j+0.5), self.sla.d[2]/self.q*(k+0.5)
                p1,p2 = self.g2l(r[e[0]]), self.g2l(r[e[1]])
                ev = self.arePointsInCylinder(r=(u,v,w), p1=p1, p2=p2, radius=self.radius)
                dset[i1:i2,j1:j2,k1:k2] = np.logical_or(dset[i1:i2,j1:j2,k1:k2], ev)
                ii += 1


    def arePointsInside(self, r, t):
        ''' determine if points r are on the inside of a simplex with a given boundary triangle t '''
        # p = np.average(t, axis=0)
        p = t[0]
        n0, n1, n2 = np.cross(t[1]-t[0], t[2]-t[0])
        v0, v1, v2 = p[0]-r[0], p[1]-r[1], p[2]-r[2] 
        return v0*n0+v1*n1+v2*n2 > 0


    def slice(self):
        if self.outfilename is None:
            self.outfilename, ext = os.path.splitext(self.infilename)
        else:
            self.outfilename, ext = os.path.splitext(self.outfilename)
        self.outfilename += '.hdf5'
        print(self.outfilename)
        with h5py.File(self.outfilename, 'w') as f:
            dset = f.create_dataset('moxel', self.shape, dtype=np.uint8, compression="gzip", compression_opts=7)
            s = self.g2l(self.mesh.points)
            for ii,c in enumerate(self.mesh.cells['tetra']):
                print('cell {:}/{:}'.format(ii, len(self.mesh.cells['tetra'])))
                ext = np.array([s[c0] for c0 in c])
                (i1,j1,k1), (i2,j2,k2) = self.l2i(np.min(ext, axis=0)), self.l2i(np.max(ext, axis=0))
                i,j,k = np.ogrid[i1:i2, j1:j2, k1:k2]
                u,v,w = self.sla.d[0]/self.q*(i+0.5), self.sla.d[1]/self.q*(j+0.5), self.sla.d[2]/self.q*(k+0.5)
                t = np.array(
                    [[s[c[0]], s[c[2]], s[c[1]]],  
                     [s[c[0]], s[c[3]], s[c[2]]],  
                     [s[c[0]], s[c[1]], s[c[3]]],  
                     [s[c[1]], s[c[2]], s[c[3]]]])
                # t = np.array(
                #     [[s[c[0]], s[c[6]], s[c[4]]], [s[c[4]], s[c[5]], s[c[1]]], [s[c[4]], s[c[6]], s[c[5]]], [s[c[6]], s[c[2]], s[c[5]]], 
                #      [s[c[0]], s[c[7]], s[c[6]]], [s[c[6]], s[c[9]], s[c[2]]], [s[c[7]], s[c[9]], s[c[6]]], [s[c[7]], s[c[3]], s[c[9]]], 
                #      [s[c[0]], s[c[4]], s[c[7]]], [s[c[7]], s[c[8]], s[c[3]]], [s[c[4]], s[c[8]], s[c[7]]], [s[c[4]], s[c[1]], s[c[8]]], 
                #      [s[c[1]], s[c[5]], s[c[8]]], [s[c[8]], s[c[5]], s[c[9]]], [s[c[8]], s[c[9]], s[c[3]]], [s[c[5]], s[c[2]], s[c[9]]]])
                p0 = np.ones((i2-i1, j2-j1, k2-k1), dtype=np.uint8)
                for t0 in t:
                    p1 = self.arePointsInside(r=(u,v,w), t=t0)
                    p0 = np.logical_and(p0, p1)
                dset[i1:i2,j1:j2,k1:k2] = np.logical_or(dset[i1:i2,j1:j2,k1:k2], p0)
                # print(k1, k2)
                # plt.imshow(dset[:,:,(k1+k2)/2])
                # plt.show()



        # import mpl_toolkits.mplot3d as a3
        # import matplotlib.colors as colors
        # import pylab as plt
        # ax = a3.Axes3D(plt.figure())
            # for t0 in t:
            #     tri = a3.art3d.Poly3DCollection([t0])
            #     tri.set_color(colors.rgb2hex(np.random.rand(3)))
            #     tri.set_edgecolor('k')
            #     ax.add_collection3d(tri)
            # ax.scatter(point[0], point[1], point[2], 'k.')
            # for c0 in c:
            #     ax.scatter(c0[0], c0[1], c0[2], '.')
            # scale = 0.005
            # for c0,n0,v0 in zip(c,n,v):
            #     ax.plot([c0[0], c0[0]+n0[0]*scale], [c0[1], c0[1]+n0[1]*scale], [c0[2], c0[2]+n0[2]*scale])
            #     ax.plot([c0[0], c0[0]+v0[0]*scale], [c0[1], c0[1]+v0[1]*scale], [c0[2], c0[2]+v0[2]*scale])
            # ax.axis('square')
            # ax.set_xlim3d([-0.02, 0.02])
            # ax.set_ylim3d([-0.02, 0.02])
            # ax.set_zlim3d([-0.02, 0.02])
            # plt.show()


    def render(self, windowsize=(800,600)):
        from glumpy import app, gloo, gl
        from glumpy.transforms import Trackball, Position

        vertices = self.mesh.points

        a,b,c  = self.shape[0]*self.sla.d[0], self.shape[1]*self.sla.d[1], self.sla.zmax
        scale = 2/a
        
        e = a*0.1
        plate_vertices = np.array([
            [0,0,0], [a+e,0,0],
            [0,b,0], [a,b,0],
            [0,0,0], [0,b+e,0],
            [a,0,0], [a,b,0],
            [0,0,c], [a,0,c],
            [0,b,c], [a,b,c],
            [0,0,c], [0,b,c],
            [a,0,c], [a,b,c],
            [0,0,0], [0,0,c+e],
            [a,0,0], [a,0,c],
            [a,b,0], [a,b,c],
            [0,b,0], [0,b,c],
            ])
        plate_vertices = self.l2g(plate_vertices)

        self.window = app.Window(width=windowsize[0], height=windowsize[1])

        vertex_prg = gloo.Program(self.vertex, self.fragment, count=len(vertices))
        vertex_prg['transform'] = Trackball(Position("position"))
        self.window.attach(vertex_prg['transform'])
        vertex_prg['position'] = vertices*scale
        vertex_prg['color'] = 1,1,1,1

        buildplate_prg = gloo.Program(self.vertex, self.fragment, count=len(plate_vertices))
        buildplate_prg['transform'] = Trackball(Position("position"))
        self.window.attach(buildplate_prg['transform'])
        buildplate_prg['position'] = plate_vertices*scale
        buildplate_prg['color'] = 1,0,0,1

        gl.glLineWidth(1.0)
        gl.glPointSize(1.0)

        @self.window.event
        def on_draw(dt):
            self.window.clear()
            vertex_prg.draw(gl.GL_POINTS)
            buildplate_prg.draw(gl.GL_LINES)

        app.run()


    vertex = """
        attribute vec3 position;
        void main()
        {
            gl_Position = <transform>;
        } """
    
    
    fragment = """
        uniform vec4 color;
        void main() {
            gl_FragColor = vec4(color);
        } """


if __name__ == "__main__":
    scale = 0.05
    rot = Rotation.from_rotvec(np.pi/2*np.array([0,1,0]))
    mxl = Moxel('./msh/sphere/sphere.msh', radius=250e-6, R=None)
    mxl.mesh.points *= scale
    mxl.locate([0.5, 0.5, 0.0])
    # mxl.render()
    mxl.eval()
    # mxl.slice()
    print(mxl.evalMaxElementDims())
    print(mxl)
    print(mxl.mesh)


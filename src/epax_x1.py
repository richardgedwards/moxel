import numpy as np
import h5py
import blosc, zlib  # used from compression

# unit conversion to meter
um = 1e-6
mm = 1e-3
inch = 0.0254
mil = 0.0000254


class EPAX_X1:
    
    def __init__(self, dz=50*um, c=155*mm):
        #dz = 50*um # [10um to 200um] 1.25um min
        # c = 155mm max
        self.px, self.py, self.pz = 2560, 1440, int(c/dz)
        self.dx = self.dy = 47.25*um
        self.dz = dz

        # Builld Volume:  115mm x 65mm x 155mm
        self.a = self.dx*self.px # 68.04*mm
        self.b = self.dy*self.py # 120.96*mm
        self.c = c 


    def __repr__(self):
        v = 'build volume: {:.2f}mm  x {:.2f}mm x {:.2f}mm'.format(self.a/mm, self.b/mm, self.c/mm)
        p = 'pixels: {:}, {:}, {:}'.format(self.px, self.py, self.pz)
        r = 'resolution: {:.2f}um, {:.2f}um, {:.2f}um'.format(self.dx/um, self.dy/um, self.dz/um)
        return 'EPAX X1\n' + v + '\n' + p + '\n' + r


if __name__=="__main__":
    x1 = EPAX_X1()
    print(x1)

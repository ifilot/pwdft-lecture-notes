import numpy as np
from pytessel import PyTessel
from pypwdft import PyPWDFT, SystemBuilder
import pickle
import os

def main():
    # create cubic periodic system with lattice size of 10 Bohr units
    npts = 64       # number of grid points
    sz = 10

    # construct CO molecule system via SystemBuilder
    s = SystemBuilder().from_name('CO', sz=sz, npts=npts)

    # construct calculator object
    calculator = PyPWDFT(s)

    # use caching
    if os.path.exists('co.pickle'):
        with open('co.pickle', 'rb') as f:
            res = pickle.load(f)
    else:
        res = calculator.scf(tol=1e-5, verbose=True)
        with open('co.pickle', 'wb') as f:
            pickle.dump(res, f)

    # print molecular orbital energies
    print(res['orbe'])

    # generate PyTessel object
    pytessel = PyTessel()

    i = 3
    for up in [1,2,4]:
        print('Building isosurfaces: %02i' % (i+1))
        scalarfield = upsample_grid(res['orbc_fft'][i], sz**3, up)
        unitcell = np.identity(3) * sz
    
        # build positive real isosurface
        vertices, normals, indices = pytessel.marching_cubes(scalarfield.real.flatten(), scalarfield.shape, unitcell.flatten(), 0.03)
        pytessel.write_ply('MO_PR_%02i_up%i.ply' % (i+1, up), vertices, normals, indices)
    
        # build negative real isosurface
        vertices, normals, indices = pytessel.marching_cubes(scalarfield.real.flatten(), scalarfield.shape, unitcell.flatten(), -0.03)
        pytessel.write_ply('MO_NR_%02i_up%i.ply' % (i+1, up), vertices, normals, indices)

def upsample_grid(scalarfield_fft, Omega, upsample=4):
    Nx, Ny, Nz = scalarfield_fft.shape
    Nx_up = Nx * upsample
    Ny_up = Nx * upsample
    Nz_up = Nx * upsample

    # shift the frequencies
    fft = np.fft.fftshift(scalarfield_fft)

    # perform padding
    fft_upsampled = np.pad(fft, [((Nz_up-Nz)//2,),
                                ((Ny_up-Ny)//2,),
                                ((Nx_up-Nx)//2,)], 'constant')

    # shift back
    fft_hires = np.fft.ifftshift(fft_upsampled)

    return np.fft.ifftn(fft_hires) * np.prod([Nx_up, Ny_up, Nz_up]) / np.sqrt(Omega)

if __name__ == '__main__':
    main()
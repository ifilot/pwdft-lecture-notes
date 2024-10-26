# -*- coding: utf-8 -*-

#
# EXERCISE 8
#

from pypwdft import SystemBuilder, PyPWDFT, PeriodicSystem
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import AxesImage
import pickle
import os
import scipy
from scipy.stats.qmc import LatinHypercube

def main():
    # build atomic system
    sz = 10
    npts = 32
    s = SystemBuilder().from_file('bh3.xyz', sz, npts)
    
    # use caching
    if os.path.exists('bh3.pickle'):
        with open('bh3.pickle', 'rb') as f:
            res = pickle.load(f)
    else:
        res = PyPWDFT(s).scf(verbose=True)
        with open('bh3.pickle', 'wb') as f:
            pickle.dump(res, f)
       
    produce_plot(res, sz, npts, 'fig9.pdf')
    
    ### TRANSFORMATION
    
    # calculate overlap matrix prior to transformation
    S = calculate_overlap_matrix(res['orbc_rs'], sz, npts)
    print('S = ', S)
    
    # calculate kinetic energies prior to transformation
    print('Kinetic energies:')
    for i in range(2,4):
        print(calculate_kinetic_energy(res['orbc_fft'][i], sz, npts).real)
    
    # perform transformation
    print('\nPerforming Transformation\n')
    for i in range(2,4):
        res['orbc_rs'][i] = optimize_real(res['orbc_rs'][i])
        
    # calculate overlap matrix after transformation
    S = calculate_overlap_matrix(res['orbc_rs'], sz, npts)
    print('S = ', S)
    
    # calculate kinetic energies after to transformation
    print('Kinetic energies:')
    Ct = np.sqrt(sz**3) / npts**3
    for i in range(2,4):
        print(calculate_kinetic_energy(np.fft.fftn(res['orbc_rs'][i]) * Ct, sz, npts).real)
    
    # reproduce plots after transformation
    produce_plot(res, sz, npts, 'fig10.pdf')

def produce_plot(res, sz, npts, filename):
    """
    Generate contour plots for the real and imaginary parts, as well as the 
    electron density, of the occupied orbitals of BH3.
    """
    fig, ax = plt.subplots(2, 2, dpi=144, figsize=(6,5))
    im = np.zeros((2,4), dtype=AxesImage)
    for i in range(2,4):
        limit = 0.3
        im[0][i-2] = ax[0,i-2].imshow(res['orbc_rs'][i,npts//2,:,:].real, extent=(0,sz,0,sz), 
                   interpolation='bicubic', cmap='RdBu',
                   vmin=-limit, vmax=limit)
        im[1][i-2] = ax[1,i-2].imshow(res['orbc_rs'][i,npts//2,:,:].imag, extent=(0,sz,0,sz), 
                   interpolation='bicubic', cmap='RdBu',
                   vmin=-limit, vmax=limit)
    
        for j in range(0,2):
            ax[j,i-2].set_xlabel(r'$x$ [a.u.]')
            ax[j,i-2].set_ylabel(r'$y$ [a.u.]')
            
            divider = make_axes_locatable(ax[j,i-2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im[j][i-2], cax=cax, orientation='vertical')
            
        ax[0,i-2].set_title(r'$\mathbb{R}\;[\psi_{%i}]$' % (i+1))
        ax[1,i-2].set_title(r'$\mathbb{I}\;[\psi_{%i}]$' % (i+1))
        #ax[2,i].set_title(r'$\rho_{%i}$' % (i+1))

    plt.tight_layout()
    plt.savefig(filename)

def calculate_overlap_matrix(orbc, sz, npts):
    """
    Calculate the overlap matrix in real-space
    """
    N = len(orbc)
    S = np.zeros((N,N))
    dV = (sz / npts)**3
    for i in range(N):
        for j in range(N):
            S[i,j] = (np.sum(orbc[i].conjugate() * orbc[j]) * dV).real
            
    return S

def calculate_kinetic_energy(orbc_fft, sz, npts):
    """
    Calculate the kinetic energy of a molecular orbital as represented by a
    set of plane-wave coefficients
    """
    s = PeriodicSystem(sz=sz, npts=npts)
    
    return 0.5 * np.einsum('ijk,ijk,ijk', orbc_fft.conjugate(), s.get_pw_k2(), orbc_fft)

def optimize_real(psi):
    """
    Perform a phase transformation such that the real part of wave function
    is maximized
    """
    def f(angle, psi):
        phase = np.exp(1j * angle)
        return -np.sum((psi * phase).real**2)

    res = scipy.optimize.differential_evolution(f, [(-np.pi,np.pi)], args=(psi,),
                                  tol=1e-12)
    
    deltaV = (1000 / 32**3)
    
    phase = np.exp(1j * res.x)
    print(np.degrees(res.x), 
          np.sum((psi).real**2) * deltaV, 
          np.sum((psi*phase).real**2) * deltaV)
    
    return psi * np.exp(1j * res.x)

if __name__ == '__main__':
    main()
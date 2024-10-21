import unittest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy

# add a reference to load the PyPWDFT module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import the required libraries for the test
from pypwdft import PeriodicSystem

def main():
    sz = 10
    lc = '#00b9f2'  # line color
    
    #--------------------------------------------------------------------------
    # Show self-overlap in real-space and reciprocal space and the kinetic
    # energy as function of the number of sampling points per Cartesian
    # direction
    #--------------------------------------------------------------------------

    edens_arr = []
    edens_fft_arr = []
    ekin_arr = []
    nptss = np.logspace(3,10,8, base=2)
    print(nptss)
    for npts in tqdm.tqdm(nptss):
        edens, edens_fft, ekin, field, fft_field = sample_1s_orbital(int(npts), sz)
        edens_arr.append(edens)
        edens_fft_arr.append(edens_fft)
        ekin_arr.append(ekin)
        
    fig, ax = plt.subplots(3,1, dpi=144)
    ax[0].loglog(nptss, np.abs(1 - np.array(edens_arr).real), 'o--', color=lc)
    ax[1].loglog(nptss, np.abs(1 - np.array(edens_fft_arr).real), 'o--', color=lc)
    ax[2].loglog(nptss, np.abs(0.5 - np.array(ekin_arr).real) / 0.5, 'o--', color=lc)
           
    ax[0].set_title(r'Numerical integration of unit cell')
    ax[1].set_title(r'Sum of squares of expansion coefficients')
    ax[2].set_title(r'Kinetic energy')
    
    for i in range(3):
        ax[i].set_ylabel(r'$E_{\text{rel}}$ [-]')
        ax[i].grid(linestyle='--')
        ax[i].set_ylim(1e-4,1.5)
    
    plt.tight_layout()
    plt.savefig('fig4_ekin_convergence.pdf')
    
def sample_1s_orbital(npts=64, sz=10):
    p = PeriodicSystem(sz, npts)
    p.add_atom(5,5,5,1)
    
    # generate scalar field
    R = np.array([5,5,5])
    r =  p.get_r() - R
    Rr = np.sqrt(np.einsum('ijkl,ijkl->ijk', r, r))
    field = 1.0 / np.sqrt(np.pi) * np.exp(-Rr)
    
    # produce its expansion in plane waves
    fft_field = scipy.fft.fftn(field) * p.get_ct()
    
    # calculate properties
    edens = np.sum(np.power(field,2)) * (sz/npts)**3
    edens_fft = np.einsum('ijk,ijk', fft_field, fft_field)
    ekin = 0.5 * np.einsum('ijk,ijk', fft_field**2, p.get_pw_k2())
    
    return edens, edens_fft, ekin, field, fft_field

if __name__ == '__main__':
    main()

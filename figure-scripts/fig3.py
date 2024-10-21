import unittest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# add a reference to load the PyPWDFT module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import the required libraries for the test
from pypwdft import PeriodicSystem

def main():
    #--------------------------------------------------------------------------
    # produce a plot of the wave function scalar field and its FFT transform
    #--------------------------------------------------------------------------
    npts = 32
    sz = 10
    edens, edens_fft, ekin, field, fft_field = sample_1s_orbital(npts, sz)
    
    # plot the two fields
    fig, ax = plt.subplots(1,2,dpi=144, figsize=(8,4))
    im = ax[0].imshow(field[npts//2,:,:], cmap='Blues', origin='lower',
                 extent=[0,sz,0,sz])
    ax[0].set_xlabel('x [a.u.]')
    ax[0].set_ylabel('y [a.u.]')
    ax[0].set_title(r'$\psi_{\text{1s}}(\vec{r})$ [a.u.]$^{-3/2}$')
    
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    im = ax[1].imshow(np.power(np.fft.fftshift(fft_field)[npts//2,:,:],2).real, 
                 cmap='Blues', origin='lower',
                 extent=[-1,1,-1,1]) # note that fftshift is applied to center
    ax[1].set_xlabel('$G_{x} [\pi \cdot N / a_{0}]^{-1}$')
    ax[1].set_ylabel('$G_{y} [\pi \cdot N / a_{0}]^{-1}$')
    ax[1].set_title(r'$|\tilde{\psi}_{\text{1s}}(\vec{G})|^{2}$')
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    plt.tight_layout()
    plt.savefig('fig3_horb1s_fields.pdf')
    
def sample_1s_orbital(npts=64, sz=10):
    p = PeriodicSystem(sz, npts)
    p.add_atom(5,5,5,1)
    
    # generate scalar field
    R = np.array([5,5,5])
    r =  p.get_r() - R
    Rr = np.sqrt(np.einsum('ijkl,ijkl->ijk', r, r))
    field = 1.0 / np.sqrt(np.pi) * np.exp(-Rr)
    
    # produce its expansion in plane waves
    fft_field = np.fft.fftn(field) * p.get_ct()
    
    # calculate properties
    edens = np.sum(np.power(field,2)) * (sz/npts)**3
    edens_fft = np.einsum('ijk,ijk', fft_field, fft_field)
    ekin = 0.5 * np.einsum('ijk,ijk', fft_field**2, p.get_pw_k2())
    
    return edens, edens_fft, ekin, field, fft_field

if __name__ == '__main__':
    main()

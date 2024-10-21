import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from pyqint import HF,Molecule,PyQInt,cgf
import sys, os
import tqdm

# add a reference to load the PyPWDFT module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import the required libraries for the test
from pypwdft import PeriodicSystem

def main():
    
    highest = 512
    nptss = np.logspace(3,
                        np.log2(highest),
                        int(np.log2(highest))-2, 
                        base=2,
                        endpoint=True)      # number of sampling points per cartesian direction
    print(nptss)
    a0s = [10,20,50,100]                    # edge sizes of cubic unit cell
    
    enucs = np.zeros((len(a0s), len(nptss))) # data container
    ereps = np.zeros((len(a0s), len(nptss))) # data container
    for i,s in enumerate(tqdm.tqdm(a0s)):
        v = []
        for j,npts in enumerate(nptss):
            
            npts = int(npts)
            
            # generate wave function
            r, kvec, k2 = build_lattice_points(s, npts)
            
            print(s,npts,np.max(k2)/2 * 27.2114)
            
            R = np.array([s/2,s/2,s/2])
            rRv =  r - R
            Rr2 = np.einsum('ijkl,ijkl->ijk', rRv, rRv)
            wf = (0.5 * np.pi)**(-3/4) * np.exp(-Rr2)
            
            # calculate Hartree and nuclear attraction potentials
            hartree = calculate_electronic_repulsion(wf * wf, npts, s)
            vnuc = calculate_nuclear_attraction(npts, s)
            
            # calculate energetic terms
            erep = (np.sum(wf * wf * hartree) * (s/npts)**3).real
            enuc = (np.sum(wf * wf * vnuc) * (s/npts)**3).real
            
            # store
            enucs[i,j] = enuc
            ereps[i,j] = erep
    
    # build graph
    markers = ['x', 'v', '^', 'o', '*']
    fig, ax = plt.subplots(nrows=3, sharex=True, dpi=144, figsize=(8,6))

    for i,(s,m) in enumerate(zip(a0s, markers)):
        ax[0].plot(nptss, ereps[i,:], linestyle='--', label=r'$a_{0} = %.1f$ a.u.' % s,
                marker=m, color='#222222', markersize=5, linewidth=1)
        ax[1].plot(nptss, enucs[i,:], linestyle='--', label=r'$a_{0} = %.1f$ a.u.' % s,
                marker=m, color='#222222', markersize=5, linewidth=1)
        ax[2].plot(nptss, (ereps + enucs)[i,:], linestyle='--', label=r'$a_{0} = %.1f$ a.u.' % s,
                marker=m, color='#222222', markersize=5, linewidth=1)
    
    ax[0].hlines(2/np.sqrt(np.pi), 1, 1000, linewidth=1, color='black', linestyle='solid')
    ax[1].hlines(-2 * np.sqrt(2/np.pi), 1, 1000, linewidth=1, color='black', linestyle='solid')
    ax[2].hlines(2/np.sqrt(np.pi) + -2 * np.sqrt(2/np.pi), 1, 1000, linewidth=1, color='black', linestyle='solid')
    
    # add arrow
    ax[0].arrow(6, 2/np.sqrt(np.pi) + 2, 
                0, -2,
                length_includes_head=True,
                head_width=0.5, 
                head_length=0.5,
                color='black')
    
    ax[1].arrow(6, -2 * np.sqrt(2/np.pi) - 2, 
                0, 2,
                length_includes_head=True,
                head_width=0.5, 
                head_length=0.5,
                color='black')
    
    ax[2].arrow(6, 2/np.sqrt(np.pi) + -2 * np.sqrt(2/np.pi) + 1.5, 
                0, -1.5,
                length_includes_head=True,
                head_width=0.5, 
                head_length=0.5,
                color='black')
    
    for i in range(0,3):
        ax[i].grid(linestyle='--')
        ax[i].set_xscale('symlog')
        ax[i].set_yscale('symlog')
        ax[i].legend(loc='lower right')
        
    ax[0].set_ylim(0,100)
    ax[0].set_xlim(5,1000)
    ax[2].set_ylim(-3,100)
        
    ax[i].set_xlabel('Number of sampling points per cartesian direction [-]')

    ax[0].set_ylabel(r'$E_{\text{rep}}$ [Ht]')
    ax[1].set_ylabel(r'$E_{\text{nuc}}$ [Ht]')
    ax[2].set_ylabel(r'$E_{\text{tot}}$ [Ht]')
        
    plt.tight_layout()
    
    plt.savefig('hartree.pdf')

def calculate_electronic_repulsion(edens, npts, sz):
    """
    Calculate the potential energy of a 1s atomic orbital
    """
    r, k, k2 = build_lattice_points(sz, npts)
    
    fft_edens = np.fft.fftn(edens)
    
    # solve Poisson equation in reciprocal space
    with np.errstate(divide='ignore', invalid='ignore'):
       potg = 4 * np.pi * fft_edens / k2    # reciprocal space potential
       potg[~np.isfinite(potg)] = 0         # set non-finite values to zero 
    
    # convert back to real space
    harpot = np.fft.ifftn(potg)
    
    return harpot

def calculate_nuclear_attraction(npts, sz):
    """
    Calculate the potential energy of a 1s atomic orbital
    """
    r, k, k2 = build_lattice_points(sz, npts)
    
    # generate wave function
    R = np.array([sz/2,sz/2,sz/2])
    
    # generate structure factor and nuclear attraction field
    sf = np.exp(-1j * k @ R) / np.sqrt(sz**3)
    ct = np.sqrt(sz**3) / npts**3
    with np.errstate(divide='ignore', invalid='ignore'):
        nupotg = -4.0 * np.pi / k2
        nupotg[0,0,0] = 0
    vnuc = sc.fft.ifftn(sf * nupotg) / ct
    
    return vnuc

def build_lattice_points(sz, npts):
    """
    Build the lattice points, both in real-space as well as in reciprocal space.
    Also calculate the squared reciprocal space vector magnitudes
    """
    # determine grid points in real space
    c = np.linspace(0, sz, npts, endpoint=False)

    # construct real space sampling vectors
    z, y, x = np.meshgrid(c, c, c, indexing='ij')
    
    N = len(c)
    cvec = np.zeros((npts, npts, npts, 3))
    cvec[:,:,:,0] = x
    cvec[:,:,:,1] = y
    cvec[:,:,:,2] = z
    
    # calculate plane wave vector coefficients in one dimension
    k = np.fft.fftfreq(npts) * 2.0 * np.pi * (npts / sz)
    
    # construct plane wave vectors
    k3, k2, k1 = np.meshgrid(k, k, k, indexing='ij')
    
    N = len(k)
    kvec = np.zeros((N,N,N,3))
    kvec[:,:,:,0] = k1
    kvec[:,:,:,1] = k2
    kvec[:,:,:,2] = k3
    
    k2 = np.einsum('ijkl,ijkl->ijk', kvec, kvec)
    
    return cvec, kvec, k2

def build_wavefunction(coeff, cgfs, a0=10, npts=64):
    integrator = PyQInt()
    grid = integrator.build_rectgrid3d(-a0/2, a0/2, npts)
    scalarfield = np.reshape(integrator.plot_wavefunction(grid, coeff, cgfs), (npts, npts, npts))

    return scalarfield

if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import tqdm
import os

#
# Figure 5
# Potential energy hydrogen atom
#

def main():    
    # plotcolors
    lc = '#00b9f2'  # line color
    highest = 512
    nptss = np.logspace(3,
                        np.log2(highest),
                        int(np.log2(highest))-2, 
                        base=2,
                        endpoint=True)      # number of sampling points per cartesian direction
    a0s = [10,20,50,100,200]                # edge sizes of cubic unit cell
    
    if not os.path.exists('vnuc.npy'):    
        vnuc = np.zeros((len(a0s), len(nptss))) # data container
        for i,s in enumerate(tqdm.tqdm(a0s)):
            v = []
            for j,npts in enumerate(nptss):
                vnuc[i,j] = calculate_nuclear_attraction(int(npts), s)
        # store vnuc
        np.save('vnuc', vnuc)
    
    else:
        vnuc = np.load('vnuc.npy')
    
    # build graph
    markers = ['x', 'v', '^', 'o', '*']
    fig, ax = plt.subplots(1,1,dpi=144, figsize=(6,3))
    for i,(s,m) in enumerate(zip(a0s, markers)):
        ax.plot(nptss, vnuc[i,:], linestyle='--', label=r'$a_{0} = %.1f$ a.u.' % s,
                marker=m, color=lc, markersize=5, linewidth=1)
    ax.grid(linestyle='--')
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    ax.set_xlabel('Number of sampling points per cartesian direction [-]')
    ax.set_ylabel(r'$E_{\text{pot}}$ [Ht]')
    ax.legend(loc='lower right')        
    plt.tight_layout()
    plt.savefig('fig5_nuclear_attraction.pdf')

def calculate_nuclear_attraction(npts, sz):
    """
    Calculate the potential energy of a 1s atomic orbital
    """
    r, k, k2 = build_lattice_points(sz, npts)
    
    # generate wave function
    R = np.array([sz/2,sz/2,sz/2])
    rRv =  r - R
    Rr = np.sqrt(np.einsum('ijkl,ijkl->ijk', rRv, rRv))
    wf = 1.0 / np.sqrt(np.pi) * np.exp(-Rr)
    
    # generate structure factor and nuclear attraction field
    sf = np.exp(-1j * k @ R) / np.sqrt(sz**3)
    ct = np.sqrt(sz**3) / npts**3
    with np.errstate(divide='ignore', invalid='ignore'):
        nupotg = -4.0 * np.pi / k2
        nupotg[0,0,0] = 0
    vnuc = sc.fft.ifftn(sf * nupotg) / ct
    
    dV = (sz/npts)**3 # real-space integration constant
    Enuc = np.einsum('ijk,ijk,ijk', wf, wf, vnuc).real * dV
    
    return Enuc

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

if __name__ == '__main__':
    main()

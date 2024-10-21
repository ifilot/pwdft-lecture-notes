# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# add a reference to load the PyPWDFT module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import the required libraries for the test
from pypwdft import PeriodicSystem

npts = 8
p = PeriodicSystem(1, npts)

fig, ax = plt.subplots(1,2, dpi=144, figsize=(7,3))
ax[0].scatter(p.get_r()[npts//2,:,:,0], p.get_r()[npts//2,:,:,1],
              color='#00b9f2', marker='o', s=10)
ax[0].set_xlim(-0.2,1.2)
ax[0].set_ylim(-0.2,1.2)
ax[0].set_aspect('equal', 'box')
ax[0].set_title('Real-space sampling points')
ax[0].hlines([0,1],0,1, linestyle='--', color='black', linewidth=0.5)
ax[0].vlines([0,1],0,1, linestyle='--', color='black', linewidth=0.5)
ax[0].set_xlabel('$x$ [a.u.]')
ax[0].set_ylabel('$x$ [a.u.]')

G = p.get_pw_k() / np.pi
ax[1].scatter(G[npts//2,:,:,0], G[npts//2,:,:,1],
              color='#00b9f2', marker='x', s=10)
ax[1].set_xlim(-10, 10)
ax[1].set_ylim(-10, 10)

ax[1].set_aspect('equal', 'box')
ax[1].set_title('Reciprocal-space vectors')
ax[1].hlines([-8,8],-8,8, linestyle='--', color='black', linewidth=0.5)
ax[1].vlines([-8,8],-8,8, linestyle='--', color='black', linewidth=0.5)
ax[1].set_xlabel('$G_{x} / \pi$ [1 / a.u.]')
ax[1].set_ylabel('$G_{y} / \pi$ [1 / a.u.]')
plt.tight_layout()

plt.savefig('fig2_sampling.pdf')

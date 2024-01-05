#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:34:03 2024

@author: jp
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import lattice
from atom import Helium
import phys_constants

h = phys_constants.h
hbar = phys_constants.hbar
gHe = phys_constants.gHe
d_latt = phys_constants.d_latt


he = Helium()

m = he.mass


# Critical exponent for universality class:
beta = 0.3485
u_c = 26


# Data input:
uj_vs_s = {
    2: 5.5425,
    5: 7.78888,
    7.5: 8.91949,
    10: 9.775,
    12.5: 10.4694,
    15: 11.0568,
    20: 12.0196,
    22: 12.3482,
    24: 12.6522,
    25: 12.7961,
    30: 13.4492,
    35: 14.014,
    }

# Invert:
s_vs_uj = {s: uj for (uj, s) in uj_vs_s.items()}

Varr = sorted(uj_vs_s.values())

eff_mass = pd.Series(index=sorted(uj_vs_s.keys()), dtype=float)
dens = pd.Series(index=sorted(uj_vs_s.keys()), dtype=float)
speed_of_sound = pd.Series(index=sorted(uj_vs_s.keys()), dtype=float)
fluct = pd.Series(index=sorted(uj_vs_s.keys()), dtype=float)

for s, uj in zip(Varr, eff_mass.index):
    
    size = 100  # number of lattice sites
    qarr = lattice.quasimomenta(size)  # quasimomenta
    n = 0  # band index
    xsampl = np.linspace(-1.5, 1.5, 200)  # real-space position
    
    Er=lattice.recoilenergy(d_latt,m)
    # print(f"Recoil energy of lattice Er= {Er/h/1e3:.3f} kHz")
    
    # U, J = [], []
    e, bs = lattice.eigenproblem(s, qarr, bands=n)
    U=gHe*(lattice.hubbardU(xsampl, bs, scale=d_latt))**3 # cubic power for 3D problem
    J=Er*lattice.hubbardJ(e, d=1)
    
    # print(f"On-site interaction U = {U/h:.3f} Hz")
    # print(f"Tunnelling energy J = {J/h:.3f} Hz")
    # print(f"Ratio U/J = {U/J:.2f}")
    
    #w = lattice.wannier(xsampl, bs)
    #plt.figure()
    #plt.plot(xsampl,w**3)
    #print(np.max(w**3))
    
    m_eff=hbar**2/(2*J*d_latt**2)
    # print("Ratio m_eff/m = {:.3f}".format(m_eff/m))
    
    # f_site=lattice.sitefreq(s*Er, d_latt, m)
    # print("Site frequency {:.3f} kHz".format(f_site/1e3))
    
    eff_mass[uj] = m_eff
    if u_c > uj:
        dens[uj] = (1 - uj / u_c)**(2*beta)
    else:
        dens[uj] = 1e-10
    comp = 1/(U * dens[uj])
    speed_of_sound[uj] = 1/np.sqrt(comp * eff_mass[uj])
    
    
    fluct[uj] = (dens[uj] * eff_mass[uj] * speed_of_sound[uj])**2
    
#%%

s = np.linspace(5, 15)
uj = pd.DataFrame(index=s, columns=['U', 'J', 'U/J'], dtype=float)

for s_ in s:
    
    e, bs = lattice.eigenproblem(s_, qarr, bands=n)
    U=gHe*(lattice.hubbardU(xsampl, bs, scale=d_latt))**3 # cubic power for 3D problem
    J=Er*lattice.hubbardJ(e, d=1)
    
    uj.loc[s_] = pd.Series({'U': U / Er, 'J': J / Er, 'U/J': U / J})
    


#%%
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].semilogy(
    s,
    uj['U'],
    label=r'$U$'
    )
axs[0].semilogy(
    s,
    uj['J'],
    label=r'$J$'
    )
axs[1].plot(
    s,
    uj['U/J'],
    label=r'$U/J$',
    color='g'
    )
axs[1].scatter(
    sorted(uj_vs_s.values()),
    sorted(uj_vs_s.keys()),
    marker='x',
    label='Exp',
    color='k'
    )
axs[-1].set_xlabel(r'$s\ [E_r]$')
axs[0].set_ylabel(r'$E\ [E_r]$')
axs[1].set_ylabel(r'$U/J$')
for ax in axs:
    ax.legend()
    ax.set_axisbelow(True)    
    ax.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
    ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
    ax.grid(visible=True)
plt.show()

    
#%%

# Plot results as a function of U/J:
fig, axs = plt.subplots(4, 2, sharex='col', sharey='row')

for col_idx, x_ax in enumerate([Varr, eff_mass.index]):
    axs[0][col_idx].plot(
        x_ax,
        eff_mass/m,
        label=r'$m*/m = \hbar ^2 / (2 J d_{\mathrm{latt}}^2 m)$'
        )
    axs[1][col_idx].plot(
        x_ax,
        speed_of_sound,
        label=r'$c = 1 / \sqrt{\kappa m*} \propto \sqrt{U N_0 / (N m*)}$'
        )
    axs[2][col_idx].plot(
        x_ax,
        dens,
        label=r'$n_0 \propto (1-u/u_c)^{2\beta}$'
        )
    axs[3][col_idx].plot(
        x_ax,
        fluct/fluct.max(),
        label=r'$\Delta N_0^2 \propto (mc)^2$'
        )
    if col_idx == 0:
        axs[-1][col_idx].set_xlabel('s')
        axs[0][col_idx].set_ylabel('Eff. mass ratio')
        axs[1][col_idx].set_ylabel('Speed of sound')
        axs[2][col_idx].set_ylabel('Condensate density')
        axs[3][col_idx].set_ylabel('Fluctuations')
    elif col_idx == 1:
        axs[-1][col_idx].set_xlabel('U/J')
for ax in axs.flatten():
    ax.legend()
    ax.set_axisbelow(True)    
    ax.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
    ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
    ax.grid(visible=True)

plt.show()


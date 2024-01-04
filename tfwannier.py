#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:25:49 2023

@author: jp
"""


# Standard library imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lattice
from scipy.fft import fft, fftfreq

# Plot setup:
textprops = {"ha": "right", "va": "top"}

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

#%%

# Calculate Wannier functions:

Varr = sorted(uj_vs_s.values())  # lattice depth in units of the recoil energy
size = 100  # number of lattice sites
qarr = lattice.quasimomenta(size)  # quasimomenta
n = 0  # band index
Npoint=500
xrange=100
xarr = np.linspace(-xrange/2, xrange/2 ,Npoint)  # real-space position

wannier_functions = []
for V in Varr:
    _, bs = lattice.eigenproblem(V, qarr, bands=n, kohns_phase=True)
    wf = lattice.wannier(xarr, bs)
    wannier_functions.append(np.real(wf))

dx = xarr[1] - xarr[0]



# Calculate TF:

T=xrange/Npoint; # sampling distance in position space
karr = fftfreq(Npoint, T)[:Npoint//2] # generating the array in the momentum space for FT calculation

FTwannier_functions = []
for i in range(len(Varr)):
    FTwannier_functions.append(fft(wannier_functions[i]))
    FTwannier_functions[i]=FTwannier_functions[i]/np.max(FTwannier_functions[i])
    FTwannier_functions[i]=FTwannier_functions[i][:len(FTwannier_functions[i])//2]
    FTwannier_functions[i]=np.concatenate((FTwannier_functions[i][::-1],FTwannier_functions[i][1:]))

karr=np.concatenate((-karr[::-1],karr[1:])) # to have a symmetric plot centered on k=0


#%%

# Plot mod. square of TF:

# fig, axes = plt.subplots(3, 4, figsize=(16, 2.5), sharey=True)
# for i, ax in enumerate(axes.flatten()):
#     ax.plot(karr, np.abs(FTwannier_functions[i])**2)#,label='$s={}E_r$'.format(Varr[i]))
#     # ax.plot(karr,np.exp(-karr**2*(4/np.sqrt(Varr[i]))),'--')#label='Approx. Gauss. $s={}E_r$'.format(Varr[0]))
#     # ax.set_xlim(0,3)
#     ax.set_xlabel('$k$ [$k_d$]')
#     ax.grid()
#     ax.text(.97, .9, f"U/J = {s_vs_uj[Varr[i]]}", textprops, transform=ax.transAxes)
# axes.flatten()[0].set_ylabel("FT wannier density")
# plt.subplots_adjust(wspace=.1)
# #plt.ylabel('Density [a.u.]')

# plt.show()




#%%

# Calculate quantities:

dk = karr[1] - karr[0]

k_bec_idxs = np.where([i.is_integer() for i in karr])[0]

pw = 0.15  # k_d

pw_dk = int(np.round(pw / dk))

k_bec_idxs_int = np.array([np.arange(i - pw_dk, i + pw_dk + 1) for i in k_bec_idxs])

print(karr[k_bec_idxs_int])

# print(karr[k_bec_idxs])

fbz_idxs = [i for i in range(len(karr)) if -0.5 <= karr[i] < 0.5]

n_0 = np.zeros_like(Varr)
n_bec = np.zeros_like(Varr)
n_fbz = np.zeros_like(Varr)
n_tot = np.zeros_like(Varr)
f_c_fbz = np.zeros_like(Varr)
f_c_tot = np.zeros_like(Varr)

# N_0:

for uj_idx, _ in enumerate(Varr):

    n_0[uj_idx] = sum(dk*abs(FTwannier_functions[uj_idx][k_bec_idxs_int[2]])**2)**3

# N_BEC:
    
n_max = int(max(karr))

miller_indices = pd.MultiIndex.from_product([
    [-n_max + l for l in range(2 * n_max + 1)],
    [-n_max + l for l in range(2 * n_max + 1)],
    [-n_max + l for l in range(2 * n_max + 1)]
    ]).to_numpy()

miller_indices = miller_indices[np.where([abs(h) + abs(k) + abs(l) <= n_max for (h, k, l) in miller_indices])]

mapping = {k: v for k, v in enumerate(k_bec_idxs)}

for uj_idx, _ in enumerate(Varr):
    
    n_bec[uj_idx] = 0
    
    for (h, k, l) in miller_indices:

        h_idx = k_bec_idxs_int[h]
        k_idx = k_bec_idxs_int[k]
        l_idx = k_bec_idxs_int[l]
        
        if uj_idx == 0: print((karr[h_idx], karr[k_idx], karr[l_idx]))

        n_bec[uj_idx] += (
            sum(dk*abs(FTwannier_functions[uj_idx][h_idx])**2)
            * sum(dk*abs(FTwannier_functions[uj_idx][k_idx])**2)
            * sum(dk*abs(FTwannier_functions[uj_idx][l_idx])**2))
                
                


    n_fbz[uj_idx] = (sum(dk*abs(FTwannier_functions[uj_idx][fbz_idxs])**2))**3
    n_tot[uj_idx] = (sum(dk*abs(FTwannier_functions[uj_idx][:])**2))**3
    
    
# Normalize:
# n_0 = n_0 / n_tot * 1
# n_bec = n_bec / n_tot * 1
# n_fbz = n_fbz / n_tot * 1
# n_tot = n_tot / n_tot * 1

for uj_idx, _ in enumerate(Varr):

   f_c_fbz[uj_idx] = n_0[uj_idx] / n_fbz[uj_idx]
   f_c_tot[uj_idx] = n_bec[uj_idx] / n_tot[uj_idx]
   
#%%

# Plot results as a function of U/J:

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(
    sorted(uj_vs_s.keys()),
    n_0,
    label=r'$N_0$'
    )
axs[0].plot(
    sorted(uj_vs_s.keys()),
    n_bec,
    label=r'$N_{\mathrm{BEC}}$'
    )
axs[1].plot(
    sorted(uj_vs_s.keys()),
    n_fbz,
    label=r'$N_{\mathrm{FBZ}}$'
    )
axs[1].plot(
    sorted(uj_vs_s.keys()),
    n_tot,
    label=r'$N_{\mathrm{tot}}$'
    )
axs[2].plot(
    sorted(uj_vs_s.keys()),
    f_c_fbz,
    label=r'$N_0 / N_{\mathrm{FBZ}}$'
    )
axs[2].plot(
    sorted(uj_vs_s.keys()),
    f_c_tot,
    label=r'$N_{\mathrm{BEC}} / N_{\mathrm{tot}}$'
    )
for idx, ax in enumerate(axs):
    if idx==len(axs)-1:
        ax.set_xlabel('U/J')
        ax.set_ylabel(r'$f_c$')
    
    else:
        ax.set_ylabel(r'$\int\mathrm{d}\mathbf{k} |\omega (\mathbf{k})|^2$')
    ax.legend()
    ax.set_axisbelow(True)    
    ax.xaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
    ax.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.3)
    ax.grid(visible=True)

plt.show()


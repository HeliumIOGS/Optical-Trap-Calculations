#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:10:59 2024

@author: jp

Define parameters for optical lattice so that they are the same in all
calculations.
"""

# Standard library imports:
import numpy as np
import pandas as pd

# Local imports:
import lattice
from atom import Helium
import phys_constants

# Physical constants:
h = phys_constants.h
hbar = phys_constants.hbar
gHe = phys_constants.gHe
d_latt = phys_constants.d_latt
he = Helium()
m = he.mass
beta = 0.3485
u_c = 26
Er = lattice.recoilenergy(d_latt,m)

# Set up lattice parameters:
n = 0  # Band index
xarr = np.linspace(-1.5, 1.5, 200)  # Real-space position
size = 100  # Number of lattice sites
qarr = lattice.quasimomenta(size)  # Quasimomenta
sarr = np.linspace(5, 14.2, 100) # Test values for lattice amplitude

""" Test that lattice amplitude correctly reproduces experimental values"""

# Data input (exp. values of s used for data acquisition):
uj_exp = pd.Series({
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
    })

# Calculate U/J values with lattice script functions:
uj = pd.DataFrame(index=sarr, columns=['U', 'J', 'U/J'], dtype=float)

for s in sarr:

    # Calculate U and J from s:
    e, bs = lattice.eigenproblem(s, qarr, bands=n)
    U = gHe * (lattice.hubbardU(xarr, bs, scale=d_latt))**3  # Cubic power for 3D problem
    J = Er * lattice.hubbardJ(e, d=1)
    uj.loc[s] = pd.Series({'U': U, 'J': J, 'U/J': U / J})

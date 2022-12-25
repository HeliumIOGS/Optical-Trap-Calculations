# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:52:27 2017

@author: Marco
"""

import scipy.constants as cnst
from numpy import pi

c = cnst.c
e = cnst.elementary_charge
epsilon0 = cnst.epsilon_0
mu0 = cnst.mu_0
h = cnst.h
hbar = cnst.hbar
uma = cnst.u
g = cnst.g
kb = cnst.Boltzmann
a0 = 0.529*1e-10
aHe = 142.0*a0
m = 4*uma
gHe = 4*pi*hbar**2*aHe/m
lambda_latt = 1550*1e-9
d_latt = lambda_latt/2
krec = 2*pi/lambda_latt
Erec = hbar**2 * krec**2 / (2*m)
vrec = 1e2*hbar*krec/m #cm/s
k_latt = 2*pi/d_latt
v_latt = 1e2*hbar*k_latt/m #cm/s
    

  
#if __name__ == '__main__':
#     phys_constants()
#     print(elec)

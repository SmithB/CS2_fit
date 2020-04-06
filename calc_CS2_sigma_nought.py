#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:11:29 2019

@author: ben
"""

import numpy as np

def calc_CS2_sigma0(D):
    wvl = 0.022084; #m
    baseline = 1.1676; #(m)
    pi=3.1415926535;
    alpha = .973*wvl*(D.phase+2*pi*D.ambiguity)/(2*pi*baseline); #alpha = inferred angle (rad)  

    alpha_3db=1.1992*pi/180;
    sigma_beam=alpha_3db*np.sqrt(-1/(2*np.log(10^-.3)));

    G=np.exp(-(alpha**2/2/sigma_beam**2));

    PR=D.range_surf**2;

    Pcorr=D.power/G**4.*PR;
    return Pcorr
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:03:20 2020

@author: ben
"""


def calc_tide(D, EPSG):

    # per wikipedia, the mjd epoch is Nov 17 1858 -> 678942
    matlab_mjd_epoch = 678942.
    # conversion for matlab-> year is :  year = (t-730486.)/365.25+2000.
    # so t_matlab=(year-2000)*365.25+730468.
    # subtract the mjd epoch to convert to mjd
    MJD_time = (D.time-2000)*365.25+730468.-matlab_mjd_epoch

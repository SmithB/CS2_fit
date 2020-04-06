#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:19:59 2020

@author: ben
"""


from smooth_xytb_fit import smooth_xytb_fit
import pointCollection as pc
import matplotlib.pyplot as plt
import numpy as np
import pointCollection as pc
from LSsurf.read_CS2_data import read_cs2_data

dem_file='/Volumes/insar9/ben/REMA/DEM_500m_filled/mosaic_500m_dem_filled.tif'


index_files={'POCA':['/Volumes/insar6/ben/Cryosat/POCA_h5_C/AA_REMA_v1_sigma4_Jan2020/GeoIndex.h5', \
                     '/Volumes/insar6/ben/Cryosat/POCA_h5_C/AA_REMA_v1_sigma4/GeoIndex.h5'], \
             'swath':['/Volumes/insar6/ben/Cryosat/SW_h5_C/AA_REMA_v1_sigma4_Jan2020/GeoIndex.h5', \
                      '/Volumes/insar6/ben/Cryosat/SW_h5_C/AA_REMA_v1_sigma4/GeoIndex.h5'] }


Wxy=4.e4

xy0=np.array([-440000., -560000.])
XR=xy0[0]+np.array([-0.5, 0.5])*Wxy
YR=xy0[1]+np.array([-0.5, 0.5])*Wxy

TR=np.array([2010, 2018])
if True:
    data=read_cs2_data(xy0, Wxy, index_files, apply_filters=True, dem_file=None, dem_tol=10)
    #TR=np.array([2010, 2015])
    #D=data[0:data.size:2]
    D=data.copy()
    
    D.assign({'z':D.h, 'sigma':np.sqrt(0.2**2+0.5*(D.swath>0.5)), 
                  'sigma_corr': 2*(D.swath>0.5)})
    D_PC=D[D.swath==0]
    D_sw=D[D.swath==1]
            
    orb_dict=pc.bin_rows(np.c_[D_sw.abs_orbit])
    D_sw_sub=pc.data().from_list([D_sw[orb_dict[key]].blockmedian(400) for key in orb_dict])
    D=pc.data().from_list([D_PC, D_sw_sub])
    
    #TR=np.mean(TR)+np.diff(TR)*np.array([-0.125, 0.125])
    D.index((D.time>TR[0])&(D.time < TR[1]))
    #E_RMS0={'d2z0_dx2':200000./3000/3000, 'd3z_dx2dt':6000./3000/3000, 'd2z_dxdt':20000/3000, 'd2z_dt2':5000}
    E_RMS0={'d2z0_dx2':200000./3000/3000, 'd3z_dx2dt':1000./3000/3000, 'd2z_dxdt':6000/3000, 'd2z_dt2':5000}
    KWargs={'data':D,
     'W': {'x':Wxy, 'y':Wxy, 't':np.diff(TR)},
     'ctr':{'x':XR.mean(), 'y':YR.mean(), 't': TR.mean()},
     'spacing':{'z0':500.,'dz':2000, 'dt':0.25},
     'E_RMS':E_RMS0, 
     'E_RMS_d2x_PS_bias':1.e-7,
     'E_RMS_PS_bias':1,
     'verbose': True, 
     'max_iterations':4,
     'bias_params':['abs_orbit','swath'],
     'bias_filter':lambda D:D[D.swath>0.5]}
    S=smooth_xytb_fit(**KWargs)


#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:12:48 2019

@author: ben
"""
import numpy as np
import matplotlib.pyplot as plt
from LSsurf.smooth_xytb_fit import smooth_xytb_fit
from LSsurf.assign_firn_correction import assign_firn_correction
from LSsurf.reread_data_from_fits import reread_data_from_fits
import pointCollection as pc
from LSsurf.read_CS2_data import read_cs2_data

import os
import h5py
import sys
import glob
import re

def get_SRS_proj4(hemisphere):
    if hemisphere==1:
        return '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    else:
        return '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

def save_fit_to_file(S,  filename, sensor_dict=None, dzdt_lags=None):
    if os.path.isfile(filename):
        os.remove(filename)
    with h5py.File(filename,'w') as h5f:
        h5f.create_group('/data')
        for key in S['data'].fields:
            h5f.create_dataset('/data/'+key, data=getattr(S['data'], key))
        h5f.create_group('/meta')
        h5f.create_group('/meta/timing')
        for key in S['timing']:
            h5f['/meta/timing/'].attrs[key]=S['timing'][key]
        if sensor_dict is not None:
            h5f.create_group('meta/sensors')
            for key in sensor_dict:
                h5f['/meta/sensors'].attrs['sensor_%d' % key]=sensor_dict[key]
        # this is how far we'll get if we're just in 'edit only' mode
        if 'dz' not in S['m']:
            return
        h5f.create_group('/dz')
        for ii, name in enumerate(['y','x','t']):
            h5f.create_dataset('/dz/'+name, data=S['grids']['dz'].ctrs[ii])
        h5f.create_dataset('/dz/dz', data=S['m']['dz'])
        h5f.create_dataset('/dz/count', data=S['m']['count'])
        h5f.create_group('/z0')
        for ii, name in enumerate(['y','x']):
            h5f.create_dataset('/z0/'+name, data=S['grids']['z0'].ctrs[ii])
        h5f.create_dataset('/z0/z0', data=S['m']['z0'])
        for lag in dzdt_lags:
            this_name='dzdt_lag%d' % lag
            h5f.create_dataset('/dz/'+this_name, data=S['m'][this_name])
        h5f.create_group('/dz/center_average')
        h5f.create_dataset('/dz/center_average/dz', data=S['m']['dz_bar'])
        for lag in dzdt_lags:
            this_name='dzdt_bar_lag%d' % lag
            h5f.create_dataset('/dz/center_average/'+this_name, data=S['m'][this_name])
        h5f.create_group('/RMS')
        for key in S['RMS']:
            h5f.create_dataset('/RMS/'+key, data=S['RMS'][key])
        h5f.create_group('E_RMS')
        for key in S['E_RMS']:
            h5f.create_dataset('E_RMS/'+key, data=S['E_RMS'][key])
        for key in S['m']['bias']:
            h5f.create_dataset('/bias/'+key, data=S['m']['bias'][key])
        if 'slope_bias' in S['m']:
            sensors=np.array(list(S['m']['slope_bias'].keys()))
            h5f.create_dataset('/slope_bias/sensors', data=sensors)
            x_slope=[S['m']['slope_bias'][key]['slope_x'] for key in sensors]
            y_slope=[S['m']['slope_bias'][key]['slope_y'] for key in sensors]
            h5f.create_dataset('/slope_bias/x_slope', data=np.array(x_slope))
            h5f.create_dataset('/slope_bias/y_slope', data=np.array(y_slope))
        if 'PS_bias' in S['m']:
            for ii, name in enumerate(['y','x']):
                h5f.create_dataset('/PS_bias/'+name, data=S['grids']['dz'].ctrs[ii])
            h5f.create_dataset('/PS_bias/PS_bias', data=S['m']['PS_bias'])
    return

def save_errors_to_file( S, filename, sensor_dict=None):
    with h5py.File(filename,'r+') as h5f:
        h5f.create_group('/dz/sigma/')
        for ii, name in enumerate(['y','x','t']):
            h5f.create_dataset('/dz/sigma/'+name, data=S['grids']['dz'].ctrs[ii])
        h5f.create_dataset('/dz/sigma/dz', data=S['E']['dz'])
        h5f.create_group('/z0/sigma')
        for ii, name in enumerate(['y','x']):
            h5f.create_dataset('/z0/sigma/'+name, data=S['grids']['z0'].ctrs[ii])
        h5f.create_dataset('/z0/sigma/z0', data=S['E']['z0'])
        for key in S['E']['bias']:
            h5f.create_dataset('/bias/sigma/'+key, data=S['E']['bias'][key])
        if 'slope_bias' in S['E']:
            sensors=np.array(list(S['E']['slope_bias'].keys()))
            h5f.create_dataset('/slope_bias/sigma/sensors', data=sensors)
            x_slope=[S['E']['slope_bias'][key]['slope_x'] for key in sensors]
            y_slope=[S['E']['slope_bias'][key]['slope_y'] for key in sensors]
            h5f.create_dataset('/slope_bias/sigma/x_slope', data=np.array(x_slope))
            h5f.create_dataset('/slope_bias//sigma/y_slope', data=np.array(y_slope))
    return
 

def fit_CS2(xy0, Wxy=4e4, E_RMS={}, t_span=[2003, 2020], spacing={'z0':2.5e2, 'dz':5.e2, 'dt':0.5},  \
            hemisphere=1, reference_epoch=None, reread_dirs=None, max_iterations=5, N_subset=None, Edit_only=False, \
            sensor_dict={}, out_name=None, replace=False, DOPLOT=False, spring_only=False, \
            geoid_file=None, mask_file=None, \
            calc_error_file=None, DEM_file=None, DEM_tol=None):
    """
        Wrapper for smooth_xytb_fit that can find data and set the appropriate parameters
    """
    print("fit_CS2: working on %s" % out_name)
    baseline_code={'C':2, 'D':3}
    # temporary:
    if hemisphere==-1:
        index_files={'C':{'POCA':['/Volumes/insar6/ben/Cryosat/POCA_h5_C/AA_REMA_v1_sigma4_Jan2020/GeoIndex.h5', \
                     '/Volumes/insar6/ben/Cryosat/POCA_h5_C/AA_REMA_v1_sigma4/GeoIndex.h5'], \
                     'swath':['/Volumes/insar6/ben/Cryosat/SW_h5_C/AA_REMA_v1_sigma4_Jan2020/GeoIndex.h5', \
                              '/Volumes/insar6/ben/Cryosat/SW_h5_C/AA_REMA_v1_sigma4/GeoIndex.h5'] },
                     'D':{'POCA': glob.glob('/Volumes/ice2/ben/CS2_data/AA/retrack/2*/index_POCA.h5'),
                          'swath':[glob.glob('/Volumes/ice2/ben/CS2_data/AA/retrack/2*/index_sw.h5')]}
                     }
    if hemisphere==1:
        index_files={'C':{'POCA':[glob.glob('/Volumes/ice2/ben/CS2_data/GL/retrack_BC/2*/index_POCA.h5')],
                          'swath':[glob.glob('/Volumes/ice2/ben/CS2_data/GLretrack_BC/2*/index_SW.h5')]},
                    'D':{'POCA':[glob.glob('/Volumes/ice2/ben/CS2_data/GL/retrack/2*/index_POCA.h5')],
                          'swath':[glob.glob('/Volumes/ice2/ben/CS2_data/GLretrack/2*/index_SW.h5')]}
                    }

    compute_E=False
    # set defaults for E_RMS, then update with input parameters
    E_RMS0={'d2z0_dx2':200000./3000/3000, 'd3z_dx2dt':1000./3000/3000, 'd2z_dxdt':6000/3000, 'd2z_dt2':5000}
    E_RMS0.update(E_RMS)

    W={'x':Wxy, 'y':Wxy,'t':np.diff(t_span)}
    ctr={'x':xy0[0], 'y':xy0[1], 't':np.mean(t_span)}
    if out_name is not None:
        try:
            out_name=out_name %(xy0[0]/1000, xy0[1]/1000)
        except:
            pass
    
    if calc_error_file is not None:
        # get xy0 from the filename
        re_match=re.compile('E(.*)_N(.*).h5').search(calc_error_file)
        xy0=[float(re_match.group(ii))*1000 for ii in [1, 2]]
        data, sensor_dict=reread_data_from_fits(xy0, Wxy, [os.path.dirname(calc_error_file)])
        compute_E=True
        max_iterations=0
        N_subset=None
    elif reread_dirs is None:
        data=[]
        for baseline in ['C','D']:

            data += read_cs2_data(xy0, Wxy, index_files[baseline],
                                  apply_filters=True, DEM_file=DEM_file,
                                  dem_tol=50)
            #'sigma':np.sqrt(0.2**2+0.5*(data.swath>0.5))
            data[-1].assign({'z':data[-1].h, \
                  'sigma_corr': 2*(data.swath[-1]>0.5),
                  'baseline': baseline_code[baseline]+np.zeros_like(data[-1].swath)})
        if len(data) > 1:
            data[0].index(~np.in1d(data[0].abs_orbit), np.unique(data[1].abs_orbit))
            data = pc.data().from_list(data)
        else:
            data=data[0]
        D_PC=data[data.swath==0]
        D_sw=data[data.swath==1]

        orb_dict=pc.bin_rows(np.c_[D_sw.abs_orbit])
        D_sw_sub=pc.data().from_list([D_sw[orb_dict[key]].blockmedian(400) for key in orb_dict])
        data=pc.data().from_list([D_PC, D_sw_sub])
    
        data.index((data.time>t_span[0]) & (data.time < t_span[1]))
    else:
        data, sensor_dict = reread_data_from_fits(xy0, Wxy, reread_dirs, template='E%d_N%d.h5')
        N_subset=None
    if reference_epoch is None:
        reference_epoch=np.ceil(len(np.arange(t_span[0], t_span[1], spacing['dt']))/2).astype(int)

    for field in data.fields:
        setattr(data, field, getattr(data, field).astype(np.float64))

    if DEM_file is not None:
        DEM=pc.grid.data().from_geotif(DEM_file, \
              bounds=[[xy0[0]-W['x']/1.8, xy0[0]+W['x']/1.8],\
                      [xy0[1]-W['y']/1.8, xy0[1]+W['y']/1.8]])
        data.assign({'DEM':DEM.interp(data.x, data.y)})

    data.index((data.swath==0) | (np.mod(np.arange(0, data.size), 2)==0))

    # run the fit
    KWargs={'data':data,
     'W': W,
     'ctr':ctr,
     'spacing':spacing,
     'E_RMS':E_RMS0, 
     'E_RMS_d2x_PS_bias':1.e-7,
     'E_RMS_PS_bias':1,
     'compute_E':compute_E,
     'verbose': True, 
     'max_iterations':max_iterations,
     'N_subset':N_subset,
     'bias_params':['abs_orbit','swath'],
     'DEM_tol':DEM_tol,
     'bias_filter':lambda D:D[D.swath>0.5]}
    #print("WARNING WARNING WARNING::::SETTING BIAS_PARAMS AND PS_BIAS_RMS TO NONE!!!!!")
    #KWargs['bias_params']=None
    #KWargs['E_RMS_d2x_PS_bias']=None
    S=smooth_xytb_fit(**KWargs)

    if out_name is not None:
        if calc_error_file is None:
            save_fit_to_file(S, out_name, sensor_dict, dzdt_lags=S['dzdt_lags'])
        else:
            save_errors_to_file(S, out_name, sensor_dict)
    print("fit_CS2: done with %s" % out_name)
    return S, data, sensor_dict

def main(argv):
    # account for a bug in argparse that misinterprets negative agruents
    for i, arg in enumerate(argv):
        if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg

    import argparse
    parser=argparse.ArgumentParser(description="function to fit icebridge data with a smooth elevation-change model", \
                                   fromfile_prefix_chars="@")
    parser.add_argument('xy0', type=float, nargs=2, help="fit center location")
    parser.add_argument('--Width','-W',  type=float, help="fit width")
    parser.add_argument('--time_span','-t', type=str, help="time span, first year,last year AD (comma separated, no spaces)")
    parser.add_argument('--grid_spacing','-g', type=str, help='grid spacing:DEM (meters),dh maps xy (meters),dh_maps time (years): comma-separated, no spaces', default='250.,4000.,1.')
    parser.add_argument('--Hemisphere','-H', type=int, default=1, help='hemisphere: -1=Antarctica, 1=Greenland')
    parser.add_argument('--base_directory','-b', type=str, help='base directory')
    parser.add_argument('--out_name', '-o', type=str, help="output file name")
    parser.add_argument('--centers', action="store_true")
    parser.add_argument('--edges', action="store_true")
    parser.add_argument('--corners', action="store_true")
    parser.add_argument('--E_d2zdt2', type=float, default=5000)
    parser.add_argument('--E_d2z0dx2', type=float, default=0.02)
    parser.add_argument('--E_d3zdx2dt', type=float, default=0.0003)
    parser.add_argument('--data_gap_scale', type=float,  default=2500)
    parser.add_argument('--mask_file', type=str)
    parser.add_argument('--geoid_file', type=str)
    parser.add_argument('--DEM_file', type=str)
    parser.add_argument('--DEM_tol', type=float, default=10)
    parser.add_argument('--calc_error_file','-c', type=str)
    args=parser.parse_args()

    args.grid_spacing = [np.float(temp) for temp in args.grid_spacing.split(',')]
    args.time_span = [np.float(temp) for temp in args.time_span.split(',')]

    spacing={'z0':args.grid_spacing[0], 'dz':args.grid_spacing[1], 'dt':args.grid_spacing[2]}
    E_RMS={'d2z0_dx2':args.E_d2z0dx2, 'd3z_dx2dt':args.E_d3zdx2dt, 'd2z_dxdt':args.E_d3zdx2dt*args.data_gap_scale,  'd2z_dt2':args.E_d2zdt2}

    reread_dirs=None
    if args.centers:
        dest_dir = args.base_directory+'/centers'
    if args.edges or args.corners:
        reread_dirs=[args.base_directory+'/centers']
        dest_dir = args.base_directory+'/edges'
    if args.corners:
        reread_dirs += [args.base_directory+'/edges']
        dest_dir = args.base_directory+'/corners'

    if args.calc_error_file is not None:
        dest_dir=os.path.dirname(args.calc_error_file)
        # get xy0 from the filename
        re_match=re.compile('E(.*)_N(.*).h5').search(args.calc_error_file)
        args.xy0=[float(re_match.group(ii))*1000 for ii in [1, 2]]

    if not os.path.isdir(args.base_directory):
        os.mkdir(args.base_directory)
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    if args.out_name is None:
        args.out_name=dest_dir + '/E%d_N%d.h5' % (args.xy0[0]/1e3, args.xy0[1]/1e3)

    if args.DEM_tol is not None and args.DEM_file is None:
        print("Must specify a DEM if DEM_tol is specified")
        sys.exit(1)

    fit_CS2(args.xy0, Wxy=args.Width, E_RMS=E_RMS, t_span=args.time_span, spacing=spacing, \
            hemisphere=args.Hemisphere, reread_dirs=reread_dirs, \
            out_name=args.out_name, DOPLOT=False, \
            mask_file=args.mask_file, geoid_file=args.geoid_file, \
                DEM_file=args.DEM_file, DEM_tol=args.DEM_tol, \
                calc_error_file=args.calc_error_file)

if __name__=='__main__':
    main(sys.argv)

# -440000. -560000. -W 40000 -t 2010,2019 -g 500,2000,1 -H -1 -b /Volumes/ice2/ben/CS2_fit/AA -o test1_halfdata.h5 --centers

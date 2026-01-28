#!/usr/bin/env python3
"""
The original source scattering data files are from the

Robin Ekelund, Manfred Brath, Jana Mendrok, & Patrick Eriksson. (2020). 
ARTS Microwave Single Scattering Properties Database (1.1.0) [Data set]. 
Zenodo. https://doi.org/10.5281/zenodo.4646605

See also:
Eriksson, P., Ekelund, R., Mendrok, J., Brath, M., Lemke, O., and Buehler, S. A.:
A general database of hydrometeor single scattering properties at microwave and
sub-millimetre wavelengths, Earth Syst. Sci. Data, 10, 1301–1326, 
https://doi.org/10.5194/essd-10-1301-2018, 2018. 

prepared using the ARTS SSDB python interface.


Script for preparing and processing scattering data for ARTS (Atmospheric Radiative Transfer Simulator).
This script processes scattering data files by:
1. Renaming meta files to follow a consistent naming convention
2. Thinning the phase matrix data by sampling at specified intervals
   IMPORTANT: By thinning the data, it can happened that ExtMat(0,0) - AbsVec(0) is not
   equal the the integral over the thinned phase function PhaMat(0,0). So you need evenually
   to adjust the check type in ARTS from check_type 'all' to 'sane' at scat_dataCheck() and
   to check level 'none' in scat_dataCheckedCalc() to avoid wrong errors.
   
3. Extending the temperature grid for ice/frozen hydrometeors to prevent extrapolation issues
4. Saving processed data in binary XML format
The script searches for scattering data directories containing '_id' in their names and processes
each item found. For liquid particles, a different sampling step is applied compared to ice particles.
Directory Structure
-------------------
Expected input structure:
    scattering_data/
        <item_id>/
            totally_random/
                *scat_data.xml
                *scat_meta.xml (optional)
                *scat_meta.xml.bin (optional)
Output structure:
    scattering_data/
        *scat_data.xml (processed)
        *.meta.* (copied meta files)
Global Variables
----------------
script_dir : str
    Directory where this script is located
scattering_data_dir : str
    Directory containing scattering data subdirectories
orientation : str
    Particle orientation type (default: 'totally_random')
sample_step : int
    Sampling step for thinning phase matrix data for ice particles (default: 3)
sample_step_liquid : int
    Sampling step for thinning phase matrix data for liquid particles (default: 30)
- The script requires that 180° is present in the zenith angle grid after sampling
- Temperature grid extension is only applied to non-liquid (frozen) hydrometeors
- All processed files are saved in binary XML format for efficiency
- Meta files are copied alongside the main scattering data files

# -*- coding: utf-8 -*-

@author: M. Brath

"""



import os
import shutil
import numpy as np
from pyarts import xml

# ===================================================================================
# %% path / constants
# ===================================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
scattering_data_dir = os.path.join(script_dir, "scattering_data")

orientation='totally_random'

sample_step=3
sample_step_liquid=30


def extend_temperature_grid(ssd_i, T_add=None):
    """
    Extend the temperature grid of a single scattering data object by one point.
    This function adds an additional temperature point to the temperature grid and 
    extends the scattering data arrays (absorption vector, extinction matrix, and 
    phase matrix) by duplicating the values from the last temperature point.
    Parameters
    ----------
    ssd_i : object
        ARTS single scattering data object containing:
        - T_grid : array-like
            Temperature grid values
        - abs_vec_data : ndarray
            Absorption vector data with shape (..., n_temps, ...)
        - ext_mat_data : ndarray
            Extinction matrix data with shape (..., n_temps, ...)
        - pha_mat_data : ndarray
            Phase matrix data with shape (..., n_temps, ...)
    T_add : float, optional
        Specific temperature value to add to the grid. If None, the new 
        temperature is calculated as the maximum temperature plus the mean 
        temperature step. Default is None.
    Returns
    -------
    ssd_i : object
        Modified ARTS single scattering data object with extended temperature grid
        and corresponding scattering data arrays. The new temperature point
        contains duplicated values from the previous last temperature point.
    Notes
    -----
    - The function modifies the input object in-place and returns it.
    - All scattering data arrays are extended by copying values from the 
      second-to-last temperature index to the new last index.
    - The temperature grid is extended by one point at the end.
    """

    
    T_grid=ssd_i.T_grid.value
    T_grid_new=np.zeros(len(T_grid)+1)
    T_grid_new[0:len(T_grid)]=T_grid
    if T_add is not None:
        T_grid_new[-1]=T_add
    else:   
        T_grid_new[-1]=np.max(T_grid)+np.mean(np.diff(T_grid))
    
    ssd_i.T_grid=T_grid_new
        
    shape_abs=list(np.shape(ssd_i.abs_vec_data))
    shape_abs[1]=len(T_grid_new)    
    abs_vec_data_new=np.zeros(tuple(shape_abs))
    abs_vec_data_new[:,0:len(T_grid),:,:,:]=ssd_i.abs_vec_data.value
    abs_vec_data_new[:,-1,:,:,:]=abs_vec_data_new[:,-2,:,:,:]

    ssd_i.abs_vec_data=abs_vec_data_new

    shape_ext=list(np.shape(ssd_i.ext_mat_data))
    shape_ext[1]=len(T_grid_new)    
    ext_mat_data_new=np.zeros(tuple(shape_ext))
    ext_mat_data_new[:,0:len(T_grid),:,:,:]=ssd_i.ext_mat_data.value
    ext_mat_data_new[:,-1,:,:,:]=ext_mat_data_new[:,-2,:,:,:]
    
    ssd_i.ext_mat_data=ext_mat_data_new

    shape_pha=list(np.shape(ssd_i.pha_mat_data))
    shape_pha[1]=len(T_grid_new)    
    pha_mat_data_new=np.zeros(tuple(shape_pha))
    pha_mat_data_new[:,0:len(T_grid),:,:,:,:,:]=ssd_i.pha_mat_data.value
    pha_mat_data_new[:,-1,:,:,:,:,:]=pha_mat_data_new[:,-2,:,:,:,:,:]

    ssd_i.pha_mat_data=pha_mat_data_new
    
    return ssd_i
    

# ===================================================================================
# %% see what we got

items = os.listdir(scattering_data_dir)

scattering_items = [ item for item in items if ('_id' in item.lower() and '.xml' not in item.lower()) ]
print('Found the following scattering data items:')
for item in scattering_items:
    print(' - '+item)
    print('\n')

# ===================================================================================
# %% process scattering data


# loop over all scattering data items
for item in scattering_items:

    item_dir = os.path.join(scattering_data_dir, item, orientation)
    files = os.listdir(item_dir)

    scattering_data_file=[ file for file in files if file.endswith('scat_data.xml') ][0]

    print('Processing item: '+item+' , file: '+scattering_data_file)

    scattering_meta_file=[ file for file in files if file.endswith('scat_meta.xml') ]
    scattering_meta_binfile=[ file for file in files if file.endswith('scat_meta.xml.bin') ]
    
    if len(scattering_meta_file)>0:

        print('Renaming meta file...')

        scat_meta_file_new_name=scattering_meta_file[0].replace('_meta','_data.meta')
    
        #rename scattering meta file from scat_meta.xml to scat_data_meta.xml
        os.rename( os.path.join(item_dir, scattering_meta_file[0]),
                   os.path.join(item_dir, scat_meta_file_new_name) )
    
    if len(scattering_meta_binfile)>0:

        print('Renaming meta bin file...')

        scat_meta_binfile_new_name=scattering_meta_binfile[0].replace('_meta','_data.meta')
    
        #rename scattering meta file from scat_meta.xml to scat_data_meta.xml
        os.rename( os.path.join(item_dir, scattering_meta_binfile[0]),
                   os.path.join(item_dir, scat_meta_binfile_new_name) )
            
        
        
    # open data file
    print('thinning scattering data...')

    scat_data = xml.load(os.path.join(item_dir, scattering_data_file))   
    
    for ssd_i in scat_data[0]:
        
        
                        
        
        if 'liquid' in item.lower():
            ssd_i.za_grid=ssd_i.za_grid[::sample_step_liquid]
            ssd_i.pha_mat_data=ssd_i.pha_mat_data[:,:,::sample_step_liquid,:,:,:,:]
        else:
            ssd_i.za_grid=ssd_i.za_grid[::sample_step]
            ssd_i.pha_mat_data=ssd_i.pha_mat_data[:,:,::sample_step,:,:,:,:]
            
            #Here we extend the temperature grid by one point by copying the last point
            #This is needed for the OEM retrievals as there can be temperature for frozen hydrometeors
            #where the scattering data is missing at the upper end of the temperature grid
            ssd_i=extend_temperature_grid(ssd_i)


        #check that 180 is in ssd_i.za_grid
        if not 180. in ssd_i.za_grid[:]:
            raise ValueError('Use a different sample_step, so that 180 is inside of za_grid')

    #save scattering data file

    print('saving thinned scattering data and copy meta files...')

    scat_data.savexml(os.path.join(scattering_data_dir, scattering_data_file), type='binary')
    
    meta_files = [xx for xx in  os.listdir(item_dir) if '.meta.' in xx ]
    for meta_file in meta_files:
        shutil.copy( os.path.join(item_dir, meta_file),
                     os.path.join(scattering_data_dir, meta_file) )
    
    print('Done with item: '+item+'\n')    

    

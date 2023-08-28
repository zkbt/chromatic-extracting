#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 09:50:41 2023

@author: zashaavery
"""

import chromatic_extracting
from astropy.io import fits
from astropy.table import Table
import os
import numpy as np

## This will be a main.py that will provide some functions for the animation file


# Directory containing FITS files
fits_directory = '/Users/zashaavery/Documents/WASP-94A_fits/ut140801'

## def headers() will print out a list of available headers from the fits files in alphabetical order
def headers():

    # Iterate over all FITS files in the directory
    for filename in os.listdir(fits_directory):
        if filename.endswith('.fits'):
            fits_file = os.path.join(fits_directory, filename)
            
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                
                headers = []
                
                for keyword in header: 
                    headers.append(keyword)
                    
    sorted_headers = list(set(headers))
    sorted_headers.sort()
    print(sorted_headers)

  
## def headers_data will allow the user to input a header name to get the data associated with that header 
def headers_data(header):
    header_upper = header.upper()
    
    data = []
    
    # Iterate over all FITS files in the directory
    for filename in os.listdir(fits_directory):
        if filename.endswith('.fits'):
            fits_file = os.path.join(fits_directory, filename)
            
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                single = hdul[0].header.get(header_upper, None)
                data.append(single)
    
    data = np.array(data)           
    return data

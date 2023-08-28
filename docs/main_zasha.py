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
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from IPython.display import display
import ipywidgets as widgets

## This will be a main.py that will provide some functions for the animation file
# as well as some other useful functions


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

## def sorted_datetime will allow the user to sort their files in terms of observation datetime
def sorted_datetime():
    
    obs_list = []
    
    # Iterate over all FITS files in the directory
    for filename in os.listdir(fits_directory):
        if filename.endswith('.fits'):
            fits_file = os.path.join(fits_directory, filename)
            
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                date_obs = hdul[0].header.get('DATE-OBS', None)
                time_obs = hdul[0].header.get('TIME-OBS', None)
                
                if date_obs and time_obs:
                    obs_datetime_str = f"{date_obs} {time_obs}"
                    obs_datetime = datetime.strptime(obs_datetime_str, '%Y-%m-%d %H:%M:%S.%f')
                    header_data = hdul[0].header.copy()
                    image_data = fits.getdata(fits_file, ext = 0)
                    
                    obs_list.append((filename, obs_datetime, header_data, image_data))
    
    # Sort the obs_list based on observation datetime
    sorted_obs_list = sorted(obs_list, key=lambda x: x[1])
    return sorted_obs_list
        
sorted_list = sorted_datetime()
       
## def sorted_headers_data will do the same as headers_data but sorted 
def sorted_headers_data(sorted_list_headers, header):
    
    header_upper = header.upper()
    
    data = []
    
    # Iterate over all FITS files in the directory
    for filename, obs_datetime, header_data, image_data in sorted_list_headers:
        fits_file = os.path.join(fits_directory, filename)
            
        # Open the FITS file and extract observation date and time
        with fits.open(fits_file) as hdul:
            single = hdul[0].header.get(header_upper, None)
            data.append(single)
    
    data = np.array(data)           
    return data

## def filtered_sorted_list will organize the files for a single chosen object. 
# for example, fits images may include both transits as well as filter images
def filtered_sorted_list(sorted_list_filtered, object_name):
    
    filtered_entries = []

    for entry in sorted_list_filtered:
        filename, obs_datetime, header_data, image_data = entry
        fits_file = os.path.join(fits_directory, filename)

        with fits.open(fits_file) as hdul:
            header_object_name = header_data.get('OBJECT', None)

            if header_object_name == object_name:
                filtered_entries.append(entry)

    return filtered_entries

## def animation will create an interactive graph that allows the user to look through
# the fits files images.       
def animation(sorted_list_animation, *header_names, vmin_initial=1000, vmax_initial=3000, valstep=1, cmap='gray', figsize=(8, 6), current_frame=0):
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    plt.subplots_adjust(right=0.75)  # Adjust to account for diagnostic information

    entry = sorted_list_animation[current_frame]
    filename, obs_datetime, header_data, image_data = entry
    fits_file = os.path.join(fits_directory, filename)

    with fits.open(fits_file) as hdul:
        # Apply a colormap (cmap) and control the strength using vmin and vmax
        im = ax.imshow(image_data, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto')

        title_datetime = obs_datetime
        title = ax.set_title(f"Observation Date and Time: {title_datetime}")

        header_texts = []  # Store the text objects for headers

        for i, header_name in enumerate(header_names):
            header_value = header_data.get(header_name, 'N/A')
            header_text = fig.text(1.025, 0.98 - i * 0.08, f"{header_name.upper()}: {header_value}", transform=ax.transAxes, fontsize=8)
            header_texts.append(header_text)

        # Create sliders for vmin and vmax. Keep in mind, vmin < vmax
        ax_slider_vmin = plt.axes([0.25, 0.25, 0.65, 0.03])
        slider_vmin = Slider(ax_slider_vmin, 'vmin', 0, 5000, valinit=vmin_initial, valstep=valstep)

        ax_slider_vmax = plt.axes([0.25, 0.2, 0.65, 0.03])
        slider_vmax = Slider(ax_slider_vmax, 'vmax', 0, 5000, valinit=vmax_initial, valstep=valstep)

        def update(val):
            vmin_value = slider_vmin.val
            vmax_value = slider_vmax.val
            im.set_clim(vmin=vmin_value, vmax=vmax_value)
            fig.canvas.draw_idle()

        slider_vmin.on_changed(update)
        slider_vmax.on_changed(update)

        # Create a slider for navigation
        ax_slider_frame = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider_frame = Slider(ax_slider_frame, 'Frame', 0, len(sorted_list_animation) - 1, valinit=current_frame, valstep=1)
        
        def update_frame(val):
            nonlocal current_frame
            current_frame = int(slider_frame.val)
            obs_datetime = sorted_list_animation[current_frame][1]
            title.set_text(f"Observation Date and Time: {obs_datetime}")
            
            entry = sorted_list_animation[current_frame]
            filename, obs_datetime, header_data, image_data = entry
            fits_file = os.path.join(fits_directory, filename)
            
            for i, header_name in enumerate(header_names):
                header_value = header_data.get(header_name, 'N/A')
                header_texts[i].set_text(f"{header_name.upper()}: {header_value}")  # Update the header text
                
            with fits.open(fits_file) as hdul:
                im.set_data(image_data)
                fig.canvas.draw_idle()
        
        slider_frame.on_changed(update_frame)
        
        # Display the interactive figure
        plt.show()
  
    
    
    
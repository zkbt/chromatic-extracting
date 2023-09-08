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
from ipywidgets import Button
from skimage.transform import resize  # Use skimage for resizing

# # This will be a main.py that will provide some functions for the animation file
# as well as some other useful functions

class FileManager:
    def __init__(self, fits_directory):
        self.fits_directory = fits_directory

    def headers(self):
    
        """
        Description:
            headers() prints out a list of available headers from the fits files in alphabetical order
        
        Parameters:
            None
        
        Returns:
            str headers
        """
        # Iterate over all FITS files in the directory
        for filename in os.listdir(self.fits_directory):
            if filename.endswith('.fits'):
                fits_file = os.path.join(self.fits_directory, filename)
                
                # Open the fits file. This method ensures it closes directly after use
                with fits.open(fits_file) as hdul:
                    header = hdul[0].header # extract the header
                    
                    headers = [] #empty list to store the headers
                    
                    for keyword in header: 
                        headers.append(keyword) # append the headers into the empty list
                        
        sorted_headers = list(set(headers)) # ensure that headers are not repeated
        sorted_headers.sort() # sort alphabetically
        print(sorted_headers)
    
    
    def headers_data(self, header):
        
        """
        Description: 
            headers_data() allows the user to get the corresponding data to a header
        
        Parameters:
            header(str): the header of the wanted data
        
        Returns:
            arr data
        """
        
        header_upper = header.upper() # capatalize the input (case insensitive)
        
        data = [] # empty list to store data
        
        # Iterate over all FITS files in the directory
        for filename in os.listdir(self.fits_directory):
            if filename.endswith('.fits'):
                fits_file = os.path.join(self.fits_directory, filename)
                    
                # Open the FITS file
                with fits.open(fits_file) as hdul:
                    single = hdul[0].header.get(header_upper, None) # extract the header data for one file
                    data.append(single) # append the data to have header data for all files
        
        data = np.array(data) # list to array        
        return data
    
    
    def sorted_datetime(self):
        
        """
        Description:
            sorted_datetime() allows the user to sort their files in terms of observation date-time
            
        Parameters:
            None
            
        Returns:
            list sorted_obs_list: all of the headers and their data sorted in order of observation date-time
        """
        obs_list = [] # empty list
        
        # Iterate over all FITS files in the directory
        for filename in os.listdir(self.fits_directory):
            if filename.endswith('.fits'):
                fits_file = os.path.join(self.fits_directory, filename)
                
                # Open the FITS file and extract observation date and time
                with fits.open(fits_file) as hdul:
                    date_obs = hdul[0].header.get('DATE-OBS', None)
                    time_obs = hdul[0].header.get('TIME-OBS', None)
                    
                    # Make data and time data in readable format
                    if date_obs and time_obs:
                        obs_datetime_str = f"{date_obs} {time_obs}"
                        obs_datetime = datetime.strptime(obs_datetime_str, '%Y-%m-%d %H:%M:%S.%f')
                        header_data = hdul[0].header.copy() # get header data
                        image_data = fits.getdata(fits_file, ext = 0) # get image data
                        
                        obs_list.append((filename, obs_datetime, header_data, image_data)) # append the listed data for all files
        
        # Sort the obs_list based on observation datetime
        sorted_obs_list = sorted(obs_list, key=lambda x: x[1])
        return sorted_obs_list
    
    
    def sorted_headers_data(self, sorted_list, header):
        
        """
        Definition:
            sorted_headers_data does the same as headers_data() but is sorted by observation date-time
            
        Parameters:
            sorted_list: the sorted date-time list from sorted_datetime()
            header: the header for which data wants to be extracted
            
        Returns:
            arr data
        """
        header_upper = header.upper() # make input case insensitive
        
        data = [] # empty list
        
        # Iterate over all values in sorted_list
        for filename, obs_datetime, header_data, image_data in sorted_list:
            fits_file = os.path.join(self.fits_directory, filename)
                
            # Open the FITS file
            with fits.open(fits_file) as hdul:
                single = hdul[0].header.get(header_upper, None) # extract header data
                data.append(single) # append header data
        
        data = np.array(data) # list to array         
        return data
    
    
    def filtered_sorted_list(self, list_to_filter, object_name):
        
        """
        Definition:
            filtered_sorted_list() organizes the files for one object 
        
        Parameters:
            list_to_filter: the list of FITS files that want to be filtered, need to have same format as sorted_list
            object_name: the name of the object that you want files for
            
        Returns:
            list filtered_entries
        """
        filtered_entries = [] # empty list
    
        # Filter through entires in the list
        for entry in list_to_filter:
            filename, obs_datetime, header_data, image_data = entry
            fits_file = os.path.join(self.fits_directory, filename)
            
            # Open the FITS file
            with fits.open(fits_file) as hdul:
                header_object_name = header_data.get('OBJECT', None) # get data from the OBJECT header
    
                if header_object_name == object_name: 
                    filtered_entries.append(entry) #get data only for given object
    
        return filtered_entries
    
    def separate_files(self, sorted_list):
        
        """
        Definiton: 
            separate_files() will create two separate lists for c1 and c2 files
            
        Paramters:
            sorted_list: the list that the user wants to separate by file ending
            
        Returns:
            list c1_files, c2_files
        """
        c1_files = []
        c2_files = []
        
        for entry in sorted_list:
            filename = entry[0]
            if filename.endswith('c1.fits'):
                c1_files.append(entry)
            elif filename.endswith('c2.fits'):
                c2_files.append(entry)
        
        return c1_files, c2_files
    
    
    def separate_header_data(self, file_list, header):
        
        """
        Definition: 
            separate_header_data() will give the data for any header for the specific file set
            
        Paramters:
            file_list: the file list that you wish to extract the header from
            header: the header you want data for
            
        Returns:
            list header_data_list
        """
        header_data_list = []
        
        for entry in file_list:
            filename = entry[0]
            obs_datetime = entry[1]
            header_data = entry[2]
            
            header_value = header_data.get(header, 'N/A')
            header_data_list.append((filename, obs_datetime, header_value))
        
        return header_data_list
    
    
    def animation(self, sorted_list_animation, *header_names, vmin_initial=1000, vmax_initial=3000, valstep=1, cmap='gray', figsize=(8, 6), current_frame=0, norm = 'linear'):
        
        """
        Definition: 
            def animation() will create an interactive graph that allows the user to look through FITS file images
            
        Parameters: 
            sorted_list_animation: the sorted FITS file list that will be formatted into images
            *headers_name: any headers that the user wants diagnostics from, shown to the right of the image
            vmin_initial/vmax: the setting for the initial slider to influence the effect of colormap (default 1000 and 3000)
            valstep: the index of the images that are looked as the slider progresses (default 1)
            cmap: the colormap for the images (default gray)
            figsize: the size of the printed figure (default (8, 6))
            current_frame: the frame that the animation begins at (default 0)
            norm: changes the scale of the images, options can be seen by running matplotlib.scale.get_scale_names() (default linear)
            
        Results:
            interactive figure
            
        """
        # Initialize figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
        plt.subplots_adjust(right=0.75)  # Adjust to account for diagnostic information
        
        entry = sorted_list_animation[current_frame]
        filename, obs_datetime, header_data, image_data = entry
        fits_file = os.path.join(self.fits_directory, filename)
    
        with fits.open(fits_file) as hdul:
            # Apply a colormap (cmap) and control the strength using vmin and vmax
            im = ax.imshow(image_data, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    
            cbar = plt.colorbar(im, ax=ax)
            
            title_datetime = obs_datetime
            title = ax.set_title(f"Observation Date and Time: {title_datetime}")
    
            header_texts = []  # Store the text objects for headers
    
            for i, header_name in enumerate(header_names):
                header_value = header_data.get(header_name, 'N/A')
                header_text = fig.text(1.25, 0.98 - i * 0.08, f"{header_name.upper()}: {header_value}", transform=ax.transAxes, fontsize=8)
                header_texts.append(header_text)
    
            
            # Create sliders for vmin and vmax. Keep in mind, vmin < vmax
            ax_slider_vmin = plt.axes([0.12, 0.25, 0.65, 0.03])
            slider_vmin = Slider(ax_slider_vmin, 'vmin', 0, 5000, valinit=vmin_initial, valstep=valstep)
    
            ax_slider_vmax = plt.axes([0.12, 0.2, 0.65, 0.03])
            slider_vmax = Slider(ax_slider_vmax, 'vmax', 0, 5000, valinit=vmax_initial, valstep=valstep)
    
            def update(val):
                vmin_value = slider_vmin.val
                vmax_value = slider_vmax.val
                im.set_clim(vmin=vmin_value, vmax=vmax_value)
                fig.canvas.draw_idle()
    
            slider_vmin.on_changed(update)
            slider_vmax.on_changed(update)
    
            # Create a slider for navigation
            ax_slider_frame = plt.axes([0.12, 0.1, 0.65, 0.03])
            slider_frame = Slider(ax_slider_frame, 'Frame', 0, len(sorted_list_animation) - 1, valinit=current_frame, valstep=1)
            
            def update_frame(val):
                nonlocal current_frame
                current_frame = int(slider_frame.val)
                obs_datetime = sorted_list_animation[current_frame][1]
                title.set_text(f"Observation Date and Time: {obs_datetime}")
                
                entry = sorted_list_animation[current_frame]
                filename, obs_datetime, header_data, image_data = entry
                fits_file = os.path.join(self.fits_directory, filename)
                
                for i, header_name in enumerate(header_names):
                    header_value = header_data.get(header_name, 'N/A')
                    header_texts[i].set_text(f"{header_name.upper()}: {header_value}")  # Update the header text
                    
                with fits.open(fits_file) as hdul:
                    im.set_data(image_data)
                    fig.canvas.draw_idle()
            
            slider_frame.on_changed(update_frame)
            
            # Display the interactive figure
            plt.show()
    
def organize_files(source_folder, destination_folder, header_keyword, file_type=None):
    """
    Definition:
        organize_files will organize files into a new folder depending on an inputted header

    Parameters:
        source_folder: the original folder with the unorganized fits files
        destination_folder: the folder where the user wants the organized files
        header_keyword: the header the user wants to organize
        file_type: fits files sometimes have suborganizations such as c1 or c2 (default = None means that fits files will not be separate by file type)

    Results:
        folder with organized files in destination folder
    """
    # Create the destination parent folder if it doesn't exist.
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Create an empty dictionary to store files based on header values.
    files_by_header = {}

    # Iterate through each file in the source folder and group them by header value.
    for filename in os.listdir(source_folder):
        if filename.endswith('.fits'):
            source_path = os.path.join(source_folder, filename)
            with fits.open(source_path) as hdul:
                header = hdul[0].header
                header_value = header.get(header_keyword, 'unknown')

                if header_value not in files_by_header:
                    files_by_header[header_value] = []

                files_by_header[header_value].append((filename, source_path))

    # Organize files into folders based on header values.
    for header_value, files in files_by_header.items():
        header_folder = os.path.join(destination_folder, header_value)
        if not os.path.exists(header_folder):
            os.makedirs(header_folder)

        if file_type is not None:
            # Create a folder for 'file_type' (e.g., 'c1' or 'c2') within the header folder.
            type_folder = os.path.join(header_folder, file_type)
            if not os.path.exists(type_folder):
                os.makedirs(type_folder)

            # Move files to the appropriate folder based on file type.
            for filename, source_path in files:
                if filename.endswith(f'{file_type}.fits'):
                    new_filename = filename
                    destination_path = os.path.join(type_folder, new_filename)
                    with fits.open(source_path) as hdul:
                        hdul.writeto(destination_path, overwrite=True)
        else:
            # Move files to the header folder directly.
            for filename, source_path in files:
                new_filename = filename
                destination_path = os.path.join(header_folder, new_filename)
                with fits.open(source_path) as hdul:
                    hdul.writeto(destination_path, overwrite=True)
                    
def median_image(source_folder, median_name, vmin_initial=1000, vmax_initial=3000, valstep = 1, cmap = 'gray', figsize = (8, 6), norm = 'linear'):
    
    """
    Definition:
        def median_image will display an single image representing the median to a set of FITS image
        
    Paramters: 
        source_folder: the location of the FITS images
        median_name: the name of the image you want to get a median of ('BIAS', 'FLAT')
        vmin_initial/vmax: the setting for the initial slider to influence the effect of colormap (default 1000 and 3000)
        valstep: the index of the images that are looked as the slider progresses (default 1)
        cmap: the colormap for the images (default gray)
        figsize: the size of the printed figure (default (8, 6))
        norm: changes the scale of the images, options can be seen by running matplotlib.scale.get_scale_names() (default linear)
        
    Results:
        a median image
    """
    
    median_pixels = []
    
    for filename in os.listdir(source_folder):
        
        if filename.endswith('.fits'):
            fits_file = os.path.join(source_folder, filename)
            
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                pixel = hdul[0].data
                median_pixels.append(pixel)
                
    med_image = np.median(median_pixels, axis = 0)
    
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    im = ax.imshow(med_image, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)

    cbar = plt.colorbar(im, ax=ax)
    
    title = ax.set_title('Median ' + median_name + ' Image')
    
    # Create sliders for vmin and vmax. Keep in mind, vmin < vmax
    ax_slider_vmin = plt.axes([0.12, 0.25, 0.65, 0.03])
    slider_vmin = Slider(ax_slider_vmin, 'vmin', 0, 5000, valinit=vmin_initial, valstep=valstep)

    ax_slider_vmax = plt.axes([0.12, 0.2, 0.65, 0.03])
    slider_vmax = Slider(ax_slider_vmax, 'vmax', 0, 5000, valinit=vmax_initial, valstep=valstep)

    def update(val):
        vmin_value = slider_vmin.val
        vmax_value = slider_vmax.val
        im.set_clim(vmin=vmin_value, vmax=vmax_value)
        fig.canvas.draw_idle()

    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)
    
    plt.plot()
    
    
def avg_image(source_folder, average_name, vmin_initial=1000, vmax_initial=3000, valstep = 1, cmap = 'gray', figsize = (8, 6), norm = 'linear'):
    
    """
    Definition:
        def average_image will display an single image representing the average to a set of FITS image
        
    Paramters: 
        source_folder: the location of the FITS images
        average_name: the name of the image you want to get a average of ('BIAS', 'FLAT')
        vmin_initial/vmax: the setting for the initial slider to influence the effect of colormap (default 1000 and 3000)
        valstep: the index of the images that are looked as the slider progresses (default 1)
        cmap: the colormap for the images (default gray)
        figsize: the size of the printed figure (default (8, 6))
        norm: changes the scale of the images, options can be seen by running matplotlib.scale.get_scale_names() (default linear)
        
    Results:
        an average image
    """
    
    average_pixels = [] 
    
    for filename in os.listdir(source_folder):
        
        if filename.endswith('.fits'):
            fits_file = os.path.join(source_folder, filename)
            
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                pixel = hdul[0].data
                average_pixels.append(pixel)
                
    avg_image = np.mean(average_pixels, axis = 0)
    
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    im = ax.imshow(avg_image, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)

    cbar = plt.colorbar(im, ax=ax)
    
    title = ax.set_title('Average ' + average_name + ' Image')
    
    # Create sliders for vmin and vmax. Keep in mind, vmin < vmax
    ax_slider_vmin = plt.axes([0.12, 0.25, 0.65, 0.03])
    slider_vmin = Slider(ax_slider_vmin, 'vmin', 0, 5000, valinit=vmin_initial, valstep=valstep)

    ax_slider_vmax = plt.axes([0.12, 0.2, 0.65, 0.03])
    slider_vmax = Slider(ax_slider_vmax, 'vmax', 0, 5000, valinit=vmax_initial, valstep=valstep)

    def update(val):
        vmin_value = slider_vmin.val
        vmax_value = slider_vmax.val
        im.set_clim(vmin=vmin_value, vmax=vmax_value)
        fig.canvas.draw_idle()

    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)
    
    plt.plot()
    

def avg_image_data(source_folder):
    
    """
    Definition:
        def avg_image_data will return the image data for a set of FITS images
        
    Paramters: 
        source_folder: the location of the FITS images
        
    Results:
        arr avg_data
    """
    
    average_pixels = []
    
    for filename in os.listdir(source_folder):
        
        if filename.endswith('.fits'):
            fits_file = os.path.join(source_folder, filename)
            
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                pixel = hdul[0].data
                average_pixels.append(pixel)
                
    avg_data = np.mean(average_pixels, axis = 0)
    return avg_data
    
'''
def bias_histogram(source_folder):
    
    for filename in os.listdir(source_folder):
          
        if filename.endswith('.fits'):
            fits_file = os.path.join(source_folder, filename)
              
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                image_data = hdul[0].data
    
    plt.hist(image_data.flatten(), bins = 'auto')
    plt.plot()
 

bias_file = '/Users/zashaavery/Documents/WASP-94A_fits/organized_files/Bias/Bias_c1/Bias_c1_ccd0001c1.fits'
hdu_list = fits.open(bias_file)
image_data = hdu_list[0].data             
        
plt.hist(image_data.flatten(), bins = 'auto')
plt.plot()
 '''  
 
def read_noise(source_folder):
    
    """
    Definition:
        def read_noise will return the read noise from the bias image 
        
    Paramters: 
        source_folder: the location of the bias images
        
    Results:
        arr read_noise
    """
    
    median_pixels = []
    
    for filename in os.listdir(source_folder):
        
        if filename.endswith('.fits'):
            fits_file = os.path.join(source_folder, filename)
            
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                pixel = hdul[0].data
                median_pixels.append(pixel)
                
    med_image = np.median(median_pixels, axis = 0)
    
    read_noise = np.std(med_image, axis = 0)
    return read_noise
 
def photons_per_pixel(image_folder, B, F, g):
    
    """
    Definition: 
        def photons_per_pixel returns the image data for a set of corrected images
        
    Parameters:
        image_folder: the images you wish to correct
        B: the image data for the median bias images
        F: the image data for the median flat images
        g: the gain for the original images
    """
    
    photons = []
    
    # Ensure that g has the same shape as B, P, and F
    g = resize(g, B.shape, anti_aliasing=True)  # Resize g to match the shape of B
    
    for filename in os.listdir(image_folder):
        if filename.endswith('.fits'):
            fits_file = os.path.join(image_folder, filename)
            
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                P = hdul[0].data
                
                # Apply gain correction using broadcasting
                S = (P - B) / (g * F)
                photons.append(S)
        
    return photons

def corrected_image(image_folder, B, F, g):
    
    # Ensure that g has the same shape as B, P, and F
    g = resize(g, B.shape, anti_aliasing=True)  # Resize g to match the shape of B
    
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    # Define variables for vmin, vmax, and current_frame
    vmin_value = 1000
    vmax_value = 3000
    current_frame = 0  # Initial frame
    
    # Function to update the displayed image
    def update_image():
        nonlocal current_frame
        fits_file = fits_files[current_frame]
        with fits.open(fits_file) as hdul:
            P = hdul[0].data
            S = (P - B) / (g * F)
            im.set_data(S)
            fig.canvas.draw_idle()
    
    # List to store FITS file paths
    fits_files = []
    
    # Collect FITS file paths
    for filename in os.listdir(image_folder):
        if filename.endswith('.fits'):
            fits_files.append(os.path.join(image_folder, filename))
    
    # Display the initial image
    fits_file = fits_files[current_frame]
    with fits.open(fits_file) as hdul:
        P = hdul[0].data
        S = (P - B) / (g * F)
        im = ax.imshow(S, cmap='gray', aspect='auto', norm='linear', vmin=vmin_value, vmax=vmax_value)
        cbar = plt.colorbar(im, ax=ax)
        title = ax.set_title('Corrected Image')
    
    # Create sliders for vmin and vmax
    ax_slider_vmin = plt.axes([0.12, 0.25, 0.65, 0.03])
    slider_vmin = Slider(ax_slider_vmin, 'vmin', 0, 5000, valinit=vmin_value, valstep=1)
    
    ax_slider_vmax = plt.axes([0.12, 0.2, 0.65, 0.03])
    slider_vmax = Slider(ax_slider_vmax, 'vmax', 0, 5000, valinit=vmax_value, valstep=1)
    
    def update(val):
        nonlocal vmin_value, vmax_value
        vmin_value = slider_vmin.val
        vmax_value = slider_vmax.val
        im.set_clim(vmin=vmin_value, vmax=vmax_value)
        fig.canvas.draw_idle()
    
    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)
    
    # Create a slider for navigation
    ax_slider_frame = plt.axes([0.12, 0.1, 0.65, 0.03])
    slider_frame = Slider(ax_slider_frame, 'Frame', 0, len(fits_files) - 1, valinit=current_frame, valstep=1)
    
    def update_frame(val):
        nonlocal current_frame
        current_frame = int(slider_frame.val)
        update_image()
    
    slider_frame.on_changed(update_frame)
    
    # Display the interactive figure
    plt.show()
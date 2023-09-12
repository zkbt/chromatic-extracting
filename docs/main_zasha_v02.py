#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:16:29 2023

@author: zashaavery
"""

import chromatic_extracting
from astropy.io import fits
from astropy.table import Table
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from IPython.display import display
import ipywidgets as widgets
import numpy as np
from scipy import ndimage, datasets
from matplotlib.backend_bases import PickEvent
from datetime import datetime
from ipywidgets import Button
from skimage.transform import resize  # Use skimage for resizing
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage, datasets
from scipy.stats import norm
import random


def organize_files(source_folder, destination_folder, header_keyword, file_type=None):

    """
    Definition:
        organize_files will organize files into a new folder depending on an
        inputted header
    
    Parameters:
        source_folder: the original folder with the unorganized fits files
        destination_folder: the folder where the user wants the organized files
        header_keyword: the header the user wants to organize
        file_type: fits files sometimes have suborganizations such as c1 or c2 
        (default = None means that fits files will not be separate by file type)
    
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
                
                
def headers(image_directory):
    
        """
        Description:
            headers() prints out a list of available headers from the fits 
            files in alphabetical order
        
        Parameters:
            image_directory(str): the file directory of the files you wish to 
            extract headers from 
        
        Returns:
            str headers
        """
        # Iterate over all FITS files in the directory
        for filename in os.listdir(image_directory):
            if filename.endswith('.fits'):
                fits_file = os.path.join(image_directory, filename)
                
                # Open the fits file. This method ensures it closes directly after use
                with fits.open(fits_file) as hdul:
                    header = hdul[0].header # extract the header
                    
                    headers = [] #empty list to store the headers
                    
                    for keyword in header: 
                        headers.append(keyword) # append the headers into the empty list
                        
        sorted_headers = list(set(headers)) # ensure that headers are not repeated
        sorted_headers.sort() # sort alphabetically
        print(sorted_headers)
    
    
def headers_data(image_directory, header):
    
    """
    Description: 
        headers_data() allows the user to get the corresponding data to a header
    
    Parameters:
        image_directory(str): the file directory of the files you wish to 
        extract header data from 
        header(str): the header of the wanted data
    
    Returns:
        arr data
    """
    
    header_upper = header.upper() # capatalize the input (case insensitive)
    
    data = [] # empty list to store data
    
    # Iterate over all FITS files in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith('.fits'):
            fits_file = os.path.join(image_directory, filename)
                
            # Open the FITS file
            with fits.open(fits_file) as hdul:
                single = hdul[0].header.get(header_upper, None) # extract the header data for one file
                data.append(single) # append the data to have header data for all files
    
    data = np.array(data) # list to array        
    return data


def sorted_datetime(image_directory):
    
    """
    Description:
        sorted_datetime() allows the user to sort their files in terms of 
        observation date-time
        
    Parameters:
        image_directory(str): file directory of images
        
    Returns:
        list sorted_obs_list: all of the headers and their data sorted in 
        order of observation date-time
    """
    obs_list = [] # empty list
    
    # Iterate over all FITS files in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith('.fits'):
            fits_file = os.path.join(image_directory, filename)
            
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


def animation(image_list, *header_names, vmin_initial=1000, vmax_initial=3000, valstep=1, 
              cmap='gray', fontsize = 8, figsize=(7, 8), current_frame=0, norm='linear'):
    
    """
    Definition:
        animation() plots an interactive figure with three total plots/images. 
        The first is the original image, the second is a time-series plot of 
        the pixels, and the third is a histogram of the pixels with a gaussian
        distribution.
        
    Parameters:
        image_directory: the directory of the images the user wish to examine
        image_list: the resulting list from sorted_datetime
        *header_names: the headers the user wishes to see information for
        vmin_initial/vmax: the setting for the initial slider to influence the 
        effect of colormap (default 1000 and 3000)
        valstep: the index of the images that are looked as the slider 
        progresses (default 1)
        cmap: the colormap for the images (default gray)
        fontsize: size of the font (default 8)
        figsize: the size of the printed figure (default (8, 6))
        norm: changes the scale of the images, options can be seen by running 
        matplotlib.scale.get_scale_names() (default linear)
        
    Returns:
        an interactive figure with three plots
    """
  
    # Initialize figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 2, 2]})
    
    # Adjust the spacing manually
    plt.subplots_adjust(bottom=0.15, right = 0.85, hspace=0.5)  # Adjust bottom and vertical spacing
  
    entry = image_list[current_frame]
    filename, obs_datetime, header_data, image_data = entry

    # Create the original image subplot
    im = ax1.imshow(image_data, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    cbar = plt.colorbar(im, ax=ax1)
    title_datetime = obs_datetime
    title = ax1.set_title(f"Observation Date and Time: {title_datetime}")
    header_texts = []  # Store the text objects for headers

    for i, header_name in enumerate(header_names):
        header_value = header_data.get(header_name, 'N/A')
        header_text = fig.text(1.22, 0.98 - i * 0.08, f"{header_name.upper()}: {header_value}",
                               transform=ax1.transAxes, fontsize=fontsize)
        header_texts.append(header_text)

    # Create sliders for vmin and vmax. Keep in mind, vmin < vmax
    ax_slider_vmin = plt.axes([0.12, 0.06, 0.65, 0.03])
    slider_vmin = Slider(ax_slider_vmin, 'vmin', 0, 5000, valinit=vmin_initial, valstep=valstep)

    ax_slider_vmax = plt.axes([0.12, 0.03, 0.65, 0.03])
    slider_vmax = Slider(ax_slider_vmax, 'vmax', 0, 5000, valinit=vmax_initial, valstep=valstep)

    def update(val):
        vmin_value = slider_vmin.val
        vmax_value = slider_vmax.val
        im.set_clim(vmin=vmin_value, vmax=vmax_value)
        fig.canvas.draw_idle()

    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)

    # Create a slider for navigation
    ax_slider_frame = plt.axes([0.12, 0, 0.65, 0.03])
    slider_frame = Slider(ax_slider_frame, 'Frame', 0, len(image_list) - 1, valinit=current_frame,
                          valstep=1)

    def update_frame(val):
        nonlocal current_frame
        current_frame = int(slider_frame.val)
        obs_datetime = image_list[current_frame][1]
        title.set_text(f"Observation Date and Time: {obs_datetime}")

        entry = image_list[current_frame]
        filename, obs_datetime, header_data, image_data = entry

        for i, header_name in enumerate(header_names):
            header_value = header_data.get(header_name, 'N/A')
            header_texts[i].set_text(f"{header_name.upper()}: {header_value}")  # Update the header text

        im.set_data(image_data)
        fig.canvas.draw_idle()

    slider_frame.on_changed(update_frame)

    # Create the second subplot for the time series (initialize with an empty plot)
    time_series_ax = ax2
    time_series_ax.set_xlabel('Frame')
    time_series_ax.set_ylabel('Pixel Value')
    time_series_ax.set_title('Time Series for Pixel (x, y)')  # Update with actual pixel coordinates
    
    hist = ax3
    hist.set_xlabel('Value')
    hist.set_ylabel('Probability Density')
    hist.set_title('Histogram for Pixel (x,y)')
    
    # Initialize the standard deviation text
    std_text = fig.text(1.015, 0.95, "", transform=ax3.transAxes, fontsize=8)
    mean_text = fig.text(1.015, 0.83, "", transform=ax3.transAxes, fontsize=8)

    def on_pixel_click(event):

        pixel_time_series = []

        if isinstance(event, PickEvent):
            x, y = int(event.mouseevent.xdata), int(event.mouseevent.ydata)

            for filename, obs_datetime, header_data, image_data in image_list:

                # Extract the pixel data for the clicked pixel (x, y) from your image_data
                pixel_data = image_data[y, x]
                pixel_time_series.append(pixel_data)

            # Create a time array if needed, representing the time points for each data point in pixel_time_series
            time_array = np.arange(len(image_list))

            # Clear the previous time series plot and plot the new one
            time_series_ax.clear()
            time_series_ax.plot(time_array, pixel_time_series)
            
            hist.clear()
            mean = np.mean(pixel_time_series)
            std_value = np.std(pixel_time_series)
           
            # Calculate the Gaussian distribution manually
            n, bins, patches = hist.hist(pixel_time_series, bins=len(image_list), density=True, alpha=0.6)
            gaussian = (1 / (std_value * np.sqrt(2 * np.pi))) * np.exp(-((bins - mean)**2)/ (2 * std_value**2))
            hist.plot(bins, gaussian, '--')
            
            std_text.set_text(r'$\sigma$' + f": {std_value:.2f}")
            mean_text.set_text(r'$mu$' + f": {mean:.2f}")
            time_series_ax.set_xlabel('Frame')
            time_series_ax.set_ylabel('Pixel Value')
            time_series_ax.set_title(f'Time Series for Pixel ({x}, {y})')
            hist.set_xlabel('Value')
            hist.set_ylabel('Probability Density')
            hist.set_title(f'Histogram for Pixel ({x}, {y})')
            fig.canvas.draw_idle()
            
            def update_std(val):
                std = fig.text(1.015, 0.95, np.std(pixel_time_series, axis = 0),
                                       transform=ax3.transAxes, fontsize=8)
                mu = fig.text(1.015, 0.83, np.mean(pixel_time_series, axis = 0),
                                       transform=ax3.transAxes, fontsize=8)

    fig.canvas.mpl_connect('pick_event', on_pixel_click)
    im = ax1.imshow(image_data, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm,
                     picker=True)
    plt.show()
  
                
def median_image(image_directory, median_name, vmin_initial=1000, vmax_initial=3000, valstep = 1, cmap = 'gray', figsize = (8, 6), norm = 'linear'):

    """
    Definition:
        def median_image will display an single image representing the median 
        to a set of FITS images
        
    Paramters: 
        image_directory: the location of the FITS images
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
    
    for filename in os.listdir(image_directory):
        
        if filename.endswith('.fits'):
            fits_file = os.path.join(image_directory, filename)
            
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


def median_image_data(image_directory):

    """
    Definition:
        def median_image_data will return a median, unnormalized image from a 
        set of FITS images
        
    Paramters: 
        image_directory: the location of the FITS images
        
    Results:
        arr median_data
    """
    
    median_pixels = []
    
    for filename in os.listdir(image_directory):
        
        if filename.endswith('.fits'):
            fits_file = os.path.join(image_directory, filename)
            
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                pixel = hdul[0].data
                median_pixels.append(pixel)
                
    median_data = np.median(median_pixels, axis = 0)
    return median_data

 
def read_noise(image_directory):

    """
    Definition:
        def read_noise will return the read noise from the bias image 
        
    Paramters: 
        image_directory: the location of the bias images
        
    Results:
        arr read_noise
    """
    
    median_pixels = []
    
    for filename in os.listdir(image_directory):
        
        if filename.endswith('.fits'):
            fits_file = os.path.join(image_directory, filename)
            
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                pixel = hdul[0].data
                median_pixels.append(pixel)
                
    std_image = np.std(median_pixels, axis = 0)
    
    read_noise = np.median(std_image)
    return read_noise


def median_filter(median_data, vmin_initial=1000, vmax_initial=3000, valstep = 1, cmap = 'gray', figsize = (8, 6), norm = 'linear'):

    """
    Definition:
        median_filter() applies a median filter to the median, unnormalized image
        
    Parameters:
        median_data: the data for the median, unnormalized image
        vmin_initial/vmax: the setting for the initial slider to influence the 
        effect of colormap (default 1000 and 3000)
        valstep: the index of the images that are looked as the slider progresses 
        (default 1)
        cmap: the colormap for the images (default gray)
        figsize: the size of the printed figure (default (8, 6))
        norm: changes the scale of the images, options can be seen by running 
        matplotlib.scale.get_scale_names() (default linear)
     
    Returns:
        an interactive figure of the median, unnormalized image as well as the 
        image with a median filter
    """
    
    # Initialize figure and axis
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    im0 = axs[0].imshow(median_data, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    
    filtered = ndimage.median_filter(median_data, size=20)
    im1 = axs[1].imshow(filtered, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    
    # Create a shared colorbar for both images
    cbar_ax = fig.add_axes([0.91, 0.2, 0.02, 0.7])  # Adjust position and size
    cbar = fig.colorbar(im0, cax=cbar_ax)
    
    axs[0].set_title('Original Median Image')
    axs[1].set_title('Filtered Median Image')
    
    # Create sliders for vmin and vmax. Keep in mind, vmin < vmax
    ax_slider_vmin = plt.axes([0.12, 0.25, 0.65, 0.03])
    slider_vmin = Slider(ax_slider_vmin, 'vmin', 0, 5000, valinit=vmin_initial, valstep=valstep)
    
    ax_slider_vmax = plt.axes([0.12, 0.2, 0.65, 0.03])
    slider_vmax = Slider(ax_slider_vmax, 'vmax', 0, 5000, valinit=vmax_initial, valstep=valstep)
    
    def update(val):
        vmin_value = slider_vmin.val
        vmax_value = slider_vmax.val
        im0.set_clim(vmin=vmin_value, vmax=vmax_value)
        im1.set_clim(vmin=vmin_value, vmax=vmax_value)
        fig.canvas.draw_idle()
    
    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)
    
    plt.show()


def median_filter_data(median_data):

    """
    Definition:
        median_filter_data() gives the data resulting from median_filter()
        
    Parameters:
        median_data: the data for the median, unnormalized image
        
    Returns:
        arr filter_data: the data resulting from median_filter()
    """
    
    filter_data = ndimage.median_filter(median_data, size = 20)
    return filter_data


def normalized_image(median_data, filter_data, vmin_initial=1000, vmax_initial=3000, valstep = 1, cmap = 'gray', figsize = (8, 6), norm = 'linear'):
    
    """
    Definition:
        normalized_image() plots the normalized, median image
        
    Parameters:
        median_data: the data for the median, unnormalized image
        filter_data: the data for the median, unnormalized image with a 
        median filter
        
    Returns:
        an interactive figure of the normalized, median image
    """
    
    normal = (median_data)/(filter_data)
    
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders

    im = ax.imshow(normal, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)

    cbar = plt.colorbar(im, ax=ax)

    title = ax.set_title('Normalized, Median Image')

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
    

def normalized_data(median_data, filter_data):

    """
    Definition:
        normalized_data() gives the data for a median, normalized image
        
    Parameters:
        median_data: the data for the median, unnormalized image
        filter_data: the data for the median, unnormalized image with a 
        median filter
        
    Returns:
        arr normal_data: the data for a normalized image
    """
    
    normal_data = (median_data)/(filter_data)
    return normal_data


def corrected_image(image_directory, B, F, g):

    """
    Definition:
        def corrected_image prints an interactive figure of the corrected images
        
    Paramters: 
        image_directory: the images you wish to correct
        B: the image data for the median bias images
        F: the image data for the normalized, median flat images
        g: the gain for the original images
        
    Results:
        an interactive figure of corrected images
    """
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
    for filename in os.listdir(image_directory):
        if filename.endswith('.fits'):
            fits_files.append(os.path.join(image_directory, filename))
    
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


def corrected_image_data(image_directory, B, F, g):

    """
    Definition: 
        def corrected_image_data returns the image data for a set of corrected images
        
    Parameters:
        image_folder: the images you wish to correct
        B: the image data for the median bias images
        F: the image data for the normalized, median flat images
        g: the gain for the original images
        
    Results: 
        returns the image data for the corrected image
    """
    
    photons = []
    
    # Ensure that g has the same shape as B, P, and F
    g = resize(g, B.shape, anti_aliasing=True)  # Resize g to match the shape of B
    
    for filename in os.listdir(image_directory):
        if filename.endswith('.fits'):
            fits_file = os.path.join(image_directory, filename)
            
            # Open the FITS file and extract observation date and time
            with fits.open(fits_file) as hdul:
                P = hdul[0].data
                
                # Apply gain correction using broadcasting
                S = (P - B) / (g * F)
                photons.append(S)
        
    return photons



def noise(image_list, image_fraction = 1):

    """
    Definition:
        noise() finds the standard deviation of individual pixels 
        
    Parameters:
        image_list: the resulting list from sorted_datetime
        image_fraction: the fraction of images to find the standard deviation 
        for (default 1)
        
    Returns:
        arr std_per_pixel: the standard deviation per pixel for all images in 
        image_list
    """
    num_images = int(len(image_list) * image_fraction)
    total_images = random.sample(image_list, num_images)
    
    pixel_time_series = []
    
    for filename, obs_datetime, header_data, image_data in total_images:
    
        # Extract the pixel data from image_data
        pixel_data = image_data
        pixel_time_series.append(pixel_data)
        
    std_per_pixel = np.std(pixel_time_series, axis=0)
    
    return std_per_pixel


def uncertainty_image(read_noise, photon_noise, vmin_initial=1000, vmax_initial=3000, 
                      valstep = 1, cmap = 'gray', figsize = (8, 6), norm = 'linear'):

    """
    Definition:
        uncertainty_image() plots the uncertainty image based off of read noise
        and photon noise
        
    Paramters:
        read_noise: the read noise (not sure if supposed to be per image or per pixel)
        photon_noise: the photon noise per pixel
        vmin_initial/vmax: the setting for the initial slider to influence the 
        effect of colormap (default 1000 and 3000)
        valstep: the index of the images that are looked as the slider progresses 
        (default 1)
        cmap: the colormap for the images (default gray)
        figsize: the size of the printed figure (default (8, 6))
        norm: changes the scale of the images, options can be seen by running 
        matplotlib.scale.get_scale_names() (default linear)
        
    Returns:
        an interactive image of the uncertainty due to photon and read noise
    """
    
    total_uncertainty = np.sqrt(read_noise**2 + photon_noise**2)
    
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    im = ax.imshow(total_uncertainty, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    
    cbar = plt.colorbar(im, ax=ax)
    
    title = ax.set_title('Uncertainty Image')
    
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






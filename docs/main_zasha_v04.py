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
from specreduce import WavelengthCalibration1D
from specreduce.core import _ImageParser
from specreduce.tracing import FlatTrace
from specreduce.tracing import FitTrace
from specreduce.background import Background
from specreduce.extract import BoxcarExtract
from specreduce.extract import HorneExtract
from matplotlib.widgets import TextBox
from astropy.visualization import quantity_support
from astropy.modeling import Model, fitting, models


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
        (default = None means that fits files will not be separated by file type)
    
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


def interactive(sorted_list, *header_names, vmin_initial=1000, vmax_initial=3000, valstep=1, 
              cmap='gray', fontsize = 8, figsize=(7, 8), current_frame=0, norm='linear'):
    
    """
    Definition:
        interactive() plots an interactive figure with three total plots/images. 
        The first is the original image, the second is a time-series plot of 
        the pixels, and the third is a histogram of the pixels with a gaussian
        distribution.
        
    Parameters:
        sorted_list: the resulting list from sorted_datetime
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
  
    entry = sorted_list[current_frame]
    filename, obs_datetime, header_data, image_data = entry

    # Create the original image subplot
    im = ax1.imshow(image_data, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    cbar = plt.colorbar(im, ax=ax1)
    title_datetime = obs_datetime
    title = ax1.set_title(f"Observation Date and Time: {title_datetime}")
    header_texts = []  # Store the text objects for headers

    for i, header_name in enumerate(header_names):
        header_value = header_data.get(header_name, 'N/A')
        header_text = fig.text(1.24, 0.98 - i * 0.08, f"{header_name.upper()}: {header_value}",
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
    slider_frame = Slider(ax_slider_frame, 'Frame', 0, len(sorted_list) - 1, valinit=current_frame,
                          valstep=1)

    def update_frame(val):
        nonlocal current_frame
        current_frame = int(slider_frame.val)
        obs_datetime = sorted_list[current_frame][1]
        title.set_text(f"Observation Date and Time: {obs_datetime}")

        entry = sorted_list[current_frame]
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

            for filename, obs_datetime, header_data, image_data in sorted_list:

                # Extract the pixel data for the clicked pixel (x, y) from your image_data
                pixel_data = image_data[y, x]
                pixel_time_series.append(pixel_data)

            # Create a time array if needed, representing the time points for each data point in pixel_time_series
            time_array = np.arange(len(sorted_list))

            # Clear the previous time series plot and plot the new one
            time_series_ax.clear()
            time_series_ax.plot(time_array, pixel_time_series)
            
            hist.clear()
            mean = np.mean(pixel_time_series)
            std_value = np.std(pixel_time_series)
           
            # Calculate the Gaussian distribution manually
            n, bins, patches = hist.hist(pixel_time_series, bins=len(sorted_list), density=True, alpha=0.6)
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
  
                
def median_image(image_directory, median_name, vmin_initial=1000, vmax_initial=3000, valstep = 1, 
                 cmap = 'gray', figsize = (8, 6), norm = 'linear'):

    """
    Definition:
        def median_image will display a single image representing the median 
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
        vmin_initial/vmax: the setting for the initial slider to influence the 
        effect of colormap (default 1000 and 3000)
        valstep: the index of the images that are looked as the slider progresses 
        (default 1)
        cmap: the colormap for the images (default gray)
        figsize: the size of the printed figure (default (8, 6))
        norm: changes the scale of the images, options can be seen by running 
        matplotlib.scale.get_scale_names() (default linear)
        
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
    

def normalized_image_data(median_data, filter_data):

    """
    Definition:
        normalized_image_data() gives the data for a median, normalized image
        
    Parameters:
        median_data: the data for the median, unnormalized image
        filter_data: the data for the median, unnormalized image with a 
        median filter
        
    Returns:
        arr normal_data: the data for a normalized image
    """
    
    normal_data = (median_data)/(filter_data)
    return normal_data


def corrected_image(sorted_list, B, F, g, vmin_initial=1000, vmax_initial=3000, valstep=1, 
                     cmap='gray', fontsize=8, figsize=(7,8), current_frame=0, norm='linear'):

    """
    Definition:
        def corrected_image plots an interactive figure of the corrected images
        
    Paramters: 
        sorted_list: the images you wish to correct
        B: the image data for the median bias images
        F: the image data for the normalized, median flat images
        g: the gain for the original images
        
    Results:
        an interactive figure of calibrated images
    """
    
    # Ensure that g has the same shape as B, P, and F
    g = resize(g, B.shape, anti_aliasing=True)  # Resize g to match the shape of B
    
    def update(val):
        vmin_value = slider_vmin.val
        vmax_value = slider_vmax.val
        im.set_clim(vmin=vmin_value, vmax=vmax_value)
        fig.canvas.draw_idle()
    
    def update_frame(val):
        nonlocal current_frame
        current_frame = int(slider_frame.val)
        obs_datetime = sorted_list[current_frame][1]
        title.set_text(f"Observation Date and Time: {obs_datetime}")

        entry = sorted_list[current_frame]
        filename, obs_datetime, header_data, image_data = entry

        P = image_data
        S = (P - B) / (g * F)
        
        im.set_data(S)
        fig.canvas.draw_idle()
        
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    entry = sorted_list[current_frame]
    filename, obs_datetime, header_data, image_data = entry

    P = image_data
    S = (P - B) / (g * F)

    # Create the original image subplot
    im = ax.imshow(S, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    cbar = plt.colorbar(im, ax=ax)
    title_datetime = obs_datetime
    title = ax.set_title(f"Observation Date and Time: {title_datetime}")
    
    # Create sliders for vmin and vmax. Keep in mind, vmin < vmax
    ax_slider_vmin = plt.axes([0.12, 0.06, 0.65, 0.03])
    slider_vmin = Slider(ax_slider_vmin, 'vmin', 0, 5000, valinit=vmin_initial, valstep=valstep)

    ax_slider_vmax = plt.axes([0.12, 0.03, 0.65, 0.03])
    slider_vmax = Slider(ax_slider_vmax, 'vmax', 0, 5000, valinit=vmax_initial, valstep=valstep)
    
    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update) 
    
    ax_slider_frame = plt.axes([0.12, 0, 0.65, 0.03])
    slider_frame = Slider(ax_slider_frame, 'Frame', 0, len(sorted_list) - 1, valinit=current_frame,
                          valstep=1)

    slider_frame.on_changed(update_frame)
    
    plt.show()


def corrected_image_data(image_list, B, F, g):

    """
    Definition: 
        def corrected_image_data returns the image data for a set of corrected images
        (returns the number of photons/pixel for each image or N)
        
    Parameters:
        image_folder: the images you wish to correct
        B: the image data for the median bias images
        F: the image data for the normalized, median flat images
        g: the gain for the original images
        
    Results: 
        returns the image data for the calibrated image
    """
    
    photons = []
    
    # Ensure that g has the same shape as B, P, and F
    g = resize(g, B.shape, anti_aliasing=True)  # Resize g to match the shape of B
    
    for filename, obs_datetime, header_data, image_data in image_list:
    
       P=image_data
       S = (P - B) / (g * F)
       photons.append(S)
        
    return photons


def save_corrected_image(image_list, B, F, g, destination_folder):

    """
    Definition: 
        def save_corrected_image saves the image data for a set of corrected images
        in a folder (saves the number of photons/pixel for each image or N)
        
    Parameters:
        image_folder: the images you wish to correct
        B: the image data for the median bias images
        F: the image data for the normalized, median flat images
        g: the gain for the original images
        
    Results: 
        returns the image data for the calibrated image
    """
    
    # Create the destination parent folder if it doesn't exist.
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    # Ensure that g has the same shape as B, P, and F
    g = resize(g, B.shape, anti_aliasing=True)  # Resize g to match the shape of B
    
    for filename, obs_datetime, header_data, image_data in image_list:
    
       P=image_data
       S = (P - B) / (g * F)
       
       # Save the corrected image as a FITS file
       corrected_image_filename = os.path.join(destination_folder, f"corrected_{filename}")
       hdu = fits.PrimaryHDU(S)
       hdul = fits.HDUList([hdu])
       hdul.writeto(corrected_image_filename, overwrite=True)


def show_corrected_image(image_filename, vmin_initial=1000, vmax_initial=3000, valstep = 1, 
                     cmap='gray', fontsize=8, figsize=(8,6)):
    
    hdu_list = fits.open(image_filename)
    image_data = hdu_list[0].data
    hdu_list.close()
    
    def update(val):
        vmin_value = slider_vmin.val
        vmax_value = slider_vmax.val
        im.set_clim(vmin=vmin_value, vmax=vmax_value)
        fig.canvas.draw_idle()
        
    fig, ax = plt.subplots(figsize = figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    im = ax.imshow(image_data, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)

    title = ax.set_title('Corrected Image')
    
    # Create sliders for vmin and vmax. Keep in mind, vmin < vmax
    ax_slider_vmin = plt.axes([0.12, 0.06, 0.65, 0.03])
    slider_vmin = Slider(ax_slider_vmin, 'vmin', 0, 5000, valinit=vmin_initial, valstep=valstep)

    ax_slider_vmax = plt.axes([0.12, 0.03, 0.65, 0.03])
    slider_vmax = Slider(ax_slider_vmax, 'vmax', 0, 5000, valinit=vmax_initial, valstep=valstep)
    
    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update) 

    plt.plot()
    

def measured_noise_image(image_list, image_number = 10, vmin_initial=1000, vmax_initial=3000, 
                      valstep = 1, cmap = 'gray', figsize = (8, 6), norm = 'linear'):
    
    """
    Definition:
        measured_noise_image() displays the image representing the 
        measured noise (the standard deviation per pixel for all images
                        in a set list)
        
    Parameters:
        image_list: the resulting list from sorted_datetime
        image_number: the number of images  in succession to find the standard deviation 
        for (default 10)
        vmin_initial/vmax: the setting for the initial slider to influence the 
        effect of colormap (default 1000 and 3000)
        valstep: the index of the images that are looked as the slider progresses 
        (default 1)
        cmap: the colormap for the images (default gray)
        figsize: the size of the printed figure (default (8, 6))
        norm: changes the scale of the images, options can be seen by running 
        matplotlib.scale.get_scale_names() (default linear)
        
    Returns:
        arr std_per_pixel: the standard deviation per pixel for all images in 
        image_list
    """

    max_initial_index = len(image_list) - image_number
    initial_index = random.randint(0, max_initial_index)
    
    total_images = image_list[initial_index:initial_index + image_number]
    
    pixel_time_series = []
    
    for filename, obs_datetime, header_data, image_data in total_images:
    
        # Extract the pixel data from image_data
        pixel_data = image_data
        pixel_time_series.append(pixel_data)
        
    std_per_pixel = np.std(pixel_time_series, axis=0)

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    im = ax.imshow(std_per_pixel, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    
    cbar = plt.colorbar(im, ax=ax)
    
    title = ax.set_title('Measured Noise Image')
    
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


def measured_noise_data(image_list, image_number = 10):
    
    """
    Definition:
        measured_noise_data() saves the data for measured_noise_image()
        
    Parameters:
        image_list: the resulting list from sorted_datetime
        image_number: the number of images  in succession to find the standard deviation 
        for (default 10)
        
    Returns:
        arr std_per_pixel: the standard deviation per pixel for all images in 
        image_list
    """
    
    max_initial_index = len(image_list) - image_number
    initial_index = random.randint(0, max_initial_index)
    
    total_images = image_list[initial_index:initial_index + image_number]
    
    pixel_time_series = []
    
    for filename, obs_datetime, header_data, image_data in total_images:
    
        # Extract the pixel data from image_data
        pixel_data = image_data
        pixel_time_series.append(pixel_data)
        
    std_per_pixel = np.std(pixel_time_series, axis=0)
    
    return std_per_pixel
    
    
def save_measured_noise(image_list, destination_folder, image_number = 10):

    """
    Definition:
        save_measured_noise() save the data found by taking
        the standard deviation of individual pixels 
        
    Parameters:
        image_list: the resulting list from sorted_datetime
        destination_folder: the folder output for the function
        image_number: the number of images in succession to find the standard devation 
        for (default 10)
        
    Returns:
        folder of images
    """
    
    # Create the destination parent folder if it doesn't exist.
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    max_initial_index = len(image_list) - image_number
    initial_index = random.randint(0, max_initial_index)
    
    total_images = image_list[initial_index:initial_index + image_number]
    
    pixel_time_series = []
    
    for filename, obs_datetime, header_data, image_data in total_images:
    
        # Extract the pixel data from image_data
        pixel_data = image_data
        pixel_time_series.append(pixel_data)
        
    std_per_pixel = np.std(pixel_time_series, axis=0)
    
    # Save the standard deviation array as a FITS file
    measured_noise_filename = os.path.join(destination_folder, f"measured_noise_image")
    hdu = fits.PrimaryHDU(std_per_pixel)
    hdul = fits.HDUList([hdu])
    hdul.writeto(measured_noise_filename, overwrite=True)


def uncertainty_image(folder_path, read_noise, vmin_initial=1000, vmax_initial=3000, 
                          valstep = 1, cmap = 'gray', figsize = (8, 6), norm = 'linear'):

    """
    Definition:
        uncertainty_image() plots the uncertainty image based off of read noise
        and photon noise
        
    Paramters:
        
        folder_path: the path for the corrected_images
        read_noise: the read noise (1 value)
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
    
    # Get the list of FITS files in the folder
    image_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.fits')]
    first_file = image_list[0]
    with fits.open(first_file) as hdul:
        image_data = hdul[0].data
        total_uncertainty = np.sqrt(read_noise**2 + image_data)
    
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    im = ax.imshow(total_uncertainty, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    
    cbar = plt.colorbar(im, ax=ax)
    
    title = ax.set_title('Predicted Uncertainty Image')
    
    # Create sliders for vmin and vmax. Keep in mind, vmin < vmax
    ax_slider_vmin = plt.axes([0.12, 0.25, 0.65, 0.03])
    slider_vmin = Slider(ax_slider_vmin, 'vmin', 0, 5000, valinit=vmin_initial, valstep=valstep)
    
    ax_slider_vmax = plt.axes([0.12, 0.2, 0.65, 0.03])
    slider_vmax = Slider(ax_slider_vmax, 'vmax', 0, 5000, valinit=vmax_initial, valstep=valstep)
    
    ax_text_image = plt.axes([0.93, 0.65, 0.06, 0.04])
    text_image = TextBox(ax_text_image, 'Image #', initial='1')
    
    def update(val):
        vmin_value = slider_vmin.val
        vmax_value = slider_vmax.val
        image_number = int(text_image.text)
        
        # Read the selected fits file
        selected_file = image_list[image_number - 1]
        with fits.open(selected_file) as hdul:
            image_data = hdul[0].data
            total_uncertainty = np.sqrt(read_noise**2 + image_data)
    
        im.set_data(total_uncertainty)
        im.set_clim(vmin=vmin_value, vmax=vmax_value)
        title.set_text(f'Predicted Uncertainty Image for {image_number}')
        fig.canvas.draw_idle()
        
    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)
    text_image.on_submit(update)
        
    plt.plot()
    

def save_uncertainty_images(folder_path, read_noise, destination_folder):
    
    """
    Definition:
        save_uncertainty_images() saves the uncertainty images 
        
    Paramters:
        
        folder_path: the path for the corrected_images
        read_noise: the read noise (1 value)
        destination_folder:: the folder output for the function
        
    Returns:
        an interactive image of the uncertainty due to photon and read noise
    """
    
    # Create the destination parent folder if it doesn't exist.
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get the list of FITS files in the folder
    image_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.fits')]

    for image_path in image_list:
        with fits.open(image_path) as hdul:
            image_data = hdul[0].data
            total_uncertainty = np.sqrt(read_noise**2 + image_data)

            # Construct the uncertainty filename based on the original image filename
            filename = os.path.splitext(os.path.basename(image_path))[0]
            uncertainty_filename = os.path.join(destination_folder, f"{filename}_uncertainty.fits")

            # Save the uncertainty image as a FITS file
            hdu = fits.PrimaryHDU(total_uncertainty)
            hdul = fits.HDUList([hdu])
            hdul.writeto(uncertainty_filename, overwrite=True)
    

def resulting_image(measured_noise_data, uncertainty_image_data, vmin_initial=1000, vmax_initial=3000, 
                      valstep = 1, cmap = 'gray', figsize = (8, 6), norm = 'linear'):
    
    resulting_image = measured_noise_data/uncertainty_image_data

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    im = ax.imshow(resulting_image, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    
    cbar = plt.colorbar(im, ax=ax)
    
    title = ax.set_title('Resulting Image')
    
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

    
def resulting_image_data(measured_noise_data, uncertainty_image_data):
    
    resulting_image = measured_noise_data/uncertainty_image_data
    return resulting_image


def three_panel(N, uncertainty_image_data, resulting_image_data, vmin_initial=1000, vmax_initial=3000, 
                      valstep = 1, cmap = 'gray', figsize = (8, 6), norm = 'linear'):
    
    # Initialize figure and axis
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    plt.subplots_adjust(bottom=0.4)  # Adjust to account for sliders
    
    im0 = axs[0].imshow(N, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    
    im1 = axs[1].imshow(uncertainty_image_data, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    
    im2 = axs[2].imshow(resulting_image_data, cmap=cmap, vmin=vmin_initial, vmax=vmax_initial, aspect='auto', norm=norm)
    
    # Create a shared colorbar for all images
    cbar_ax = fig.add_axes([0.91, 0.2, 0.02, 0.7])  # Adjust position and size
    cbar = fig.colorbar(im0, cax=cbar_ax)
    
    axs[0].set_title('Measured Noise Image')
    axs[1].set_title('Uncertainty Image')
    axs[2].set_title('Resulting Image')
    
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
        im2.set_clim(vmin=vmin_value, vmax=vmax_value)
        fig.canvas.draw_idle()
    
    slider_vmin.on_changed(update)
    slider_vmax.on_changed(update)
    
    plt.show()

def plot_spectrum(folder_path, image_data, guess=276, window=10, separation=5, width=2, figsize=(10, 4)):
    
    # Get the list of FITS files in the folder
    fits_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.fits')]

    # Read the first fits file
    first_file = fits_files[0]
    with fits.open(first_file) as hdul:
        image_data = hdul[0].data.T  # extract the image data and transpose
        fit_trace = FitTrace(image_data, guess=guess, window=window) 
        # Create initial spectral extraction
        bg_onesided = Background.one_sided(image_data, fit_trace, separation=separation, width=width)
        extract_box = BoxcarExtract(image_data - bg_onesided, fit_trace)
        spectrum = extract_box.spectrum

        flux = spectrum.flux
        spectral_axis = spectrum.spectral_axis
        trace = fit_trace.trace


    # Create a plot for the first image
    fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios':[2,3]})
    
    axs[0].plot(spectral_axis, flux, lw=1)
    axs[0].set_xlabel('Pixel Coordinate')  # Replace with 'Wavelength (units)' if it's wavelength
    axs[0].set_ylabel('Flux (photons)')  # Replace 'DN' with the appropriate units
    axs[0].set_title('Spectrum for Image 1')
    # Set y and x-axis limits to cover the full range of flux values
    axs[0].set_xlim(0, 2100)
    axs[0].set_ylim(0, 100000)
    
    axs[1].imshow(image_data, cmap='gray')
    axs[1].plot(trace, color='black', linestyle=':', linewidth=1, label='Trace')
    axs[1].set_title('Image with Trace Line')
    
    # Create text boxes for pixel, separation, width, and image number
    ax_text_guess = plt.axes([0.53, 0.85, 0.08, 0.04])
    text_guess = TextBox(ax_text_guess, 'Pixel', initial=str(guess))
    
    ax_text_window = plt.axes([0.53, 0.8, 0.08, 0.04])
    text_window = TextBox(ax_text_window, 'Window', initial=str(window))

    ax_text_sep = plt.axes([0.53, 0.75, 0.08, 0.04])
    text_sep = TextBox(ax_text_sep, 'Separation', initial=str(separation))

    ax_text_width = plt.axes([0.53, 0.7, 0.08, 0.04])
    text_width = TextBox(ax_text_width, 'Width', initial=str(width))

    ax_text_image = plt.axes([0.53, 0.65, 0.08, 0.04])
    text_image = TextBox(ax_text_image, 'Image #', initial='1')

    def update(val):
        guess_value = float(text_guess.text)
        window_value = float(text_window.text)
        separation_value = float(text_sep.text)
        width_value = float(text_width.text)
        image_number = int(text_image.text)

        # Read the selected fits file
        selected_file = fits_files[image_number - 1]
        with fits.open(selected_file) as hdul:
            image_data = hdul[0].data.T  # extract the image data and transpose
            fit_trace = FitTrace(image_data, guess=guess_value, window=window_value)
            bg_onesided = Background.one_sided(image_data, fit_trace, separation=separation_value, width=width_value)
            extract_box = BoxcarExtract(image_data - bg_onesided, fit_trace)
            spectrum = extract_box.spectrum

            flux = spectrum.flux
            spectral_axis = spectrum.spectral_axis
            trace = fit_trace.trace

        axs[0].clear()
        axs[0].plot(spectral_axis, flux, lw=1)
        axs[0].set_xlabel('Pixel Coordinate')  # Replace with 'Wavelength (units)' if it's wavelength
        axs[0].set_ylabel('Flux (DN)')  # Replace 'DN' with the appropriate units
        axs[0].set_title('Spectrum for Image 1')
        # Set y and x-axis limits to cover the full range of flux values
        axs[0].set_xlim(0, 2100)
        axs[0].set_ylim(0, 100000)
        
        axs[1].clear()
        axs[1].imshow(image_data, cmap='gray')
        axs[1].plot(trace, color='black', linestyle=':', linewidth=1, label='Trace')
        axs[1].set_title('Image with Trace Line')
        
        fig.canvas.draw_idle()

    # Connect the update function to all text boxes
    text_guess.on_submit(update)
    text_window.on_submit(update)
    text_sep.on_submit(update)
    text_width.on_submit(update)
    text_image.on_submit(update)
    
    fig.subplots_adjust(right=0.85)
    plt.show()
    
    
def plot_spectrum(folder_path, image_data, guess=276, window=10, separation=5, width=2, figsize=(10, 4)):
    
    # Get the list of FITS files in the folder
    fits_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.fits')]

    # Read the first fits file
    first_file = fits_files[0]
    with fits.open(first_file) as hdul:
        image_data = hdul[0].data.T  # extract the image data and transpose
        fit_trace = FitTrace(image_data, guess=guess, window=window) 
        # Create initial spectral extraction
        bg_onesided = Background.one_sided(image_data, fit_trace, separation=separation, width=width)
        extract_box = BoxcarExtract(image_data - bg_onesided, fit_trace)
        spectrum = extract_box.spectrum

        flux = spectrum.flux
        spectral_axis = spectrum.spectral_axis
        trace = fit_trace.trace


    # Create a plot for the first image
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Plot the spectrum on the left
    axs[0].plot(spectral_axis, flux, lw=1)
    axs[0].set_xlabel('Pixel Coordinate')  # Replace with 'Wavelength (units)' if it's wavelength
    axs[0].set_ylabel('Flux (photons)')  # Replace 'DN' with the appropriate units
    axs[0].set_title('Spectrum for Image 1')
    # Set y and x-axis limits to cover the full range of flux values
    axs[0].set_xlim(0, 2100)
    axs[0].set_ylim(0, 100000)

    # Plot the image on the right
    axs[1].imshow(image_data, cmap='gray')
    axs[1].plot(trace, color='black', linestyle=':', linewidth=1, label='Trace')
    axs[1].set_title('Image with Trace Line')
    
    # Create text boxes for pixel, separation, width, and image number
    ax_text_guess = plt.axes([0.93, 0.85, 0.06, 0.04])
    text_guess = TextBox(ax_text_guess, 'Pixel', initial=str(guess))
    
    ax_text_window = plt.axes([0.93, 0.8, 0.06, 0.04])
    text_window = TextBox(ax_text_window, 'Window', initial=str(window))

    ax_text_sep = plt.axes([0.93, 0.75, 0.06, 0.04])
    text_sep = TextBox(ax_text_sep, 'Separation', initial=str(separation))

    ax_text_width = plt.axes([0.93, 0.7, 0.06, 0.04])
    text_width = TextBox(ax_text_width, 'Width', initial=str(width))

    ax_text_image = plt.axes([0.93, 0.65, 0.06, 0.04])
    text_image = TextBox(ax_text_image, 'Image #', initial='1')

    def update(val):
        guess_value = float(text_guess.text)
        window_value = float(text_window.text)
        separation_value = float(text_sep.text)
        width_value = float(text_width.text)
        image_number = int(text_image.text)

        # Read the selected fits file
        selected_file = fits_files[image_number - 1]
        with fits.open(selected_file) as hdul:
            image_data = hdul[0].data.T  # extract the image data and transpose
            fit_trace = FitTrace(image_data, guess=guess_value, window=window_value)
            bg_onesided = Background.one_sided(image_data, fit_trace, separation=separation_value, width=width_value)
            extract_box = BoxcarExtract(image_data - bg_onesided, fit_trace)
            spectrum = extract_box.spectrum

            flux = spectrum.flux
            spectral_axis = spectrum.spectral_axis
            trace = fit_trace.trace

        axs[0].clear()
        axs[0].plot(spectral_axis, flux, lw=1)
        axs[0].set_xlabel('Pixel Coordinate')  # Replace with 'Wavelength (units)' if it's wavelength
        axs[0].set_ylabel('Flux (DN)')  # Replace 'DN' with the appropriate units
        axs[0].set_title(f'Spectrum for Image {image_number}')
        # Set y and x-axis limits to cover the full range of flux values
        axs[0].set_xlim(0, 2100)
        axs[0].set_ylim(0, 100000)
        
        axs[1].clear()
        axs[1].imshow(image_data, cmap='gray')
        axs[1].plot(trace, color='black', linestyle=':', linewidth=1, label='Trace')
        axs[1].set_title('Image with Trace Line')
        
        fig.canvas.draw_idle()

    # Connect the update function to all text boxes
    text_guess.on_submit(update)
    text_window.on_submit(update)
    text_sep.on_submit(update)
    text_width.on_submit(update)
    text_image.on_submit(update)
    
    fig.subplots_adjust(right=0.85)
    
    plt.show()
    
    
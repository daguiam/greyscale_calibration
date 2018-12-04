###########
version = 20181204

### Analysis of profilometer data of Gray Scale lithography tests

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import colors
from scipy import stats

import tkinter as tk
from tkinter import filedialog

import time
import os
import re
import sys
from threading import Timer
##### Expected grayscale parameters


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def cumsum_difference(x1, x2, size=None):
    maxidx = size or min([len(x1),len(x2)])
    return np.sum(np.abs(x1[0:maxidx]-x2[0:maxidx]))
    
def middle_idx(y, search_range=None, size=None, debug=False):
    list_idx = np.arange(0,len(y),1)
    mididx = int(len(y)/2)
    search_range = search_range or int(len(y)*1/4)
    list_idx = np.arange(mididx-search_range, mididx+search_range,1)
    # list_idx = [int(len(y)/2)]

    result_diff = []
    result_idx = []
    for i,idx in enumerate(list_idx):
        part1, part2 = np.split(y, [idx])
        part1 = np.flip(part1, axis=0)

        diff = cumsum_difference(part1, part2, size=size)
    #     print(i, diff)
        result_diff.append(diff)
        result_idx.append(idx)
        
    midpointidx = list_idx[np.argmin(result_diff)]
    
    if debug:
        plt.plot(y, label='left', c='C0') 
        plt.plot(result_idx, result_diff/np.ptp(result_diff),c='C3')
        plt.axvline(midpointidx, c='k')
    return midpointidx


def get_directory(file_path):
    """ Gets the directory of the file_path
    """
    directory, filename = os.path.split(file_path)
    directory = os.path.normpath(directory)
    return directory



def get_filename_radical(file_path, characters_to_remove=None):
    """ Gets the radical of the filename
    """

    assert os.path.isfile(file_path), file_path+ " does not exist" 

    # Gets base directory and filename of selected file
    directory, filename = os.path.split(file_path)
    directory = os.path.normpath(directory)
    # Splits filename into radical and extension
    filename_radical,filename_ext = os.path.splitext(filename)
    # Cleans the filename radical to remove digits at the end 
    filename_radical = re.sub(r'\d+$', '', filename_radical)

    if characters_to_remove:
        filename_radical = filename_radical[:-characters_to_remove]
    return filename_radical



if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    print("""Calculate Greyscale curve from step height calibration wafer
            Version: %d
        Select an profilometer file with full wafer ascii data
    """%(version))

    # Opens windows dialog to select file

    if len(sys.argv)>1:
        file_path = sys.argv[1]
        assert os.path.isfile(file_path), "File not found: %s"%(file_path)
    else:
        # file_path = filedialog.askopenfilename()
        filez = filedialog.askopenfilenames(parent=root,title='Select profilometer files to compile')
        list_paths = list(filez)
        # print (list_paths)

    files = list_paths





    #### FOR complete 128 depths measurement
    max_depths = 128
    pitch = 100
    skip_beginning = 20

    split_sections = np.arange(max_depths)*pitch+skip_beginning
    surface_percentile = 90
    bottom_percentile = 10
    greyscales = np.zeros(max_depths)

    max_depth = -2.5
    max_height = 0.1

    save_plots = False
    save_plots = True

    print("""
    Greyscale values: %d
    Pitch: %0.3f um
    Skipped first: %0.3f um
    Surface percentile: %0.3f
    Bottom percentile: %0.3f
    Plot [ymin, ymax]: [%d, %d] um
    """%(len(greyscales), pitch, skip_beginning, surface_percentile, bottom_percentile,
    max_depth, max_height))

    for file_i,file in enumerate(files):
        # print(file_i, file)
        root_dir = os.path.dirname(file)
        file = os.path.basename(file)
        

        filename = os.path.join(root_dir, file)

    #     print("Opening",filename)
        
        gs1, gs2 = file.split('.txt')[0].split('to')
        gs1 = np.int(gs1)
        gs2= np.int(gs2)
        gs1,gs2
        gslen = gs2-gs1
        
        # if gslen >max_depths-1 :
        #     continue  
        gs_values = np.arange(gs1,gs2+1)
    #     print('from',gs1,'to',gs2, gslen, gs_values)
        


            
        # Header points

        df_header = pd.read_csv(filename, sep='\s+',header=None, nrows=7 )
        df_header = df_header.T
        df_header.columns = df_header.iloc[0]
        df_header = df_header.reindex(df_header.index.drop(0))
        name = df_header.Data
        x_resolution = df_header['X-Resolution'].astype('float').values
        y_resolution = 0.0001
        x_coord = df_header['X-Coord'].values
        y_coord = df_header['Y-Coord'].values

        # Data points
        df = pd.read_csv(filename, sep='\s+',header=0, skiprows=7)

        x = np.arange(len(df.Intermediate))*x_resolution
    #     print(x)
        y = df['Normal']*y_resolution
    #     plt.plot(x, y, label=file_i)


        sections = [find_nearest_idx(x, val) for val in split_sections]
        x_divs = np.split(x, sections)
        y_divs = np.split(y, sections)
        
        
    #     print(len(x_divs), x_divs)

        # INDIVIDUAL MEASUREMENTS

        total_notches = len(x_divs)
        print("File\t\t Notch \t Greyscale @ Depth")

        for i,(x, y) in enumerate(zip(x_divs,y_divs)):
            if i > gslen:
                break
            x = x-x[0]
            gs = gs_values[i]
            
    #         plt.plot(x, y, label=gs)

        # RELINEARIZE

            edge_thresholds = [20,90]
        #     plt.plot(x,y)

            sections = [find_nearest_idx(x, val) for val in edge_thresholds]
            x_edge = np.split(x, sections)
            y_edge = np.split(y, sections)

            x_edge = np.append(x_edge[0],x_edge[-1])
            y_edge = np.append(y_edge[0],y_edge[-1])


        #     plt.plot(x_edge, y_edge, '.')
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_edge,y_edge)

            y_lin = slope*x+intercept
    #         plt.plot(x,y_lin,linestyle='--', lw=0.5)
            y_linearized = y-y_lin
            
            plt.figure()
            
            
            plt.plot(x,y_linearized)
            
            surface = np.percentile(y_linearized, surface_percentile)
            bottom = np.percentile(y_linearized, bottom_percentile)
            height = bottom-surface
            
            # print(gs, "%0.3f"%surface, "%0.3f"%bottom, "%0.3f"%height)
            print("%s\t %d/%d \t %3d   @   %0.3f um"%(file, i, total_notches, gs, height))
            
            greyscales[gs] = height
        

            title = 'GS %d - %0.3f um'%(gs, height)
            plt.title(title)
            
            plt.xlabel('Scan length [$\mu$m]')
            plt.ylabel('Height [$\mu$m]')
            plt.ylim(max_depth,max_height)
            plt.grid('on')
            fig_filename = 'gs%d.png'%(gs)
            filename = os.path.join(root_dir, fig_filename)
            if save_plots:
                plt.savefig(filename)
            
        
    gs_heights = greyscales
    # print(greyscales)

    plt.figure()
    gs_levels = np.arange(len(gs_heights))
    plt.plot(gs_levels, gs_heights,)
    plt.grid('on')
    plt.xlabel('Greyscale values')
    plt.ylabel('Height [$\mu$m]')
    fig_filename = 'gs_scale.png'
    filename = os.path.join(root_dir, fig_filename)
    if save_plots:

        plt.savefig(filename)
    plt.legend()


    greyscale_calibration = np.array([gs_levels, gs_heights]).T
    gs_filename = 'greyscale_calibration.csv'
    filename = os.path.join(root_dir, gs_filename)
    header = 'Grayscale level, step height [um]'
    np.savetxt(filename, greyscale_calibration, delimiter=',', fmt='%d,%0.4f', header=header) 

    print("""
    Finished calculating greyscale
    """)
    time.sleep(5)
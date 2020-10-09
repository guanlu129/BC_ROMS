globals().clear()
clear all
os.system("clear")

#-------------------  import packages ----------------------------------------------------------------------------------
import xarray as xr
import sys
import os
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from itertools import islice
from mpl_toolkits.basemap import Basemap
import fiona
import rasterio.mask
import rasterio
import pyproj
from rasterio.transform import Affine

#-----------------------------set file paths----------------------------------------------------------------------------
file_path_BCROMS = '/home/guanl/Desktop/MSP/BCROMS'
output_path_BCROMS = '/home/guanl/Desktop/MSP/BCROMS/'
file_path_Climatology = '/home/guanl/Desktop/MSP/Climatology'
output_path_Climatology = '/home/guanl/Desktop/MSP/Climatology/'


#----------------------------------- Read ROMS output------------------------------------------------------------
in_file_spr = 'Validation/Roms37_bcc42era5_clm81to10_sprAMJ.nc'
in_file_sum = 'Validation/Roms37_bcc42era5_clm81to10_sumJAS.nc'
in_file_fal = 'Validation/Roms37_bcc42era5_clm81to10_falOND.nc'
in_file_win = 'Validation/Roms37_bcc42era5_clm81to10_winJFM.nc'



roms_output_spr = os.path.join(file_path_BCROMS, in_file_spr)
roms_output_sum = os.path.join(file_path_BCROMS, in_file_sum)
roms_output_fal = os.path.join(file_path_BCROMS, in_file_fal)
roms_output_win = os.path.join(file_path_BCROMS, in_file_win)

#----------------------------------------------------------------------------------------------------------------------------
def read_BCROMS(season, index): # index - temp, salt, zeta
    #function to read the ROMS output .nc files
    file_path_BCROMS = '/home/guanl/Desktop/MSP/BCROMS'
    if season == 'spr':
        in_file = 'Validation/Roms37_bcc42era5_clm81to10_'+ season + 'AMJ.nc'
    elif season == 'sum':
        in_file = 'Validation/Roms37_bcc42era5_clm81to10_' + season + 'JAS.nc'
    elif season == 'fal':
        in_file = 'Validation/Roms37_bcc42era5_clm81to10_' + season + 'OND.nc'
    elif season == 'win':
        in_file = 'Validation/Roms37_bcc42era5_clm81to10_' + season + 'JFM.nc'
    else:
        print('wrong season input')
    roms_output = os.path.join(file_path_BCROMS, in_file)
    nc = xr.open_dataset(roms_output)
    grid_depth = nc.depth.values
    n_grid_depth = nc.temp.values.shape[0]

    if index == 'temp':
        data_dict = dict()
        data_dict['grid_depth'] = grid_depth
        data_dict['y_lat'] = nc.temp.lat_rho.values
        data_dict['x_lon'] = nc.temp.lon_rho.values

    # write index for each grid depth
        for j in range(0, n_grid_depth, 1):
            T_temp = np.array([])
            variable_name = 'grid_depth_' + str(int(abs(grid_depth[j]))) + 'm'
            T_temp = nc.temp.values[j, :, :]
            data_dict[variable_name] = T_temp

    elif index == 'salt':
        data_dict = dict()
        data_dict['grid_depth'] = grid_depth
        data_dict['y_lat'] = nc.salt.lat_rho.values
        data_dict['x_lon'] = nc.salt.lon_rho.values

        # write index for each grid depth
        for j in range(0, n_grid_depth, 1):
            S_temp = np.array([])
            variable_name = 'grid_depth_' + str(int(abs(grid_depth[j]))) + 'm'
            S_temp = nc.salt.values[j, :, :]
            data_dict[variable_name] = S_temp

    elif index == 'zeta':
        data_dict = dict()
        data_dict['grid_depth'] = grid_depth
        data_dict['y_lat'] = nc.zeta.lat_rho.values
        data_dict['x_lon'] = nc.zeta.lon_rho.values

        # write index for each grid depth
        for j in range(0, n_grid_depth, 1):
            Z_temp = np.array([])
            variable_name = 'grid_depth_' + str(int(abs(grid_depth[j]))) + 'm'
            Z_temp = nc.zeta.values[j, :, :]
            data_dict[variable_name] = Z_temp

    else:
        print('input index is not available')

    return data_dict

# call function
roms = read_BCROMS(season = 'spr', index = 'temp')



#--------------------------------------------------------------------------------------------------------------------------------------------
left_lon, right_lon, bot_lat, top_lat = [-140, -120, 43, 57]   # BCROMS

def plot_BCROMS(data_dict, depth, index, left_lon, right_lon, bot_lat, top_lat):
    #function to plot ROMS output at certain depths, using the output from read_BCROMS()
    # get triangular mesh information
    lon = data_dict['x_lon'] + 360
    lat = data_dict['y_lat']

    # create basemap
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat,
                projection='lcc',  # width=40000, height=40000, #lambert conformal project
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

    xpt, ypt = m(lon, lat)  # convert lat/lon to x/y map projection coordinates in meters
    var_name = 'grid_depth_' + depth + 'm'
    var = np.array(data_dict[var_name]) #get the index values at certain depth

    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    m.drawcoastlines(linewidth=0.2)
    # m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    # m.scatter(xpt, ypt, color='0.7', lw = 0.2)
    #cax = plt.pcolor(xpt, ypt, var, cmap='YlOrBr', edgecolors='none')
    if index == 'temp':
        cax = plt.pcolor(xpt, ypt, var, cmap='YlOrBr', edgecolors='none', vmin=np.nanmin(var), vmax=np.nanmax(var))
        plt.clim(0, 22)
    elif index == 'salt':
        cax = plt.pcolor(xpt, ypt, var, cmap='Blues', edgecolors='none', vmin=np.nanmin(var), vmax=np.nanmax(var))
        plt.clim(0, 36)
    else:
        print('wrong variable input')
   # masked_array = np.ma.array(temp_5, mask=np.isnan(temp_5)) #mask the nan values
    color_map = plt.cm.get_cmap()
    color_map.set_bad('w') #set the nan values to white on the plot

    cbar = fig.colorbar(cax, shrink=0.7, extend='both') #set scale bar
    if index == 'temp':
        cbar.set_label('Temperature [°C]', size=14) #scale label
    elif index == 'salt':
        cbar.set_label('Salinity [psu]', size=14)  # scale label
    elif index == 'zeta':
        cbar.set_label('Elevation', size=14)  # scale label
    else:
        print('the input variable is not available')

    parallels = np.arange(bot_lat-1, top_lat+1, 3.) #  parallels = np.arange(48., 54, 0.2), parallels = np.linspace(bot_lat, top_lat, 10)
    m.drawparallels(parallels, labels=[True, False, False, False])  #draw parallel lat lines
    meridians = np.arange(-140, -120.0, 5.) # meridians = np.linspace(int(left_lon), right_lon, 5)
    m.drawmeridians(meridians, labels=[False, False, False, True])
    # labels = [left,right,top,bottom]
    #title_name = "Climatology_T_" + season + "_" + depth + 'm'
    #plt.title(title_name, size=15, y=1.08)
    plt.show()


#call function
plot_BCROMS(data_dict = roms, depth = '0', index = 'temp', left_lon = -140, right_lon = -120, bot_lat = 43, top_lat = 57)


#---------------------------------------------------------------------------------------------------------------------------------------
def read_obs_climatologies(season, index, output_folder):
    #define function to read observational climatology
    file_path_obs_Climatology = '/home/guanl/Desktop/MSP/Climatology'
    grid_filename = os.path.join(file_path_obs_Climatology , 'nep35_reord_latlon_wgeo.ngh')
    tri_filename = os.path.join(file_path_obs_Climatology , 'nep35_reord.tri')

    data = np.genfromtxt(grid_filename, dtype="i8,f8,f8, i4, f8, i4, i4, i4, i4, i4, i4, i4",
                          names=['node', 'lon', 'lat', 'type', 'depth',
                                's1', 's2', 's3', 's4', 's5', 's6'],
                          delimiter="", skip_header=3)

    tri_data = np.genfromtxt(tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3))-1 #python starts from 0
    if index == 'temp':
        array_filename = os.path.join(file_path_obs_Climatology, output_folder + '/nep35_tem_' + season + '_extrap2_reformat.npy')
        array = np.load(array_filename)
        array_t = array[1:]
        array_t = np.transpose(array_t)
        grid_depth = abs(array[0])
        array_t = np.vstack((array_t, data['depth']))
        for i in range(0, 51, 1):
            array_t[i] = np.where(array_t[52] < grid_depth[i], np.nan, array_t[i]) #replace the value below bottom depth with nan

        # create a data dictionary, and write data into dictionary
        data_dict = dict()
        data_dict['node_number'] = data['node'] - 1 # use node_number as Key
        data_dict['depth_in_m'] = data['depth']
        data_dict['y_lat'] = data['lat']
        data_dict['x_lon'] = data['lon']
        data_dict['grid_depth'] = abs(array[0])

        #write index for each grid depth
        for i in range(0, 52, 1):
            variable_name = 'grid_depth_' + str(int(abs(grid_depth[i]))) + 'm'
            data_dict[variable_name] = array_t[i]
    elif index == 'salt':
        array_filename = os.path.join(file_path_obs_Climatology, output_folder + '/nep35_sal_' + season + '_extrap2_reformat.npy')
        array = np.load(array_filename)
        array_t = array[1:]
        array_t = np.transpose(array_t)
        grid_depth = abs(array[0])
        array_t = np.vstack((array_t, data['depth']))
        for i in range(0, 51, 1):
            array_t[i] = np.where(array_t[52] < grid_depth[i], np.nan,
                                  array_t[i])  # replace the value below bottom depth with nan

        # create a data dictionary, and write data into dictionary
        data_dict = dict()
        data_dict['node_number'] = data['node'] - 1  # use node_number as Key
        data_dict['depth_in_m'] = data['depth']
        data_dict['y_lat'] = data['lat']
        data_dict['x_lon'] = data['lon']
        data_dict['grid_depth'] = abs(array[0])

        # write index for each grid depth
        for i in range(0, 52, 1):
            variable_name = 'grid_depth_' + str(int(abs(grid_depth[i]))) + 'm'
            data_dict[variable_name] = array_t[i]
    else:
        print('input variable is not available')

    tri = mtri.Triangulation(data_dict['x_lon'], data_dict['y_lat'], tri_data) # attributes: .mask, .triangles, .edges, .neighbors
    #min_circle_ratio = 0.1
    #mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
    #tri.set_mask(mask)
    data_dict['triangles'] = tri.triangles
    plt.triplot(tri, color='0.7', lw = 0.2)  #check grid plot
    plt.show()

    return data_dict

#run function
obs = read_obs_climatologies(season = 'spr', index = 'temp', output_folder = 'T_spr')



#---------------------------------------------------------------------------------------------------------------------------------------
left_lon, right_lon, bot_lat, top_lat = [-140, -120, 43, 57]

def Obs_to_ROMS_grid(data_dict_roms, data_dict_obs, index, season, depth, left_lon, right_lon, bot_lat, top_lat):
    #function to project observational climatology to ROMS grid
    file_path_NEP_Climatology = '/home/guanl/Desktop/MSP/Climatology'
    tri_filename = os.path.join(file_path_NEP_Climatology, 'nep35_reord.tri')
    tri_data = np.genfromtxt(tri_filename, skip_header=0, skip_footer=0, usecols=(1, 2, 3)) - 1
    x_lon_roms = data_dict_roms['x_lon'] + 360
    y_lat_roms = data_dict_roms['y_lat']

    # create basemap
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat,
                projection='lcc',  # width=40000, height=40000, #lambert conformal project
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

    x_roms, y_roms = m(x_lon_roms, y_lat_roms)  # convert lat/lon to x/y map projection coordinates in meters using basemap

    # 2nd method to convert lat/lon to x/y
    # import pyproj
    # proj_basemap = pyproj.Proj(m.proj4string) # find out the basemap projection
    # t_lon, t_lat = proj_basemap(x_lon_g, y_lat_g)

    # get triangular mesh information
    x_lon_obs = data_dict_obs['x_lon']
    y_lat_obs = data_dict_obs['y_lat']
    x_obs, y_obs = m(x_lon_obs, y_lat_obs)  # convert lat/lon to x/y map projection coordinates in meters
    tri_obs = mtri.Triangulation(x_obs, y_obs, tri_data)
    trifinder = tri_obs.get_trifinder()  # trifinder= mtri.Triangulation.get_trifinder(tri_pt), return the default of this triangulation

    var_name = 'grid_depth_' + depth + 'm'
    var = np.array(data_dict_obs[var_name]) # pull the values from observational climatology

    # interpolate from triangular to regular mesh
    interp_lin = mtri.LinearTriInterpolator(tri_obs, var, trifinder=None)  # conduct interpolation on lcc projection, not on lat/long
    var_new = interp_lin(x_roms, y_roms) #get the variable value at the new locations
    var_new[var_new.mask] = np.nan  # set the value of masked point to nan

    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    m.drawcoastlines(linewidth=0.2)
    # m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    # m.scatter(xpr, ypr, color='black')
    # cax = plt.pcolor(xpt, ypt, var_r, cmap='YlOrBr', edgecolors='none', vmin=np.nanmin(var_r), vmax=np.nanmax(var_r))
    if index == 'temp':
        cax = plt.pcolor(x_roms, y_roms, var_new, cmap='YlOrBr', edgecolors='none', vmin=np.nanmin(var_new),
                         vmax=np.nanmax(var_new))
        plt.clim(0, 22)
    elif index == 'salt':
        cax = plt.pcolor(x_roms, y_roms, var_new, cmap='Blues', edgecolors='none', vmin=np.nanmin(var_new),
                         vmax=np.nanmax(var_new))
        plt.clim(0, 36)
    else:
        print('wrong variable input')
    # masked_array = np.ma.array(temp_5, mask=np.isnan(temp_5)) #mask the nan values
    color_map = plt.cm.get_cmap()
    color_map.set_bad('w')  # set the nan values to white on the plot
    cbar = fig.colorbar(cax, shrink=0.7, extend='both')  # set scale bar
    if index == 'temp':
        cbar.set_label('Temperature [°C]', size=14)  # scale label
    elif index == 'salt':
        cbar.set_label('Salinity [psu]', size=14)  # scale label
    elif index == 'zeta':
        cbar.set_label('Elevation', size=14)  # scale label
    else:
        print('input variable is not available')
    parallels = np.arange(bot_lat - 1, top_lat + 1,3.)  # parallels = np.arange(48., 54, 0.2), parallels = np.linspace(bot_lat, top_lat, 10)
    m.drawparallels(parallels, labels=[True, False, False, False])  # draw parallel lat lines
    meridians = np.arange(-140, -120.0, 5.)  # meridians = np.linspace(int(left_lon), right_lon, 5)
    m.drawmeridians(meridians, labels=[False, False, True, True])
    # labels = [left,right,top,bottom]
    # title_name = "Climatology_T_" + season + "_" + depth + 'm'
    # plt.title(title_name, size=15, y=1.08)

    plt.show()
    # png_name = os.path.join(file_path, output_folder + '/T_' + season + '_reg_' + depth + 'm.png')
    # fig.savefig(png_name, dpi=400)
    # plt.savefig(png_name, dpi=400)

    # save the lat, lon and var on regular grid
    data_dict_new = dict()
    data_dict_new['x_lon'] = x_lon_roms - 360
    data_dict_new['y_lat'] = y_lat_roms
    data_dict_new[var_name] = var_new
    return data_dict_new

#run function

obs_to_roms = Obs_to_ROMS_grid(data_dict_roms = roms, data_dict_obs = obs, index = 'temp', season = 'spr', depth = '0', left_lon = -140, right_lon = -120, bot_lat = 43, top_lat = 57)



#---------------------------  Plot frequency differences of Diff  -----------------------------------------------------------------------------------
def plot_roms_obs_diff_freq(data_dict_roms, data_dict_obs_to_roms, depth, index, season):
    var_name = 'grid_depth_' + depth + 'm'
    var_roms = np.array(data_dict_roms[var_name])
    var_obs_to_roms = np.array(data_dict_obs_to_roms[var_name])
    diff = var_roms - var_obs_to_roms
    # size frequency of differences
    Diff = np.array([])
    for i in range(0, 410, 1):
        temp = diff[i , :]
        Diff = np.concatenate((Diff, temp), axis = 0)
    plt.hist(Diff, bins = 50)
    plt.show()

plot_roms_obs_diff_freq(data_dict_roms = roms, data_dict_obs_to_roms = obs_to_roms, depth = '0', index = 'temp', season = 'spr')



#------------------------------------------------------------------------------------------------------------
def plot_roms_obs_diff(data_dict_roms, data_dict_obs_to_roms, depth, index, season):
    #function to calculate the differences between the observational climatology and ROMS climatology, and plot the result
    var_name = 'grid_depth_' + depth + 'm'
    var_roms = np.array(data_dict_roms[var_name])
    var_obs_to_roms = np.array(data_dict_obs_to_roms[var_name])
    diff = var_roms - var_obs_to_roms

    lon = data_dict_roms['x_lon'] + 360
    lat = data_dict_roms['y_lat']

    # create basemap
    m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
                urcrnrlon=right_lon, urcrnrlat=top_lat,
                projection='lcc',  # width=40000, height=40000, #lambert conformal project
                resolution='h', lat_0=0.5 * (bot_lat + top_lat),
                lon_0=0.5 * (left_lon + right_lon))  # lat_0=53.4, lon_0=-129.0)

    xpt, ypt = m(lon, lat)  # convert lat/lon to x/y map projection coordinates in meters

    fig = plt.figure(num=None, figsize=(8, 6), dpi=100)
    m.drawcoastlines(linewidth=0.2)
    # m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='0.8')
    # m.scatter(xpt, ypt, color='0.7', lw = 0.2)
    #cax = plt.pcolor(xpt, ypt, var, cmap='YlOrBr', edgecolors='none')
    cax = plt.pcolor(xpt, ypt, diff, cmap='RdBu_r', edgecolors='none', vmin=np.nanmin(diff), vmax=np.nanmax(diff))
    plt.clim(-1, 1)

   # masked_array = np.ma.array(temp_5, mask=np.isnan(temp_5)) #mask the nan values
    color_map = plt.cm.get_cmap()
    color_map.set_bad('w') #set the nan values to white on the plot

    cbar = fig.colorbar(cax, shrink=0.7, extend='both') #set scale bar
    if index == 'temp':
        cbar.set_label('Temperature [°C]', size=14) #scale label
    elif index == 'salt':
        cbar.set_label('Salinity [psu]', size=14)  # scale label
    elif index == 'zeta':
        cbar.set_label('Elevation', size=14)  # scale label
    else:
        print('the input variable is not available')

    parallels = np.arange(bot_lat-1, top_lat+1, 3.) #  parallels = np.arange(48., 54, 0.2), parallels = np.linspace(bot_lat, top_lat, 10)
    m.drawparallels(parallels, labels=[True, False, False, False])  #draw parallel lat lines
    meridians = np.arange(-140, -120.0, 5.) # meridians = np.linspace(int(left_lon), right_lon, 5)
    m.drawmeridians(meridians, labels=[False, False, False, True])
    #labels = [left,right,top,bottom]
    title_name = 'Differences_in_climatologies_'+ index + '_' + season + "_" + depth + 'm'
    plt.title(title_name, size=15, y=1.08)
    plt.show()


left_lon, right_lon, bot_lat, top_lat = [-140, -120, 43, 57]
plot_roms_obs_diff(data_dict_roms = roms, data_dict_obs_to_roms = obs_to_roms, depth = '0', index = 'temp', season = 'spr')


#--------------------------------------------------------  Main functions  -----------------------------------------------------------------------
roms = read_BCROMS(season = 'fal', index = 'salt')
obs = read_obs_climatologies(season = 'fal', index = 'salt', output_folder = 'S_fal')

plot_BCROMS(data_dict = roms, depth = '80', index = 'salt', left_lon = -140, right_lon = -120, bot_lat = 43, top_lat = 57)

obs_to_roms = Obs_to_ROMS_grid(data_dict_roms = roms, data_dict_obs = obs, index = 'salt', season = 'fal', depth = '80', left_lon = -140, right_lon = -120, bot_lat = 43, top_lat = 57)
plot_roms_obs_diff_freq(data_dict_roms = roms, data_dict_obs_to_roms = obs_to_roms, depth = '80', index = 'salt', season = 'fal')

left_lon, right_lon, bot_lat, top_lat = [-140, -120, 43, 57]
plot_roms_obs_diff(data_dict_roms = roms, data_dict_obs_to_roms = obs_to_roms, depth = '80', index = 'salt', season = 'fal')

#plot_roms_obs_diff(data_dict_roms = roms, data_dict_obs_to_roms = obs_to_roms, depth = '0', index = 'temp', season = 'spr', left_lon = -140, right_lon = -120, bot_lat = 43, top_lat = 57)



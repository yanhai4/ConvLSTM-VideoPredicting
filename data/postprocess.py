# -*- coding: utf-8 -*-
# @File : postprocess.py
# @Author : 秦博
# @Time : 2022/07/16
import cmaps
import cartopy
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# conda install -c conda-forge cartopy

def get_climatology(path, scope=[1440, 1800]):
    sst = np.array(nc.Dataset(path, mode='r').variables['sst'])[range(scope[0], scope[1])]
    sst[sst == -1e+30] = 0
    sst[sst == -1000] = 0
    climatology = []
    for i in range(12):
        climatology.append(np.mean(sst[i::12], axis=0))
    return climatology

def get_land(path):
    sst = np.array(nc.Dataset(path, mode='r').variables['sst'])[-1]
    land = sst == -1e+30
    return land

def plot_helper(data, lons, lats, climatology, land, save=True, filename='pred.png'):
    data = data - climatology
    # data[data > 5.0] = 5.0
    # data[data < -5.0] = -5.0
    data[land] = np.nan
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
    # lons = lons - 180
    data = np.concatenate((data[:, 180:360], data[:, 0:180]), axis=1)
    fig.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    m = ax.contourf(lons, lats, data, 60, transform=ccrs.PlateCarree(central_longitude=180), cmap=cmaps.GMT_panoply, vmin =-5, vmax =5)

    posn = ax.get_position()
    cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0, 0.01, posn.height])

    ax.coastlines()
    plt.colorbar(m, cax=cbar_ax)
    if save:
        plt.savefig(filename)
        plt.cla(); plt.clf(); plt.close()


# 辅助计算各种nino指数
def index_helper(data, climatology):
    data = data - climatology
    nino3 = np.mean(data[85:95, 30:90])
    nino4 = np.mean(np.concatenate((data[85:95, 340:360], data[85:95, :30]), axis=1))
    nino34 = np.mean(data[85:95, 10:60])
    return nino3, nino4, nino34


if __name__ == '__main__':
    path = './file/HadISST_sst.nc'
    ncfile = nc.Dataset(path, mode='r')
    sst = np.array(ncfile.variables['sst'])
    lons = np.array(ncfile.variables['longitude'][:])
    lats = np.array(ncfile.variables['latitude'][:])
    sst[sst == -1e+30] = 0
    sst[sst == -1000] = 0
    plot_helper(sst[-1], lons, lats, get_climatology(path)[-1], get_land(path), save=True)
    plt.show()

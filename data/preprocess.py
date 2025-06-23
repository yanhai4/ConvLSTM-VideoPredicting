# -*- coding: utf-8 -*-
# @File : preprocess.py
# @Author : 秦博
# @Time : 2022/07/16
import numpy as np
import netCDF4 as nc

from sklearn.preprocessing import MinMaxScaler


def get_scaler(path):
    sst = np.array(nc.Dataset(path, mode='r').variables['sst'])
    sst[sst == -1e+30] = 0
    sst[sst == -1000] = 0
    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, 180*360))), (-1, 180, 360))
    return scaler


if __name__ == '__main__':
    path = r'file/data1.nc'
    sst = np.array(nc.Dataset(path, mode='r').variables['thetao'])
    sst[sst == -1e+30] = 0
    sst[sst == -1000] = 0
    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst, (-1, 180*360))), (-1, 180, 360))
    np.savez('./file/data.npz', sst=sst)

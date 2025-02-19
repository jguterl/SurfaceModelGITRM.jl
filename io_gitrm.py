# -*- coding: utf-8 -*-


import numpy as np
from math_helper import *
from typing import Callable
import matplotlib.pyplot as plt
import pydoc
import netCDF4
import os


import netCDF4


def check_particles(particles):
    for k, v in particles.items():
        if k != 'Np' and type(v) == np.ndarray:
            assert len(v.shape) == 1 and v.shape[0] == particles['np'], 'Wrong dimension for particle attribute {}  with shape {}. Np={}'.format(
                k, v.shape, particles['np'])


def write_particle_source(fn: str, particles: dict):
    """
    write particle source
    """
    check_particles(particles)
    ncfile = netCDF4.Dataset(fn, 'w', format='NETCDF4')
    ncfile.createDimension('nP', particles['np'])
    for k in particles.keys():
        if type(particles[k]) == np.ndarray:
            if k == 'globalId':
                var = ncfile.createVariable(k, 'i4', 'nP')
                var[:] = particles[k]
            else:
                var = ncfile.createVariable(k, 'f8', 'nP')
                var[:] = particles[k]

    ncfile.close()
    


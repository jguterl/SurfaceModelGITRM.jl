"""
Created on Wed Nov 15 17:39:55 2023

@author: nathd
"""
import numpy as np
import netCDF4

data=np.load('/users/nathd/pyGITRM/test_167196_bgp_mesh.npy', allow_pickle=True).tolist()
key_list=[k for k in data.keys() if k!='mesh']

#['KNBS', 'KTEBS', 'KTIBS', 'KVHS', 'KES', 'TEGS', 'TIGS']
#["ne", "te", "ti", "Velocity", "efield","gradTe","gradTi"];
#Fields
ne=np.array(data['KNBS'])
te=np.array(data['KTEBS'])
ti=np.array(data['KTIBS'])
v_t=np.array(data['KVHS'])
e_f=np.array(data['KES'])
gradTe = 0*e_f
gradTi = 0*e_f

# Create NETCDF file
profiles_filename = "profiles_OEDGE_dimes.nc"
rootgrp = netCDF4.Dataset(profiles_filename, "w", format="NETCDF4")
#Dimesnion
nx=np.shape(ne)[0]
ny=1
nrr = rootgrp.createDimension("nR", nx)
nzz = rootgrp.createDimension("nZ", ny)

#Electron Density
nee = rootgrp.createVariable("ne", "f8", ("nZ", "nR"))
nee[:]=ne

#Ion Density
nii = rootgrp.createVariable("ni", "f8", ("nZ", "nR"))
nii[:]=ne

#Electron Temperature
tee = rootgrp.createVariable("te", "f8", ("nZ", "nR"))
tee[:]=te

#Ion Temperature
tii = rootgrp.createVariable("ti", "f8", ("nZ", "nR"))
tii[:]=ti

rootgrp.close()
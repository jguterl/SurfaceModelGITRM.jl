#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:00:21 2023

@author: nathd
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from rdp import rdp


data=np.load('/users/nathd/pyGITRM/test_167196_contour.npy', allow_pickle=True).tolist()
key_list=[k for k in data.keys()]


epsilon=0.5

plt.close('all')
fig = plt.figure()

wall=np.array(data['wall'])
plt.plot(wall[:,0], wall[:,1], '-ok')
wall_new=rdp(wall, epsilon)
plt.plot(wall_new[:,0], wall_new[:,1], '-sr', markersize=12, markerfacecolor='none')
wall_nx=np.array([wall_new[:,0], np.zeros(np.shape(wall_new)[0]),wall_new[:,1]])
#np.savetxt('wall_nx.txt', np.transpose(wall_nx))

core=np.array(data['core'])
plt.plot(core[:,0], core[:,1], '-ob')
core_new=rdp(core, epsilon)
plt.plot(core_new[:,0], core_new[:,1], '-sr', markersize=12, markerfacecolor='none')
core_nx=np.array([core_new[:,0], np.zeros(np.shape(core_new)[0]),core_new[:,1]])
np.savetxt('core_nx.txt', np.transpose(core_nx))

outer=np.array(data['outer'])
plt.plot(outer[:,0], outer[:,1],'-og')
outer_new=rdp(outer, epsilon)
plt.plot(outer_new[:,0], outer_new[:,1], '-sr', markersize=12, markerfacecolor='none')
outer_nx=np.array([outer_new[:,0], np.zeros(np.shape(outer_new)[0]),outer_new[:,1]])
np.savetxt('outer_nx.txt', np.transpose(outer_nx))

sep=np.array(data['sep'])
plt.plot(sep[:,0], sep[:,1],'-oy')
sep_new=rdp(sep, epsilon)
plt.plot(sep_new[:,0], sep_new[:,1], '-sr', markersize=12, markerfacecolor='none')
sep_nx=np.array([sep_new[:,0], np.zeros(np.shape(sep_new)[0]),sep_new[:,1]])
np.savetxt('sep_nx.txt', np.transpose(sep_nx))
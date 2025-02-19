#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:03:09 2023

@author: jeromeguterl
"""
import numpy as np
import libconf
import io
import click
import os
import gmsh
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import collections as mc


def Cross(V1, V2):
    CrossVec = np.copy(V1)
    CrossVec[:, 0] = V1[:, 1]*V2[:, 2]-V1[:, 2]*V2[:, 1]
    CrossVec[:, 1] = V1[:, 2]*V2[:, 0]-V1[:, 0]*V2[:, 2]
    CrossVec[:, 2] = V1[:, 0]*V2[:, 1]-V1[:, 1]*V2[:, 0]
    return CrossVec


class PlotSurfaceElements():
    '''Subclass for plotting methods.'''

    def _Plot(self, GroupID=None, ax=None, cmap_name='viridis', ElemAttr=None, Alpha=0.1, EdgeColor='k', FaceColor='b'):
        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')
        else:
            self.ax = ax
        cmap = mpl.cm.get_cmap(cmap_name)               # Get colormap by name
        # c = cmap(mpl.colors.Normalize(vmin, vmax)(v))   # Normalize value and get color
        Idx = self.GetGroupIdx(GroupID)
        # Create PolyCollection from coords
        pc = Poly3DCollection(
            self.Triangles[Idx, :, :], cmap=mpl.cm.jet, alpha=0.4)

        # Color set by values of element attribute if ElemAttr not None (e.g. ElemAttr='Z')
        if ElemAttr is not None and self.GeomInput.get(ElemAttr) is not None:
            pc.set_array(self.GeomInput.get(ElemAttr))
        else:

            pc.set_facecolor(FaceColor)
        pc.set_edgecolor(EdgeColor)
        pc.set_alpha(Alpha)

        # Add PolyCollection to axes
        self.ax.add_collection3d(pc)
        return pc

    def Plot(self, GroupID=None, ax=None, **kwargs):

        self._Plot(GroupID, ax, **kwargs)
        self.CurrentGroupID = GroupID
        self.SetAxisLim()
        self.ax.set_zlabel('Z-Axis')
        self.ax.set_xlabel('X-Axis')
        self.ax.set_ylabel('Y-Axis')

    def ShowCentroids(self, ax=None, GroupID=None, annoted=False):
        Idx = self.GetGroupIdx(GroupID)
        self.ax.scatter(self.Centroid[Idx, 0], self.Centroid[Idx, 1],
                        self.Centroid[Idx, 2], marker='o', color='b')
        if annoted:
            for i in Idx:
                # print(index)
                self.ax.text(self.Centroid[i, 0], self.Centroid[i, 1],
                             self.Centroid[i, 2], str(i), color='red')

    def ShowNormals(self, GroupID=None, ax=None, L=0.002, Color='b'):
        c = self.Centroid
        v = self.Normal

        Idx = self.GetGroupIdx(GroupID)
        self.ax.quiver(c[Idx, 0], c[Idx, 1], c[Idx, 2], v[Idx, 0],
                       v[Idx, 1], v[Idx, 2], length=L, normalize=True, color=Color)

    def SetAxisLim(self, square=True):
        xmin = np.min(self.Triangles[:, :, 0])
        xmax = np.max(self.Triangles[:, :, 0])
        ymin = np.min(self.Triangles[:, :, 1])
        ymax = np.max(self.Triangles[:, :, 1])
        zmin = np.min(self.Triangles[:, :, 2])
        zmax = np.max(self.Triangles[:, :, 2])
        mn = min([xmin, ymin, zmin])
        mx = max([xmax, ymax, zmax])
        if not square:
            self.ax.set_xlim3d(xmin, xmax)
            self.ax.set_ylim3d(ymin, ymax)
            self.ax.set_zlim3d(zmin, zmax)
        else:
            self.ax.set_xlim3d(mn, mx)
            self.ax.set_ylim3d(mn, mx)
            self.ax.set_zlim3d(mn, mx)

    def ShowInDir(self, GroupID=None, ax=None, L=0.005, Color='g'):
        if ax is None:
            ax = self.ax
        if ax is None:
            ax = plt.gca()
        c = self.Centroid
        v = self.normalVec
        n = self.GeomInput['inDir']

        Idx = self.GetGroupIdx(GroupID)
        if self.Verbose:
            print('InDir Idx:', Idx)
        ax.quiver(c[Idx, 0], c[Idx, 1], c[Idx, 2], v[Idx, 0]*n[Idx], v[Idx, 1]
                  * n[Idx], v[Idx, 2]*n[Idx], length=5*L, normalize=True, color=Color)
        plt.show()
        # ax.add_collection(lc)


class SurfaceElements(PlotSurfaceElements):
    '''  '''

    def __init__(self, filename):
        with open(filename, "r") as f:
            self.geom_data = libconf.load(f)['geom']
            self.set_triangles()
            self.set_centroids()
            self.set_normals()
            self.set_groups()
            self.global_id = self.geom_data['globalId']
            self.geom_id = self.geom_data['geomId']

    def set_triangles(self):
        N = len(self.geom_data['x1'])

        s_coords1 = ['x1', 'y1', 'z1']
        self.v1 = np.vstack([np.array(self.geom_data[s]) for s in s_coords1]).T
        s_coords3 = ['x3', 'y3', 'z3']
        self.v3 = np.vstack([np.array(self.geom_data[s]) for s in s_coords3]).T
        s_coords2 = ['x2', 'y2', 'z2']
        self.v2 = np.vstack([np.array(self.geom_data[s]) for s in s_coords2]).T
        self.Triangles = np.zeros((N, 3, 3))
        for i in range(N):
            self.Triangles[i, 0, :] = self.v1[i]
            self.Triangles[i, 1, :] = self.v2[i]
            self.Triangles[i, 2, :] = self.v3[i]

    def set_centroids(self):
        self.Centroid = 1/3*(np.sum(self.Triangles, 1)).squeeze()

    def set_normals(self):
        A = self.Triangles[:, 0, :].squeeze()
        B = self.Triangles[:, 1, :].squeeze()
        C = self.Triangles[:, 2, :].squeeze()

        AB = B-A
        AC = C-A
        self.Normal = Cross(-AB, AC)

    def set_groups(self):
        self.GroupID = np.unique(self.geom_data['geomId'])
        print("Groups ID:", self.GroupID)

    def GetGroupIdx(self, GroupID):
        if GroupID is None:
            if self.CurrentGroupID is None:
                return np.array([i for i in range(len(self.geom_data['geomId']))])
            else:
                GroupID = self.CurrentGroupID

        if type(GroupID) == int:
            GroupID = [GroupID]

        self.CurrentGroupID = GroupID
        return np.array([i for i in range(len(self.geom_data['geomId'])) if self.geom_data['geomId'][i] in GroupID])


# def plot_triangles(self, GroupID=None, ax=None, cmap_name='viridis', ElemAttr=None, alpha=0.1, edgeColor='k', facecolor=None):
#     if ax is None:
#         fig = plt.figure(figsize=(4, 4))
#         ax = fig.add_subplot(111, projection='3d')
#     cmap = mpl.cm.get_cmap(cmap_name)               # Get colormap by name
#     # c = cmap(mpl.colors.Normalize(vmin, vmax)(v))   # Normalize value and get color
#     # Idx = self.GetGroupIdx(GroupID)
#     # Create PolyCollection from coords
#     pc = Poly3DCollection(
#         self.Triangles, cmap=mpl.cm.jet, alpha=0.4)

#     # Color set by values of element attribute if ElemAttr not None (e.g. ElemAttr='Z')
#     if ElemAttr is not None and self.GeomInput.get(ElemAttr) is not None:
#         pc.set_array(self.GeomInput.get(ElemAttr))
#     else:
#         pc.set_facecolor(FaceColor)
#     pc.set_edgecolor(EdgeColor)
#     pc.set_alpha(Alpha)
#     # Add PolyCollection to axes
#     ax.add_collection3d(pc)
#     plt.show()
#     return pc
fe = SurfaceElements("/Users/jeromeguterl/development/gitrm/nathd_example.cfg")
fe.Plot(GroupID=547)
fe.ShowNormals()

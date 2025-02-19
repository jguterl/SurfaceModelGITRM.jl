#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:03:09 2023

@author: jeromeguterl
"""
import pathlib
import gzip
import numpy as np
import libconf
import io
import click
import os
import gmsh
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

    def _Plot(self, GroupId=None, ax=None, cmap_name='jet', ElemAttr=None, alpha=0.1, EdgeColor='k', FaceColor='b', show_groups=True):
        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')
        else:
            self.ax = ax
        cmap = mpl.cm.get_cmap(cmap_name)               # Get colormap by name
        # c = cmap(mpl.colors.Normalize(vmin, vmax)(v))   # Normalize value and get color
        if GroupId is None:
            GroupId = self.GroupId
        Idx = self.GetGroupIdx(GroupId)
        # Create PolyCollection from coords
        pc = Poly3DCollection(
            self.Triangles[Idx, :, :], cmap=cmap, alpha=0.4)

        # Color set by values of element attribute if ElemAttr not None (e.g. ElemAttr='Z')
        if ElemAttr is not None and self.GeomInput.get(ElemAttr) is not None:
            pc.set_array(self.GeomInput.get(ElemAttr))
        else:
            pc.set_facecolor(FaceColor)

        if show_groups:
            EdgeColor = None
            pc.set_array(self.geom_id[Idx])
            alpha = 0.3

        pc.set_edgecolor(EdgeColor)
        pc.set_alpha(alpha)
        # self.ax.legend()
        # Add PolyCollection to axes
        self.ax.add_collection3d(pc)
        # (cmin, cmax) = pc.get_clim()
        cmin = np.min(GroupId)
        cmax = np.max(GroupId)
        print(cmin, cmax)
        self.ax.legend(handles=[mpatches.Patch(alpha=alpha, color=cmap(
            (idx-cmin)/(cmax - cmin)), label=f'geom id:{idx}') for idx in GroupId])
        return pc

    def Plot(self, GroupId=None, ax=None, **kwargs):

        pc = self._Plot(GroupId, ax, **kwargs)
        self.CurrentGroupID = GroupId
        self.SetAxisLim()
        self.ax.set_zlabel('Z-Axis')
        self.ax.set_xlabel('X-Axis')
        self.ax.set_ylabel('Y-Axis')
        return pc

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
        c = self.centroid
        v = self.normal

        Idx = self.GetGroupIdx(GroupID)
        self.ax.quiver(c[Idx, 0], c[Idx, 1], c[Idx, 2], v[Idx, 0],
                       v[Idx, 1], v[Idx, 2], length=L, normalize=True, color=Color)

    def SetAxisLim(self, square=False):
        xmin = np.min(self.Triangles[:, :, 0])
        xmax = np.max(self.Triangles[:, :, 0])
        ymin = np.min(self.Triangles[:, :, 1])
        ymax = np.max(self.Triangles[:, :, 1])
        zmin = np.min(self.Triangles[:, :, 2])
        zmax = np.max(self.Triangles[:, :, 2])
        mn = np.min([xmin, ymin, zmin])
        mx = np.max([xmax, ymax, zmax])
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
        c = self.centroid
        v = self.normalVec
        n = self.GeomInput['inDir']

        Idx = self.GetGroupIdx(GroupID)
        if self.Verbose:
            print('InDir Idx:', Idx)
        ax.quiver(c[Idx, 0], c[Idx, 1], c[Idx, 2], v[Idx, 0]*n[Idx], v[Idx, 1]
                  * n[Idx], v[Idx, 2]*n[Idx], length=5*L, normalize=True, color=Color)
        plt.show()
        # ax.add_collection(lc)


class SurfaceElement():
    def __init__(self, elements, i):
        self.centroid = elements.centroid[i, :]
        self.normal = elements.normal[i, :]
        self.u1 = elements.u1[i, :]
        self.u2 = elements.u2[i, :]
        self.u3 = elements.u3[i, :]
        # self.vertices = elements.vertices[i, :, :]
        self.global_id = elements.global_id[i]

        self.model = None
        self.deposited_particles = None
        self.reflected_particles = None
        self.sputtered_particles = None
        self.p = None


class SurfaceElements(PlotSurfaceElements):
    '''  '''

    def __init__(self, filename=None):
        if filename is not None:
            self.import_mesh_element(filename)

    def import_mesh_elements(self, filename):
        ext = pathlib.Path("filename").suffix:
        if ext == '.cfg':
            self.import_mesh_elements_from_cfg(filename)
        elif ext == :
            ".msh":
            self.import_mesh_elements_from_msh(filename)
        else:
            raise ValueError()

    def import_mesh_elements_from_model(self, model) -> None:
        model = gmsh.model
        mesh = self.model.mesh
        elem_types, elem_tags, node_tags = mesh.getElements(dim=2)
        print('elem_types:{}\n elem_tags:{} \n node_tags:{}'.format(
            elem_types, elem_tags, node_tags))
        idx, points, param = mesh.getNodes(2, -1, True)
        self.vertices = np.asarray(points).reshape(-1, 3)
        self.vertice_tags = np.asarray(node_tags[0]).reshape(-1, 3)
        self.elements_tags = np.asarray(elem_tags[0])
        self.idx_vertices = np.zeros((int(np.max(idx))+1,), dtype=int)
        self.idx_elements = np.zeros(
            (int(np.max(self.elements_tags))+1,), dtype=int)
        for i in range(np.size(idx)):
            self.idx_vertices[idx[i]] = int(i)
        for i in range(np.size(self.elements_tags)):
            self.idx_elements[self.elements_tags[i]] = int(i)

        self.Triangles = self.vertices[self.idx_vertices[self.vertice_tags], :]
        self.n_elems = self.Triangles.shape[0]
        self.physical_groups = dict((self.model.get_physical_name(
            2, i), i) for (k, i) in self.model.get_physical_groups(2))
        self.GroupID = np.array([i for (k, i) in self.model.get_entities(2)])
        assert len(self.Triangles.shape) > 1 and self.Triangles.shape[1] == 3 and self.Triangles.shape[
            2] == 3, "Points must be a Nx3x3 numpy arrays: N_triangle x 3_vertices x (x_vertice,y_vertice,z_vertice)"
        print('Triangular mesh elements imported ... Number of triangular elements: {}'.format(
            self.Triangles.shape[0]))

        A = self.Triangles[:, 0, :].squeeze()
        B = self.Triangles[:, 1, :].squeeze()
        C = self.Triangles[:, 2, :].squeeze()

        AB = B-A
        AC = C-A
        BC = C-B
        BA = A-B
        CA = -AC
        CB = -BC

        dAB = Norm(AB)
        dBC = Norm(BC)
        dAC = Norm(AC)

        s = (dAB+dBC+dAC)/2
        self.area = np.sqrt(s*(s-dAB)*(s-dBC)*(s-dAC))
        self.normal = Cross(AB, AC)
        self.centroid = 1/3*(np.sum(self.Triangles, 1)).squeeze()

    def import_mesh_elements_cfg(self, filename):
        with open(filename, "r") as f:
            self.geom_data = libconf.load(f)['geom']

        self.global_id = np.array(self.geom_data['globalId'])
        self.geom_id = np.array(self.geom_data['geomId'])
        self.SetTriangles()
        self.SetCentroids()
        self.SetNormals()
        self.SetGroups()
        self.CurrentGroupID = None

    @staticmethod
    def isGZIP(filename):
        if filename.split('.')[-1] == 'gz':
            return True
        return False

    # Using HIGHEST_PROTOCOL is almost 2X faster and creates a file that
    # is ~10% smaller.  Load times go down by a factor of about 3X.
    def save(self, filename):
        if self.isGZIP(filename):
            f = gzip.open(filename, 'wb')
        else:
            f = open(filename, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    # Note that loading to a string with pickle.loads is about 10% faster
    # but probaly comsumes a lot more memory so we'll skip that for now.
    @classmethod
    def load(cls, filename):
        if cls.isGZIP(filename):
            f = gzip.open(filename, 'rb')
        else:
            f = open(filename, 'rb')
        n = pickle.load(f)
        f.close()
        return n

    def get_elements(self, GroupId):
        idx = self.GetGroupIdx(GroupId)
        return np.array([SurfaceElement(self, i)for i in idx])

    def SetTriangles(self):
        self.n_elems = len(self.geom_data['x1'])

        s_coords1 = ['x1', 'y1', 'z1']
        self.v1 = np.vstack([np.array(self.geom_data[s]) for s in s_coords1]).T
        s_coords3 = ['x3', 'y3', 'z3']
        self.v3 = np.vstack([np.array(self.geom_data[s]) for s in s_coords3]).T
        s_coords2 = ['x2', 'y2', 'z2']
        self.v2 = np.vstack([np.array(self.geom_data[s]) for s in s_coords2]).T
        self.u1 = self.v2 - self.v1
        self.u2 = self.v3 - self.v2
        self.u3 = self.v1 - self.v3
        self.Triangles = np.zeros((self.n_elems, 3, 3))
        for i in range(self.n_elems):
            self.Triangles[i, 0, :] = self.v1[i]
            self.Triangles[i, 1, :] = self.v2[i]
            self.Triangles[i, 2, :] = self.v3[i]

    def SetCentroids(self):
        self.centroid = 1/3*(np.sum(self.Triangles, 1)).squeeze()

    def SetNormals(self):
        A = self.Triangles[:, 0, :].squeeze()
        B = self.Triangles[:, 1, :].squeeze()
        C = self.Triangles[:, 2, :].squeeze()

        AB = B-A
        AC = C-A
        self.normal = Cross(-AB, AC)

    def SetGroups(self):
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


class Msh():
    def __init__(self):
        pass


def load_msh(self, fn):
    filename = "/Users/jeromeguterl/development/gitrm/mesh_generations/meshes/small_large_dots_DiMES_uniform_naming.msh"
    fn = os.path.abspath(filename)
    print('Loading {} ...'.format(fn))
    assert os.path.exists(fn), "Cannot read '{}'".format(fn)

    try:
        gmsh.finalize()
    except:
        pass
    finally:
        gmsh.initialize()

    gmsh.open(fn)
    return gmsh.model

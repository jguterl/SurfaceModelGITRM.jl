#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a box with DiMES at the bottom and geometrical elements on DiMES
Created on Fri Nov 17 14:08:42 2023

@author: jeromeguterl

"""
import gmsh
import numpy as np
import matplotlib.pyplot as plt
import math_helper
# Initialize gmsh session
gmsh.initialize()

# Define reference points in coordinate system (R,y,Z)
# Center of DiMES
pDiMES_center = Point(1.485, 0.0, -1, 25)

# Create the simulation box
outer_target = gmsh.model.occ.addBox()

# adjust last z
sep.Z[-1] = outer.Z[-1]
Z_Dimes = sep.Z[-1]
P_os = gmsh.model.occ.addPoint(sep.R[-1], 0, sep.Z[-1])
P_is = gmsh.model.occ.addPoint(sep.R[0], 0, sep.Z[0])
P_oo = gmsh.model.occ.addPoint(outer.R[-1], 0, outer.Z[-1])
P_io = gmsh.model.occ.addPoint(outer.R[0], 0, outer.Z[0])
gmsh.model.occ.synchronize()
outer_target = gmsh.model.occ.addLine(P_os, P_oo)
inner_target = gmsh.model.occ.addLine(P_is, P_io)
gmsh.model.occ.synchronize()
gmsh.model.occ.rotate([(1, inner_target)], 0, 0, 0, 0, 0, 1, np.pi/3)
gmsh.model.occ.rotate([(1, outer_target)], 0, 0, 0, 0, 0, 1, np.pi/3)
gmsh.model.occ.synchronize()
P_o = [gmsh.model.occ.addPoint(R, 0, Z) for (R, Z) in outer]
P_s = [gmsh.model.occ.addPoint(R, 0, Z) for (R, Z) in sep]
gmsh.model.occ.synchronize()
# line_o = [gmsh.model.occ.addLine(P_o[i], P_o[i+1]) for i in range(len(P_o)-1)]
# line_s = [gmsh.model.occ.addLine(P_s[i], P_s[i+1]) for i in range(len(P_s)-1)]

curve_outer = gmsh.model.occ.add_spline([l for l in P_o])
curve_sep = gmsh.model.occ.add_spline([l for l in P_s])
# [(1, l) for l in line_o]
# [(1, l) for l in line_s]
gmsh.model.occ.rotate([(1, curve_outer)], 0, 0, 0, 0, 0, 1, np.pi/3)
gmsh.model.occ.rotate([(1, curve_sep)], 0, 0, 0, 0, 0, 1, np.pi/3)

gmsh.model.occ.synchronize()
cloop = gmsh.model.occ.add_curve_loop(
    [curve_sep, inner_target, curve_outer, outer_target])
gmsh.model.occ.synchronize()
pl = gmsh.model.occ.add_plane_surface([cloop])
gmsh.model.occ.synchronize()

#
gmsh.model.occ.remove([(0, t) for t in P_o])
gmsh.model.occ.remove([(0, t) for t in P_s])
gmsh.model.occ.remove([(0, P_os), (0, P_is)])
gmsh.model.occ.remove([(0, P_oo), (0, P_io)])
gmsh.model.occ.remove([(1, inner_target), (1, outer_target)])
gmsh.model.occ.remove([(1, curve_sep), (1, curve_outer)])
gmsh.model.occ.synchronize()
v = gmsh.model.occ.revolve([(2, pl)], 0, 0, 0, 0, 0, 1, 2*np.pi/3)
gmsh.model.occ.synchronize()
# Create 2D surfaces
xDiMES = 1.485
yDiMES = 0.0
zDiMES = Z_Dimes
rDiMES = 0.025

TagDiMES0 = gmsh.model.occ.addDisk(xDiMES, yDiMES, zDiMES, rDiMES, rDiMES)

gmsh.model.occ.rotate([(2, TagDiMES0)], 0, 0, 0, 0, 0, 1, 2*np.pi/3)
# gmsh.fltk.run()
surface_outer_target_final = gmsh.model.occ.cut(
    [(2, 3)], [(2, TagDiMES0)], removeTool=False, removeObject=True)
ids = surface_outer_target_final[0][0][1]
idxss = [TagDiMES0] + [ids] + [v_[1]
                               for v_ in v if (v_[0] == 2 and v_[1] != ids)] + [pl]
surface_loop = gmsh.model.occ.add_surface_loop(idxss)
vol = gmsh.model.occ.add_volume([surface_loop])
gmsh.model.occ.remove([(3, 1)])
gmsh.model.occ.remove(gmsh.model.get_entities(2))
gmsh.model.occ.remove(gmsh.model.get_entities(1))
gmsh.model.occ.synchronize()
mesh = gmsh.model.mesh.generate(3)
gmsh.fltk.run()
gmsh.write("test_DiMES_000.stl")
gmsh.write("test_DiMES_000.step")
gmsh.write("test_DiMES_000.msh")
gmsh.write("test_DiMES_000.brep")

# Initialize gmsh session
gmsh.initialize()

# Define DiMES cap surface
xDiMES = 0.0
yDiMES = 0.0
zDiMES = 0.0
rDiMES = 0.025

# Define small dot on DiMES cap surface
xsDot = -0.01
ysDot = 0.0
zsDot = 0.0
rsDot = 0.0005

# Define large dot on DiMES cap surface
xlDot = 0.0
ylDot = 0.0
zlDot = 0.0
rlDot = 0.005

# Tags
TagDiMES0 = 0
TagDiMES = 10
TaglDot = 20
TagsDot = 30

# Create 2D surfaces
gmsh.model.occ.addDisk(xDiMES, yDiMES, zDiMES, rDiMES, rDiMES, TagDiMES0)
gmsh.model.occ.addDisk(xsDot, ysDot, zsDot, rsDot, rsDot, TagsDot)
gmsh.model.occ.addDisk(xlDot, ylDot, zlDot, rlDot, rlDot, TaglDot)
gmsh.model.occ.cut([(2, TagDiMES0)], [(2, TagsDot),
                   (2, TaglDot)], TagDiMES, removeTool=False)

# Synchronize necessary before mesh setup and generation
gmsh.model.occ.synchronize()

# Set number of elements on the boundary of each dots and DiMES cap
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 50)

# Prevent very small elements in small dots
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.001)
# Generate 2D mesh
mesh = gmsh.model.mesh.generate(3)

# Launch the GUI to see the results:
gmsh.fltk.run()

# Write mesh into a meshio format
gmsh.write("small_large_dots_DiMES.msh")

# Close gmsh session
gmsh.finalize()

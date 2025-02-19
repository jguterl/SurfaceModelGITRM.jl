#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:58:57 2023

@author: jeromeguterl
"""
import numpy as np

from distribution_generators import Distribs


class ParticleDistribution():

    def __init__(self, np, M):
        self.np = np
        self.mass = M

    def ShowAvailablePdfs(self):
        Distribs.ShowAvailablePdfs()

    def RotateAngle(self, Field: str, theta: float, phi: float, Degree=True):
        self.Rotate(Field, [0, 1, 0], theta, Degree)
        self.Rotate(Field, [0, 0, 1], phi, Degree)

    def Rotate(self, Field: str, AxisVector: np.ndarray or list, Angle: float, Degree=True) -> None:
        """
        Rotate the vector field (x,y,z) or (vx,vy,vz) around the axis AxisVector by angle Angle

        Args:
            Field (str): DESCRIPTION.
            AxisVector (np.ndarray or list): DESCRIPTION.
            Angle (float): DESCRIPTION.
            Degree (TYPE, optional): if True, angle are in degree. Defaults to True.

        Returns:
            None: DESCRIPTION.

        """
        assert Field == 'v' or Field == 'x', 'Field must be either "v" or "x"'
        assert type(
            AxisVector) == np.ndarray or list, 'AxisVector must be a list or numpy array'

        if Field == 'x':
            Fields = ['x', 'y', 'z']
        else:
            Fields = ['vx', 'vy', 'vz']

        Vs = [self.Particles[k] for k in Fields]
        V = RotateCoordinates(*Vs, AxisVector, Angle, Degree)
        for i, F in enumerate(Fields):
            self.Particles[F] = V[:, i].squeeze()

    def generate(self, DistribName, **kwargs):
        """


        Args:
            Np (int, optional): Number fo samples. Defaults to 10000.
            DistribName (TYPE): Probability distribution function.

        Returns:
            Distribution (np.ndarray): distribution

        """

        x, pdf = Distribs.GetPdf(DistribName, **kwargs)

        return Distribs.GenerateDistribution(x, pdf, self.np)

    def Generate_Weighted(self, Np, q_DistribName, p_DistribName, **kwargs):
        """
        Args:
            Np (int, optional): Number fo samples. Defaults to 10000.
            DistribName (TYPE): Probability distribution function.

        Returns:
            Distribution (np.ndarray): distribution

        """

        x, pdf_q, pdf_p = Distribs.GetWeightedPdf(
            q_DistribName, p_DistribName, **kwargs)

        pdf_p_times_q = np.multiply(pdf_p, pdf_q)

        Normalization = Integrale(pdf_p_times_q, x, Array=False)
        # Normalization = 1.0

        pdf_p_times_q = np.divide(pdf_p_times_q, Normalization)

        sampled_x, weights, sampled_x_physics = Distribs.GenerateDistributionWeighted(
            x, pdf_p_times_q, pdf_p, pdf_q, Np)

        # weights = np.divide(1.0,pdf_q[np.searchsorted(x,sampled_x)])

        return sampled_x, weights, sampled_x_physics

    def compute_velocity(p, E, theta, phi):
        beta = 0.5*p.mass*1.66e-27/1.602e-19
        p.phi = phi
        p.theta = theta
        p.E = E
        p.v = np.sqrt(E/beta)
        p.vx = p.v * np.sin(theta) * np.cos(phi)
        p.vy = p.v * np.sin(theta) * np.sin(phi)
        p.vz = p.v * np.cos(theta)

    def generate_positions(p, el):
        sqrt_a = np.sqrt(p.generate('Uniform', x=np.linspace(0, 1, p.np)))
        b = p.generate('Uniform', x=np.linspace(0, 1, p.np))
        pos = np.vstack([(1-sqrt_a) * el.u1[i] + sqrt_a*(1-b)
                        * el.u2[i] + b*sqrt_a*el.u3[i] for i in range(3)]).T
        p.x = pos[:, 0] + el.centroid[0]
        p.y = pos[:, 1] + el.centroid[1]
        p.z = pos[:, 2] + el.centroid[2]

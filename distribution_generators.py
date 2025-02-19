#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:04:50 2023

@author: jeromeguterl
"""

import numpy as np

import numpy as np
from math_helper import *
from typing import Callable
import matplotlib.pyplot as plt
import pydoc
import netCDF4
import os


class Distribs():

    # def Gaussian(x: np.ndarray = np.linspace(-10, 10, 10000), sigma: float = 1.0, mu: float = 0.0, beta: float = 0.0, Normalized=True):
    #    f = np.abs(x)**beta*np.exp(-1.0/2.0*((x-mu)/sigma)**2)
    #    #if beta > 0:
    #    #    f[np.argwhere(x<0)] = 0
    #    if Normalized:
    #        f = f/Integrale(f, x, Array=False)
    #    return f

    def Gaussian(x: np.ndarray = np.linspace(-15000, 15000, 100000), sigma: float = 30.0, mu: float = 70.0, Normalized=True):
        f = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((x-mu)/sigma)**2)
        # f[np.argwhere(x<0)] = 0
        if Normalized:
            f = f/Integrale(f, x, Array=False)
        return f

    def Gaussian_test(x: np.ndarray = np.linspace(-15000, 15000, 100000), sigma: float = 15.0, mu: float = 30.0, Normalized=True):
        f = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((x-mu)/sigma)**2)
        # f[np.argwhere(x<0)] = 0
        if Normalized:
            f = f/Integrale(f, x, Array=False)
        return f

    def Gaussian_Jerome(x: np.ndarray = np.linspace(0, 15000, 100000), sigma: float = 20.0, mu: float = 20, beta: float = 0.0, Normalized=True):
        f = np.abs(x)**beta*np.exp(-1.0/2.0*((x-mu)/sigma)**2)
        if beta > 0:
            f[np.argwhere(x < 0)] = 0
        if Normalized:
            f = f/Integrale(f, x, Array=False)
        return f

    def Gaussian_Zack(x: np.ndarray = np.linspace(-10, 10, 10000), sigma: float = 1.0, mu: float = 0.0, beta: float = 0.0, Normalized=True):
        f = np.abs(x)**beta*np.exp(-1.0/2.0*((x-mu)/sigma)**2)
        if beta > 0:
            f[np.argwhere(x < 0)] = 0
        if Normalized:
            f = f/Integrale(f, x, Array=False)
        return f

    # def Maxwellian(x: np.ndarray, sigma: float = 1, mu: float = 0, beta: float = 0, Normalized=True):
    #     f = np.abs(x)**beta*np.exp(-1.0/2.0*((x-mu)/sigma)**2)
    #     if Normalized:
    #         f = f/Integrale(f, x, Array=False)
    #     return f

    def Thomson(x: np.ndarray = np.linspace(0, 300, 10000), xb: float = 6.67, xc: float = 200, Normalized=True):
        # xb = 6.67 for C, xb = 8.64 for W, xb = 4.73 for Si

        assert not (xc <= xb), "xc cannot be <= xb"
        f = x/(x + xb) ** 3*(1.0-np.sqrt((x+xb)/(xc+xb)))
        # f[np.argwhere(x > xc)] = 0.0
        if Normalized:
            f = f/Integrale(f, x, Array=False)
        return f

    def Uniform(x=np.linspace(0, 1, 10), xmin=None, xmax=None, Normalized=True):
        assert not (xmin is not None and xmax is not None and xmin >=
                    xmax), "xmin cannot be <= xmax"
        f = np.full(x.shape, 1)
        if xmin is not None:
            f[np.argwhere(x < xmin)] = 0
        if xmax is not None:
            f[np.argwhere(x > xmax)] = 0

        if Normalized:
            f = f/Integrale(f, x, Array=False)

        return f

    def SinCos(x=np.linspace(0, np.pi/2, 10000), xmin=None, xmax=None, Normalized=True):
        assert not (xmin is not None and xmax is not None and xmin >=
                    xmax), "xmin cannot be <= xmax"

        f = np.sin(x)*np.cos(x)
        if xmin is not None:
            f[np.argwhere(x < xmin)] = 0
        if xmax is not None:
            f[np.argwhere(x > xmax)] = 0
        if Normalized:
            f = f/Integrale(f, x, Array=False)
        return f

    def Levy(x=np.linspace(0.1, 10, 10000), c=1, mu=0):
        return np.sqrt(c/2/np.pi)*np.exp(-c/(x-mu))/((x-mu)**1.5)

    @classmethod
    def GetPdf(cls, f: str or Callable[np.array, np.ndarray], **kwargs) -> (np.ndarray, np.ndarray):

        if type(f) == str:
            if hasattr(cls, f):
                f = getattr(cls, f)
                if kwargs.get('x') is None:
                    x = f.__defaults__[0]
                else:
                    x = kwargs.pop('x')

            else:
                raise KeywordError(
                    'Cannot find the function "{}" in attributes.'.format(f))
        else:
            if kwargs.get('x') is None:
                raise KeywordError(
                    'Must provide a vector of x values for the function f to get f(x)')
            else:
                x = kwargs.pop('x')

        return x, f(x, **kwargs)

    @classmethod
    def GetWeightedPdf(cls, f_q: str or Callable[np.array, np.ndarray], f_p: str or Callable[np.array, np.ndarray], **kwargs) -> (np.ndarray, np.ndarray, np.ndarray):

        if type(f_q) == str:
            if hasattr(cls, f_q):
                f_q = getattr(cls, f_q)
                if kwargs.get('x') is None:
                    x = np.linspace(0, 5000, 100000)  # specifically for vz
                else:
                    x = kwargs.pop('x')
            else:
                raise KeywordError(
                    'Cannot find the function "{}" in attributes.'.format(f_q))
        else:
            if kwargs.get('x') is None:
                raise KeywordError(
                    'Must provide a vector of x values for the function f to get f(x)')
            else:
                x = kwargs.pop('x')

        if type(f_p) == str:
            if hasattr(cls, f_p):
                f_p = getattr(cls, f_p)

            else:
                raise KeywordError(
                    'Cannot find the function "{}" in attributes.'.format(f_p))

        # print(len(x))
        # print(f_p(x,**kwargs))
        # print(f_q(x,sigma,mu,**kwargs))

        return x, f_q(x, **kwargs), f_p(x, **kwargs)

    @classmethod
    def PlotDistrib(cls, DistribName, **kwargs):
        """


        Args:
            cls (TYPE): DESCRIPTION.
            DistribName (TYPE): DESCRIPTION.
            **kwargs (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        if hasattr(cls, DistribName):
            f = getattr(cls, DistribName)
            if kwargs.get('x') is None:
                kwargs['x'] = f.__defaults__[0]
                Name = ';' + DistribName+(';').join(['{}={}'.format(v, d) for v, d in zip(
                    f.__code__.co_varnames[1:-2], f.__defaults__[1:])])

            plt.plot(kwargs['x'], f(**kwargs), label=Name)
            plt.legend()
        else:
            print('Cannot find the distribution function "{}" in BaseDistrib.'.format(
                DistribName))

    @classmethod
    def ShowAvailablePdfs(cls):
        for D in ['Gaussian', 'Uniform', 'SinCos', 'Thomson']:
            print('Distribution: {}'.format(
                pydoc.render_doc(getattr(cls, D), "%s")))

    def GenerateDistribution(x: np.ndarray, pdf: np.ndarray, N: int = 10000):
        assert type(
            N) == int and N > 0, "Nsamples must be an integer > 0. N={}".format(N)
        assert type(x) == np.ndarray and type(
            pdf) == np.ndarray, " x and pdf must be a numpy array"
        assert x.shape == pdf.shape, " x and pdf must have the same shape.:x.shape={}; pdf.shape={}".format(
            x.shape, pdf, shape)

        # print(len(x[np.searchsorted(Integrale(pdf, x, Normalized=True), np.random.rand(N), side='left')]))
        return x[np.searchsorted(Integrale(pdf, x, Normalized=True), np.random.rand(N), side='left')]

    def GenerateDistributionWeighted(x: np.ndarray, pdf: np.ndarray, pdf_p: np.ndarray, pdf_q: np.ndarray, N: int = 10000):
        assert type(
            N) == int and N > 0, "Nsamples must be an integer > 0. N={}".format(N)
        assert type(x) == np.ndarray and type(
            pdf) == np.ndarray, " x and pdf must be a numpy array"
        assert x.shape == pdf.shape, " x and pdf must have the same shape.:x.shape={}; pdf.shape={}".format(
            x.shape, pdf, shape)

        rng = np.random.default_rng()

        xi = rng.random(N)

        sampled_x_physics = x[np.searchsorted(
            Integrale(pdf_p, x, Normalized=True), xi, side='left')]
        sampled_x = x[np.searchsorted(
            Integrale(pdf, x, Normalized=True), xi, side='left')]

        weights = 1 / \
            pdf_q[np.searchsorted(
                Integrale(pdf, x, Normalized=True), xi, side='left')]

        # plt.figure()
        # plt.hist(sampled_x_physics,density=True,bins=100,alpha=0.5,label="physics")
        # plt.hist(sampled_x ,density=True,bins=100, weights=weights,label="weighted samples")
        # plt.legend()
        # plt.show()

        return sampled_x, weights, sampled_x_physics

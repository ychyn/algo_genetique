#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Modules of python
# -----------------

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import seaborn as sns
import random

# Modules with own classes
# ------------------------

from glassdata import GlassData
from network import NeuralNetwork
from glassproperties import VFTcoefficients

# ---------------------------------------
# Data-set on rho and ANN on molar volume
# ---------------------------------------


def check_convex(df):
    # -------------------------
    # Data-set of glass density
    # -------------------------

    filedbrho='DataBase/rho20oxides.csv'
    dbrho=GlassData(filedbrho)
    dbrho.info()
    dbrho.bounds()

    # Determination of the molar volume
    dbrho.oxidemolarmass()
    dbrho.molarmass()
    dbrho.y=dbrho.MolarM/dbrho.y
    dbrho.normalize_y()

    # Loading of the ANN model
    arch=[20,20,20]
    nnmolvol=NeuralNetwork(dbrho.noxide,arch,'gelu','linear')
    nnmolvol.compile(3.e-4)
    nnmolvol.ArchName(arch)
    nnmolvol.load('Models/nnmolarvol'+nnmolvol.namearch+'.h5')
    nnmolvol.info()

    # ------------------------------------------------
    # Data-set on Young's modulus and ANN on Vt=E/(2G)
    # ------------------------------------------------

    filedbE='DataBase/E20oxides.csv'
    dbE=GlassData(filedbE)
    dbE.info()
    dbE.bounds()

    # ------------------------------
    # Loading of dissociation energy
    # ------------------------------

    datadisso=pd.read_csv('dissociationenergy.csv')
    G=np.zeros(dbE.nsample)
    for i in range(dbE.nsample):
        G[i]=np.sum(datadisso['G'].values*dbE.x[i,:])
    #end for

    # Determination of E/G and normalization
    dbE.y=dbE.y/(2.*G)
    dbE.normalize_y()

    # ------------------------------
    # Loading of the ANN model on Vt
    # ------------------------------

    arch=[20,20,20]
    nnmodelEsG=NeuralNetwork(dbE.noxide,arch,'gelu','linear')
    nnmodelEsG.compile(1.e-4)
    nnmodelEsG.ArchName(arch)
    nnmodelEsG.load('Models/nnEsG'+nnmodelEsG.namearch+'.h5')
    nnmodelEsG.info()

    # ---------------------------------------
    # Data-set on Tannealing=Tg and ANN model
    # ---------------------------------------

    filedbTannealing='DataBase/Tannealing20oxides.csv'
    dbTannealing=GlassData(filedbTannealing)
    dbTannealing.info()
    dbTannealing.bounds()
    dbTannealing.normalize_y()

    # ANN model on Tannealing
    # -----------------------

    arch=[20,20,20]
    nnTannealing=NeuralNetwork(dbTannealing.noxide,arch,'gelu','linear')
    nnTannealing.compile(3.e-4)
    nnTannealing.ArchName(arch)
    nnTannealing.load('Models/nn'+dbTannealing.nameproperty+nnTannealing.namearch+'.h5')
    # nnTannealing.info()

    # -------------------------------
    # Data-set on Tmelt and ANN Model
    # -------------------------------
    # ! This data-set does not include V2O5. Only 19 oxides are involved.

    filedbTmelt='DataBase/Tmelt19oxides.csv'
    dbTmelt=GlassData(filedbTmelt)
    # dbTmelt.info()
    dbTmelt.bounds()
    dbTmelt.normalize_y()

    # ANN model on Tmelt
    # ------------------

    arch=[20,20,20]
    nnTmelt=NeuralNetwork(dbTmelt.noxide,arch,'gelu','linear')
    nnTmelt.compile(3.e-4)
    nnTmelt.ArchName(arch)
    nnTmelt.load('Models/nn'+dbTmelt.nameproperty+nnTmelt.namearch+'.h5')
    # nnTmelt.info()

    # -------------------------------
    # Data-set on Tsoft and ANN model
    # -------------------------------

    # Data-set on Tsoft
    # -----------------

    filedbTsoft='DataBase/Tsoft20oxides.csv'
    dbTsoft=GlassData(filedbTsoft)
    dbTsoft.info()
    dbTsoft.bounds()
    dbTsoft.normalize_y()

    # ANN model on Tsoft
    # ------------------

    arch=[20,20,20]
    nnTsoft=NeuralNetwork(dbTsoft.noxide,arch,'gelu','linear')
    nnTsoft.compile(3.e-4)
    nnTsoft.ArchName(arch)
    modelfile='Models/nn'+dbTsoft.nameproperty+nnTsoft.namearch+'.h5'
    nnTsoft.load(modelfile)
    nnTsoft.info()

    # Graphical representation
    # ------------------------

    # # Loading of compositions
    dcompo=df.iloc[:, :20]
    xcompo=dcompo.values[:,:]
    Ncompo=np.size(xcompo,0)
    rhocompo=dbrho.GlassDensity(nnmolvol,dbrho.oxide,xcompo)
    Ecompo=dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,xcompo)
    Tgcompo=dbTannealing.physicaly(nnTannealing.model.predict(xcompo).transpose()[0,:])
    Tmeltcompo=dbTmelt.physicaly(nnTmelt.model.predict(xcompo[:,:-1]).transpose()[0,:])
    Tsoftcompo=dbTsoft.physicaly(nnTsoft.model.predict(xcompo).transpose()[0,:])

    # Determination of the viscosity500
    Acompo,Bcompo,T0compo=VFTcoefficients(Tmeltcompo,Tsoftcompo,Tgcompo)

    is_sorted = lambda a: np.all(a[:-1] <= a[1:])

    NT=50
    eta = np.zeros((Ncompo, NT))
    T = np.zeros((Ncompo, NT))
    slope = np.zeros((Ncompo, NT - 1))
    checker = np.zeros(Ncompo, dtype=bool)
    for i in range(Ncompo):
        Tmin=1.1*Tgcompo[i]
        Tmax=Tmeltcompo[i]
        T[i]=np.linspace(Tmin,Tmax,NT)
        eta[i]=10**(Acompo[i]+Bcompo[i]/(T[i]-T0compo[i]))
        slope[i] = (np.log(eta[i, 1:]) - np.log(eta[i, :-1])) / (T[i, 1:] - T[i, :-1])
        checker[i] = is_sorted(slope[i])
    return checker


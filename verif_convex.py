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


class ConvexModel():

    def __init__(self):
        # ---------------------------------------
        # Data-set on Tannealing=Tg and ANN model
        # ---------------------------------------

        filedbTannealing='DataBase/Tannealing20oxides.csv'
        self.dbTannealing=GlassData(filedbTannealing)
        self.dbTannealing.info()
        self.dbTannealing.bounds()
        self.dbTannealing.normalize_y()

        # ANN model on Tannealing
        # -----------------------

        arch=[20,20,20]
        self.nnTannealing=NeuralNetwork(self.dbTannealing.noxide,arch,'gelu','linear')
        self.nnTannealing.compile(3.e-4)
        self.nnTannealing.ArchName(arch)
        self.nnTannealing.load('Models/nn'+self.dbTannealing.nameproperty+self.nnTannealing.namearch+'.h5')
        # nnTannealing.info()

        # -------------------------------
        # Data-set on Tmelt and ANN Model
        # -------------------------------
        # ! This data-set does not include V2O5. Only 19 oxides are involved.

        filedbTmelt='DataBase/Tmelt19oxides.csv'
        self.dbTmelt=GlassData(filedbTmelt)
        self.dbTmelt.info()
        self.dbTmelt.bounds()
        self.dbTmelt.normalize_y()

        # ANN model on Tmelt
        # ------------------

        arch=[20,20,20]
        self.nnTmelt=NeuralNetwork(self.dbTmelt.noxide,arch,'gelu','linear')
        self.nnTmelt.compile(3.e-4)
        self.nnTmelt.ArchName(arch)
        self.nnTmelt.load('Models/nn'+ self.dbTmelt.nameproperty + self.nnTmelt.namearch+'.h5')
        # nnTmelt.info()

        # -------------------------------
        # Data-set on Tsoft and ANN model
        # -------------------------------

        # Data-set on Tsoft
        # -----------------

        filedbTsoft='DataBase/Tsoft20oxides.csv'
        self.dbTsoft=GlassData(filedbTsoft)
        self.dbTsoft.info()
        self.dbTsoft.bounds()
        self.dbTsoft.normalize_y()

        # ANN model on Tsoft
        # ------------------

        arch=[20,20,20]
        self.nnTsoft=NeuralNetwork(self.dbTsoft.noxide,arch,'gelu','linear')
        self.nnTsoft.compile(3.e-4)
        self.nnTsoft.ArchName(arch)
        modelfile='Models/nn'+ self.dbTsoft.nameproperty + self.nnTsoft.namearch+'.h5'
        self.nnTsoft.load(modelfile)
        self.nnTsoft.info()

    def convex_coefficient(self, generation):
        # # Loading of compositions
        xcompo=generation[:, :20]
        Tgcompo=self.dbTannealing.physicaly(self.nnTannealing.model.predict(xcompo).transpose()[0,:])
        Tmeltcompo=self.dbTmelt.physicaly(self.nnTmelt.model.predict(xcompo[:,:-1]).transpose()[0,:])
        Tsoftcompo=self.dbTsoft.physicaly(self.nnTsoft.model.predict(xcompo).transpose()[0,:])

        # Determination of the viscosity500
        Acompo,Bcompo,T0compo=VFTcoefficients(Tmeltcompo,Tsoftcompo,Tgcompo)

        return Bcompo * np.sign((Tgcompo * 1.1 - T0compo))


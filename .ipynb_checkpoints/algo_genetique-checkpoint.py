#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is our oroject mainfile
"""

# Modules of python
# -----------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Modules with own classes
# ------------------------

from glassdata import GlassData
from network import NeuralNetwork

# ---------------------------------------
# Data-set on rho and ANN on molar volume
# ---------------------------------------

# Dataset of rho
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

# Data-set of Tannealing
filedbTannealing='DataBase/Tannealing20oxides.csv'
dbTannealing=GlassData(filedbTannealing)
dbTannealing.bounds()
dbTannealing.normalize_y()

# ANN model on Tannealing
# -----------------------
arch=[20,20,20]
nnTannealing=NeuralNetwork(dbTannealing.noxide,arch,'gelu','linear')
nnTannealing.compile(3.e-4)
nnTannealing.ArchName(arch)
nnTannealing.load('Models/nn'+dbTannealing.nameproperty+nnTannealing.namearch+'.h5')
nnTannealing.info()

# Data-set on Tmelt
# -----------------

filedbTmelt='DataBase/Tmelt19oxides.csv'
dbTmelt=GlassData(filedbTmelt)
dbTmelt.info()
dbTmelt.bounds()
dbTmelt.normalize_y()

# ANN model on Tmelt
# ------------------
arch=[20,20,20]
nnTmelt=NeuralNetwork(dbTmelt.noxide,arch,'gelu','linear')
nnTmelt.compile(3.e-4)
nnTmelt.ArchName(arch)
nnTmelt.load('Models/nn'+dbTmelt.nameproperty+nnTmelt.namearch+'.h5')
nnTmelt.info()

# ------------------------------
# Data-set on Tliq and ANN model
# ------------------------------

filedbTliq='DataBase\Tsoft20oxides.csv'
dbTliq=GlassData(filedbTliq)
dbTliq.info()
dbTliq.bounds()
dbTliq.normalize_y()

# ANN model on Tliq
# -----------------
arch=[32,32,32,32]
nnTliq=NeuralNetwork(dbTliq.noxide,arch,'gelu','linear')
nnTliq.compile(3.e-4)
nnTliq.ArchName(arch)
#modelfile='Models\nn'+dbTliq.nameproperty+nnTliq.namearch+'.h5'
modelfile='Models/nnTsoft3c20.h5'
nnTliq.load(modelfile)
nnTliq.info()

# ------------------------------------------
# Determination of the bounds for each oxide
# ------------------------------------------

# # Algo genetique

# ## Variables utiles

# +
labels = dbrho.oxide
available_mat = ['SiO2', 'Al2O3', 'MgO', 'CaO', 'Na2O', 'K2O','ZnO', 'TiO2']

#Contraintes
si_min = 0.35
si_max = 0.75

xmaxt=np.array([dbrho.xmax,dbE.xmax,dbTannealing.xmax,np.append(dbTmelt.xmax,1.),dbTliq.xmax])
xmax=np.zeros(dbrho.noxide)
for i in range(dbrho.noxide):
    if dbrho.oxide[i] in available_mat:
        xmax[i]=np.min(xmaxt[:,i])
xmax[0] = 0

prop_label = ['rho','E','Tg','Tmelt']
weight=[0,0.5,0,0.5]
minimize=[True,True,False,False]
# -

# ## Creation de generations

N_population = 1000
def init_pop(N_population):
    xglass,Mmolar=dbrho.randomcomposition(N_population,xmax)
    #x_si est la proportion malaire de SiO2, qu'on impose entre deux valeurs
    x_si = np.random.uniform(si_min,si_max,N_population)
    for i in range(20):
        xglass[:,i] = xglass[:,i] * (1 - x_si)
    xglass[:,0] = x_si
    return xglass


def prop_calculation(population):
    rho=dbrho.GlassDensity(nnmolvol,dbrho.oxide,population)
    E=dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,population)
    Tg=dbTannealing.physicaly(nnTannealing.model.predict(population).transpose()[0,:])
    Tmelt=dbTmelt.physicaly(nnTmelt.model.predict(population[:,:-1]).transpose()[0,:])
    return np.vstack((rho,E,Tg,Tmelt)).transpose()


def normalize(prop):
    return (prop - prop.min(axis=0))/(prop.max(axis=0)-prop.min(axis=0))


#prop est une array avec les proprietes du verre normalisées, weight est le poids qu'on accorde
#à chacune des proprietes, et minize est une liste de booléens selon qu'on veuille minimiser
#ou maximiser une certaine variable
def fitness_func(prop_normalized,weight,minimize):
    rating = np.zeros(prop_normalized.shape[0])
    for i in range(len(weight)):
        if minimize[i]:
            rating += (1-prop_normalized[:,i])*weight[i]
        else:
            rating += prop_normalized[:,i]*weight[i]
    return rating


def sort_by_f(population,F):
    population_info = np.column_stack((population,F))
    sorted_arr = population_info[population_info[:, -1].argsort()][::-1]
    return sorted_arr


population = init_pop(N_population)

prop = prop_calculation(population)

normalized_prop = normalize(prop)

F = fitness_func(normalized_prop,weight,minimize)

sorted_arr = sort_by_f(population, F)

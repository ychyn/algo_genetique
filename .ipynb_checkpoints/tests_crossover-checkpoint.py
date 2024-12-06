#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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

# +
# dbrho?
# -



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

# -------------------------------
# Data-set on Tmelt and ANN Model
# -------------------------------
# ! This data-set does not include V2O5. Only 19 oxides are involved.

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

xmaxt=np.array([dbrho.xmax,dbE.xmax,dbTannealing.xmax,np.append(dbTmelt.xmax,1.),dbTliq.xmax])
xmax=np.zeros(dbrho.noxide)
x_on_a = ['SiO2', 'Al2O3', 'MgO', 'CaO', 'Na2O', 'K2O','ZnO', 'TiO2']
for i in range(dbrho.noxide):
    if dbrho.oxide[i] in x_on_a:
        xmax[i]=np.min(xmaxt[:,i])
    # xmax[i]=np.min(xmaxt[:,i])
#endif
xmax[0] = 0


# -----------------------------------------------------
# Generation of random Nglass compositions without V2O3
# -----------------------------------------------------

Nglass=10000
xglass,Mmolar=dbrho.randomcomposition(Nglass,xmax)
xglass

xglassn = np.zeros((Nglass,20))
lamb = np.random.uniform(0.35,0.75,Nglass)
print(lamb)
for i in range(20):
    xglassn[:,i] = xglass[:,i] * (1 - lamb)
xglassn[:,0] = lamb

xglass,xglassn = xglassn,xglass

# ---------------------------------
# Computation of various properties
# ---------------------------------

# Computation of rho from the ANN model on molar volume
# -----------------------------------------------------

rho=dbrho.GlassDensity(nnmolvol,dbrho.oxide,xglass)

# Computation of E from the ANN model on Vt
# -----------------------------------------

E=dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,xglass)

# Computation of Tg from the ANN model on Tannealing
# --------------------------------------------------

Tg=dbTannealing.physicaly(nnTannealing.model.predict(xglass).transpose()[0,:])

# Computation of Tmelt from the ANN model on Tmelt
# ------------------------------------------------
# #! The last molar fraction is removed since V2O3 is not involved.
Tmelt=dbTmelt.physicaly(nnTmelt.model.predict(xglass[:,:-1]).transpose()[0,:])

# Computation of Tliq from the ANN model on Tliq
# ----------------------------------------------

Tliq=dbTliq.physicaly(nnTliq.model.predict(xglass).transpose()[0,:])

# Research of composition
Tmmin=1000+273.15
Tmmax=1300.+273.15
Tgmin=500.+273.15
Tgmax=600.+273.15
rhomin=2.4e3
rhomax=2.9e3
Emin=70
#Emax=140
xcompo=np.zeros(dbE.noxide)
proglass=np.zeros(4)
for i in range(len(xglass)):
    #if (E[i]>Emin and E[i]<Emax and Tg[i]>Tgmin and Tg[i]<Tgmax and 
    #    rho[i]>rhomin and rho[i]<rhomax and Tmelt[i]>Tmmin and Tmelt[i]<Tmmax):
    if (Tg[i]>Tgmin and Tg[i]<Tgmax and 
        rho[i]>rhomin and rho[i]<rhomax and
        Tmelt[i]>Tmmin and Tmelt[i]<Tmmax and
        E[i]>Emin):
        xcompo=np.vstack((xcompo,xglass[i,:]))
        proglass=np.vstack((proglass,np.hstack(([rho[i],E[i],Tg[i]-273.15,Tmelt[i]-273.15]))))
        '''if i<1000:
            print(xcompo, 'xcompo',i)
            print(proglass,'proglass',i)'''
    #end if
#end if
xcompo=xcompo[1:,:]
proglass=proglass[1:,:]

XY=np.zeros((np.size(xcompo,0),dbE.noxide+4))
XY[:,0:dbE.noxide]=xcompo
XY[:,dbE.noxide:dbE.noxide+4]=proglass
columns=dbE.oxide
columns=np.hstack((columns,['rho','E','Tg','Tm']))
print(XY)
datacompo=pd.DataFrame(XY,columns=columns,dtype='float')
datacompo.to_csv('compoverrelowTm_Ivan.csv')

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


population = init_pop(N_population)

population

prop = prop_calculation(population)

normalized_prop = normalize(prop)

F = fitness_func(normalized_prop,weight,minimize)

population.shape

F.shape

pop_info = np.column_stack((population,F))

pop_info.shape

from random import randint

N_parents = int(0.1 * N_population)
N_enfants = int(0.4 * N_population)

from crossover import crossover

N_nonMutants = int(0.9 * N_population)
epsilon = 0.05

from crossover import mutation

mutants = mutation(population)

np.min(mutants)

enfants = crossover(pop_info)

enfants

enfants.shape







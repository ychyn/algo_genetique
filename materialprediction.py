#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:41:41 2024

Example program to determine:
    
    -rho: density (kg/m3)
    -E: Young's modulus (GPa)
    -Tmelt: Melting temperature corresponding to log(eta)=1 (K)
    -Tg: Glass tansition temperature defined by log(eta)=12 (K)
    -Tliq: Liquidus temperature (K)

@author: fpigeonneau
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
xmax

dbrho.oxide

# -----------------------------------------------------
# Generation of random Nglass compositions without V2O3
# -----------------------------------------------------

Nglass=10000
# ['SiO2', 'B2O3', 'Al2O3', 'MgO', 'CaO', 'BaO', 'Li2O', 'Na2O', 'K2O', 'ZnO', 'SrO', 'PbO', 'Fe2O3', 'Y2O3', 'La2O3', 'TiO2', 'GeO2', 'ZrO2', 'P2O5', 'V2O5']
xmin = np.zeros(20)
xmin[0] = 0.3
xmin[7] = 0.1
xglass,Mmolar=dbrho.better_random_composition(Nglass,xmin,xmax)
xglass

xglassn = np.zeros((Nglass,20))
lamb = np.random.uniform(0.35,0.75,Nglass)
print(lamb)
for i in range(20):
    xglassn[:,i] = xglass[:,i] * (1 - lamb)
xglassn[:,0] = lamb

xglass,xglassn = xglassn,xglass

xglass.sum(axis=1)

'''xglass[:,0] = xglass[:,0]+0.5
xglass = xglass /1.5
xglass'''



# +
# dbrho.randomcomposition?
# -

xglass.sum(axis=1)

# ---------------------------------
# Computation of various properties
# ---------------------------------

# Computation of rho from the ANN model on molar volume
# -----------------------------------------------------

dbrho.GlassDensity(nnmolvol,dbrho.oxide,np.array([xglass[0]]))

rho=dbrho.GlassDensity(nnmolvol,dbrho.oxide,xglass)

rho

# Computation of E from the ANN model on Vt
# -----------------------------------------

E=dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,xglass)

E

# Computation of Tg from the ANN model on Tannealing
# --------------------------------------------------

Tg=dbTannealing.physicaly(nnTannealing.model.predict(xglass).transpose()[0,:])

# Computation of Tmelt from the ANN model on Tmelt
# ------------------------------------------------
# #! The last molar fraction is removed since V2O3 is not involved.
Tmelt=dbTmelt.physicaly(nnTmelt.model.predict(xglass[:,:-1]).transpose()[0,:])

Tmelt

# Computation of Tliq from the ANN model on Tliq
# ----------------------------------------------

Tliq=dbTliq.physicaly(nnTliq.model.predict(xglass).transpose()[0,:])

Tliq

# Graphical representation
# ------------------------
Plot=True

'''if (Plot):
    fig1,ax1=plt.subplots()
    data1=pd.DataFrame(np.transpose(np.array([rho,E])),columns=['rho','E'])
    sns.kdeplot(data1,x='rho',y='E',color='k',fill=True,ax=ax1,alpha=0.5)
    
    fig2,ax2=plt.subplots()
    data2=pd.DataFrame(np.transpose(np.array([Tmelt,Tg])),columns=['Tmelt','Tg'])
    sns.kdeplot(data2,x='Tmelt',y='Tg',color='k',fill=True,ax=ax2,alpha=0.5)
#end if'''

# +
# dbE?
# -

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

'''if (Plot):
    data1=pd.DataFrame(np.transpose(np.array([proglass[:,0],proglass[:,1]])),columns=['rho','E'])
    sns.kdeplot(data1,x='rho',y='E',color='b',fill=True,ax=ax1,alpha=0.5)
        
    data2=pd.DataFrame(np.transpose(np.array([proglass[:,3],proglass[:,2]])),columns=['Tmelt','Tg'])
    sns.kdeplot(data2,x='Tmelt',y='Tg',color='b',fill=True,ax=ax2,alpha=0.5)

    plt.show()
#end if'''

XY=np.zeros((np.size(xcompo,0),dbE.noxide+4))
XY[:,0:dbE.noxide]=xcompo
XY[:,dbE.noxide:dbE.noxide+4]=proglass
columns=dbE.oxide
columns=np.hstack((columns,['rho','E','Tg','Tm']))
print(XY)
datacompo=pd.DataFrame(XY,columns=columns,dtype='float')
datacompo.to_csv('compoverrelowTm_Ivan.csv')

datacompo

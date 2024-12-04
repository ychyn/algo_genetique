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

# Get the home directory
# ----------------------

HOME=os.environ['HOMEPATH']
Plot=True
SAVEFIG=True

# -------------------------
# Data-set of glass density
# -------------------------

# filedbrho='DataBase/rho20oxides.csv'
# dbrho=GlassData(filedbrho)
# dbrho.info()
# dbrho.bounds()

# Determination of the molar volume
# dbrho.oxidemolarmass()
# dbrho.molarmass()
# dbrho.y=dbrho.MolarM/dbrho.y
# dbrho.normalize_y()

# Loading of the ANN model
# arch=[20,20,20]
# nnmolvol=NeuralNetwork(dbrho.noxide,arch,'gelu','linear')
# nnmolvol.compile(3.e-4)
# nnmolvol.ArchName(arch)
# nnmolvol.load('Models/nnmolarvol'+nnmolvol.namearch+'.h5')
# nnmolvol.info()

# ------------------------------------------------
# Data-set on Young's modulus and ANN on Vt=E/(2G)
# ------------------------------------------------

# filedbE='DataBase/E20oxides.csv'
# dbE=GlassData(filedbE)
# dbE.info()
# dbE.bounds()

# ------------------------------
# Loading of dissociation energy
# ------------------------------

# datadisso=pd.read_csv('dissociationenergy.csv')
# G=np.zeros(dbE.nsample)
# for i in range(dbE.nsample):
#     G[i]=np.sum(datadisso['G'].values*dbE.x[i,:])
# #end for

# Determination of E/G and normalization
# dbE.y=dbE.y/(2.*G)
# dbE.normalize_y()

# ------------------------------
# Loading of the ANN model on Vt
# ------------------------------

# arch=[20,20,20]
# nnmodelEsG=NeuralNetwork(dbE.noxide,arch,'gelu','linear')
# nnmodelEsG.compile(1.e-4)
# nnmodelEsG.ArchName(arch)
# nnmodelEsG.load('Models/nnEsG'+nnmodelEsG.namearch+'.h5')
# nnmodelEsG.info()

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
nnTannealing.info()

# -------------------------------
# Data-set on Tmelt and ANN Model
# -------------------------------
# ! This data-set does not include V2O5. Only 19 oxides are involved.

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

# ------------------------------------------
# Determination of the bounds for each oxide
# ------------------------------------------

# xmaxt=np.array([dbrho.xmax,dbE.xmax,dbTannealing.xmax,np.append(dbTmelt.xmax,1.)])
# xmax=np.zeros(dbrho.noxide)
# for i in range(dbrho.noxide):
#     xmax[i]=np.min(xmaxt[:,i])
# #endif

# -----------------------------------------------------
# Generation of random Nglass compositions without V2O3
# -----------------------------------------------------

# Nglass=1
# xglass,Mmolar=dbrho.randomcomposition(Nglass,xmax)

# ---------------------------------
# Computation of various properties
# ---------------------------------

# Computation of rho from the ANN model on molar volume
# -----------------------------------------------------

# rho=dbrho.GlassDensity(nnmolvol,dbrho.oxide,xglass)

# # Computation of E from the ANN model on Vt
# # -----------------------------------------

# E=dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,xglass)

# # Computation of Tg from the ANN model on Tannealing
# # --------------------------------------------------

# Tg=dbTannealing.physicaly(nnTannealing.model.predict(xglass).transpose()[0,:])

# # Computation of Tmelt from the ANN model on Tmelt
# # ------------------------------------------------
# # ! The last molar fraction is removed since V2O3 is not involved.
# Tmelt=dbTmelt.physicaly(nnTmelt.model.predict(xglass[:,:-1]).transpose()[0,:])

# Graphical representation
# ------------------------

# if (Plot):
#     fig1,ax1=plt.subplots()
#     data1=pd.DataFrame(np.transpose(np.array([rho,E])),columns=['rho','E'])
#     sns.kdeplot(data1,x='rho',y='E',color='k',fill=True,ax=ax1,alpha=0.5)
        
#     fig2,ax2=plt.subplots()
#     data2=pd.DataFrame(np.transpose(np.array([Tmelt-273.15,Tg-273.15])),columns=['Tmelt','Tg'])
#     sns.kdeplot(data2,x='Tmelt',y='Tg',color='k',fill=True,ax=ax2,alpha=0.5)
# #end if


# Loading of compositions
file='newglasscompo.csv'
dcompo=pd.read_csv(file,index_col=0)
xcompo=dcompo.values[:,:]
Ncompo=np.size(xcompo,0)
# rhocompo=dbrho.GlassDensity(nnmolvol,dbrho.oxide,xcompo)
# Ecompo=dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,xcompo)
Tgcompo=dbTannealing.physicaly(nnTannealing.model.predict(xcompo).transpose()[0,:])
Tmeltcompo=dbTmelt.physicaly(nnTmelt.model.predict(xcompo[:,:-1]).transpose()[0,:])
Tsoftcompo=dbTsoft.physicaly(nnTsoft.model.predict(xcompo).transpose()[0,:])
# print('rhocompo=',rhocompo)
# print('Ecompo=',Ecompo)
# print('Tgcompo=',Tgcompo)
# print('Tmeltcompo=',Tmeltcompo)
# print('Tsoftcompo=',Tsoftcompo)

# if (Plot):
#     ax1.plot(rhocompo[:-1],Ecompo[:-1],'ko')
#     ax1.plot(rhocompo[-1],Ecompo[-1],'bo')
#     ax2.plot(Tmeltcompo[:-1]-273.15,Tgcompo[:-1]-273.15,'ko')
#     ax2.plot(Tmeltcompo[-1]-273.15,Tgcompo[-1]-273.15,'bo')
#     ax1.set_xlabel(r'$\rho$ (kg/m$^3$)')
#     ax1.set_ylabel(r'$E$ (GPa)')
#     ax2.set_xlabel(r'$T_m$ (°C)')
#     ax2.set_ylabel(r'$T_g$ (°C)')
# #endif

# Determination of the viscosity500
Acompo,Bcompo,T0compo = VFTcoefficients(Tmeltcompo,Tsoftcompo,Tgcompo)

NT=50
plt.figure()
for i in range(Ncompo):
    Tmin=1.1*Tgcompo[i]
    Tmax=Tmeltcompo[i]
    T=np.linspace(Tmin,Tmax,NT)
    eta=10**(Acompo[i]+Bcompo[i]/(T-T0compo[i]))
    print(T, T0compo)
    plt.semilogy(T-273.15,eta)
    plt.annotate('Verre '+str(i + 1),(T[0]-273.15,eta[0]))
#end for
# Tmin=1.1*Tgcompo[-1]
# Tmax=Tmeltcompo[-1]
# T=np.linspace(Tmin,Tmax,NT)
# eta=10**(Acompo[-1]+Bcompo[-1]/(T-T0compo[-1]))
# plt.semilogy(T-273.15,eta,'k--')
#plt.annotate('Lion Glass',(T[0]-273.15,eta[0]))
plt.xlabel(r'$T$ (°C)')
plt.ylabel(r'$\eta$ (Pa.s)')

plt.show()

if (SAVEFIG):
    # fig1.savefig('EvsrhoHighE.png',dpi=300,bbox_inches='tight')
    # fig2.savefig('TgvsTmeltHighE.png',dpi=300,bbox_inches='tight')
    plt.savefig('etavsTHighE.png',dpi=300,bbox_inches='tight')
#end if


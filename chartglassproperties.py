#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:58:27 2024

Script used to determine charts of mechanical properties.

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

PATH='/home/fpigeonneau/ownCloud/Figures/MachineLearning/' # To be changed.
PLOT=True
SAVEFIG=False

# Dataset of rho
filedbrho='DataBase/rho20oxides.csv'
dbrho=GlassData(filedbrho)
dbrho.info()
dbrho.bounds()

# Determination of the molar volume and normalization of the molar volume
dbrho.oxidemolarmass()
dbrho.molarmass()
dbrho.y=dbrho.MolarM/dbrho.y
dbrho.normalize_y()

# Dataset of Tannealing
filedbTannealing='DataBase/Tannealing20oxides.csv'
dbTannealing=GlassData(filedbTannealing)
dbTannealing.info()
dbTannealing.bounds()
dbTannealing.normalize_y()

arch=[20,20,20]
nnTannealing=NeuralNetwork(dbTannealing.noxide,arch,'gelu','linear')
nnTannealing.compile(3.e-4)
nnTannealing.info()
nnTannealing.ArchName(arch)
nnTannealing.load('Models/nn'+dbTannealing.nameproperty+nnTannealing.namearch+'.h5')
nnTannealing.info()

# Dataset of  Tmelt
# -----------------
filedbTmelt='DataBase/Tmelt19oxides.csv'
dbTmelt=GlassData(filedbTmelt)
dbTmelt.info()
dbTmelt.bounds()
dbTmelt.normalize_y()

arch=[20,20,20]
nnTmelt=NeuralNetwork(dbTmelt.noxide,arch,'gelu','linear')
nnTmelt.compile(3.e-4)
nnTmelt.info()
nnTmelt.ArchName(arch)
nnTmelt.load('Models/nn'+dbTmelt.nameproperty+nnTmelt.namearch+'.h5')

# ---------------------------------------------------
# Loading of neural network model of the molar volume
# ---------------------------------------------------

arch=[20,20,20]
nnmolvol=NeuralNetwork(dbrho.noxide,arch,'gelu','linear')
nnmolvol.compile(3.e-4)
nnmolvol.info()
nnmolvol.load('Models/nnmolarvol3c20.h5')

# ----------------------------------------
# Loading of the database of Young modulus
# ----------------------------------------
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

# Determination of E/2G in SI unit
dbE.y=dbE.y/(2.*G)

# Determination of the Neural network model
dbE.normalize_y()
dbE.split(0.6,0.2)

# ----------------------------
# Generation of neural network
# ----------------------------
arch=[20,20,20]
nnmodelEsG=NeuralNetwork(dbE.noxide,arch,'gelu','linear')
nnmodelEsG.compile(1.e-4)
nnmodelEsG.info()
nnmodelEsG.load('Models/nnEsG3c20.h5')

# Getting of indexes of particular oxides
# ---------------------------------------

iSiO2=np.argwhere(dbE.oxide=='SiO2')[0][0]
iNa2O=np.argwhere(dbE.oxide=='Na2O')[0][0]
iCaO=np.argwhere(dbE.oxide=='CaO')[0][0]
iAl2O3=np.argwhere(dbE.oxide=='Al2O3')[0][0]
iMgO=np.argwhere(dbE.oxide=='MgO')[0][0]
iP2O5=np.argwhere(dbE.oxide=='P2O5')[0][0]
iB2O3=np.argwhere(dbE.oxide=='B2O3')[0][0]
iZnO=np.argwhere(dbE.oxide=='ZnO')[0][0]
iK2O=np.argwhere(dbE.oxide=='K2O')[0][0]

# -----------------------------------------
# composition of reference soda-lime-silica
# -----------------------------------------

# Number of glasses in the family
# -------------------------------
Nglass=10000

# Average composition of the glass family
# ---------------------------------------
xmol0=np.zeros(dbE.noxide)
xmol0[iSiO2]=0.7
xmol0[iNa2O]=0.1
xmol0[iCaO]=0.1

# Variation of molar fraction of the remaining oxides
# ---------------------------------------------------
dx0=(1.-np.sum(xmol0))/3.

# List of oxides of the glass family
# ----------------------------------
I=np.array([iSiO2,iCaO,iNa2O,dbE.noxide-1])

# List of the remaining oxides
# ----------------------------
J=[i for i in np.array(range(dbE.noxide)) if i not in I]

# Determination of glass composition and molar mass
# -------------------------------------------------
xglass,Mmolar=dbE.familyrandomcomposition(Nglass,I,J,xmol0,dx0)

# Determination of the glass properties
# -------------------------------------
rho=dbrho.GlassDensity(nnmolvol,dbrho.oxide,xglass)
E=1.e9*dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,xglass)
Tmelt=dbTmelt.physicaly(nnTmelt.model.predict(xglass[:,:-1]).transpose()[0,:])
Tg=dbTannealing.physicaly(nnTannealing.model.predict(xglass).transpose()[0,:])

# Research of characteristic
# --------------------------
iminEsrho=np.argmin(E/rho)
imaxEsrho=np.argmax(E/rho)
print('xglass=',xglass[imaxEsrho,:])
print('rho=',rho[imaxEsrho])
print('E=',E[imaxEsrho]*1.e-9)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

data1=pd.DataFrame(np.transpose(np.array([Tmelt-273.15,np.sqrt(E/rho)])),columns=['Tm','sqrt(E/rho)'])
sns.kdeplot(data1,x='Tm',y='sqrt(E/rho)',color='k',fill=True,ax=ax1,alpha=0.5)
ax1.annotate('soda-lime-silica',(np.mean(Tmelt-273.15),np.mean(np.sqrt(E/rho))+np.std(np.sqrt(E/rho))),color='k')


iminE=np.argmin(E)
imaxE=np.argmax(E)
print('xglass=',xglass[imaxE,:])
print('rho=',rho[imaxE])
print('E=',E[imaxE]*1.e-9)

data2=pd.DataFrame(np.transpose(np.array([rho,E*1.e-9])),columns=['rho','E'])
sns.kdeplot(data2,x='rho',y='E',color='k',fill=True,ax=ax2,alpha=0.5)
ax2.annotate('soda-lime-silica',(np.mean(rho),np.mean(E*1.e-9)+np.std(E*1.e-9)),color='k')

data3=pd.DataFrame(np.transpose(np.array([Tg-273.15,Tmelt-273.15])),columns=['Tg','Tmelt'])
sns.kdeplot(data3,x='Tg',y='Tmelt',color='k',fill=True,ax=ax3,alpha=0.5)
ax3.annotate('soda-lime-silica',(np.mean(Tg-273.15),np.mean(Tmelt-273.15)),color='k')

# -----------------------------------------
# composition of reference silico-phosphate
# -----------------------------------------

xmol0=np.zeros(dbE.noxide)
xmol0[iP2O5]=0.4
xmol0[iSiO2]=0.3
xmol0[iNa2O]=0.1
xmol0[iCaO]=0.1

dx0=(1.-np.sum(xmol0))/3.
I=np.array([iP2O5,iSiO2,iNa2O,iCaO,dbE.noxide-1])
J=[i for i in np.array(range(dbE.noxide)) if i not in I]
xglass,Mmolar=dbE.familyrandomcomposition(Nglass,I,J,xmol0,dx0)

# Determination of the glass properties
# -------------------------------------
rho=dbrho.GlassDensity(nnmolvol,dbrho.oxide,xglass)
E=1.e9*dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,xglass)
Tmelt=dbTmelt.physicaly(nnTmelt.model.predict(xglass[:,:-1]).transpose()[0,:])
Tg=dbTannealing.physicaly(nnTannealing.model.predict(xglass).transpose()[0,:])

imaxEsrho=np.argmax(E/rho)
iminEsrho=np.argmin(E/rho)
print('xglass=',xglass[imaxEsrho,:])
print('rho=',rho[imaxEsrho])
print('E=',E[imaxEsrho]*1.e-9)

data1=pd.DataFrame(np.transpose(np.array([Tmelt-273.15,np.sqrt(E/rho)])),columns=['Tm','sqrt(E/rho)'])
sns.kdeplot(data1,x='Tm',y='sqrt(E/rho)',color='b',fill=True,ax=ax1,alpha=0.5)
ax1.annotate('phosphosilicate',(np.mean(Tmelt-273.15),np.mean(np.sqrt(E/rho))+np.std(np.sqrt(E/rho))),color='b')

iminE=np.argmin(E)
imaxE=np.argmax(E)
print('xglass=',xglass[imaxE,:])
print('rho=',rho[imaxE])
print('E=',E[imaxE]*1.e-9)

data2=pd.DataFrame(np.transpose(np.array([rho,E*1.e-9])),columns=['rho','E'])
sns.kdeplot(data2,x='rho',y='E',color='b',fill=True,ax=ax2,alpha=0.5)
ax2.annotate('phosphosilicate',(np.mean(rho),np.mean(E*1.e-9)+np.std(E*1.e-9)),color='b')

data3=pd.DataFrame(np.transpose(np.array([Tg-273.15,Tmelt-273.15])),columns=['Tg','Tmelt'])
sns.kdeplot(data3,x='Tg',y='Tmelt',color='b',fill=True,ax=ax3,alpha=0.5)
ax3.annotate('phosphosilicate',(np.mean(Tg-273.15),np.mean(Tmelt-273.15)),color='b')

# ----------------
# Alumino-silicate
# ----------------

xmol0=np.zeros(dbE.noxide)
xmol0[iAl2O3]=0.1
xmol0[iSiO2]=0.6
xmol0[iNa2O]=0.1
xmol0[iCaO]=0.1
dx0=(1.-np.sum(xmol0))/3.
I=np.array([iAl2O3,iSiO2,iNa2O,iCaO,dbE.noxide-1])
J=[i for i in np.array(range(dbE.noxide)) if i not in I]
xglass,Mmolar=dbE.familyrandomcomposition(Nglass,I,J,xmol0,dx0)

# Determination of the glass properties
# -------------------------------------
rho=dbrho.GlassDensity(nnmolvol,dbrho.oxide,xglass)
E=1.e9*dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,xglass)
Tmelt=dbTmelt.physicaly(nnTmelt.model.predict(xglass[:,:-1]).transpose()[0,:])
Tg=dbTannealing.physicaly(nnTannealing.model.predict(xglass).transpose()[0,:])
iminEsrho=np.argmin(E/rho)
imaxEsrho=np.argmax(E/rho)
print('xglass=',xglass[imaxEsrho,:])
print('rho=',rho[imaxEsrho])
print('E=',E[imaxEsrho]*1.e-9)

data1=pd.DataFrame(np.transpose(np.array([Tmelt-273.15,np.sqrt(E/rho)])),columns=['Tm','sqrt(E/rho)'])
sns.kdeplot(data1,x='Tm',y='sqrt(E/rho)',color='g',fill=True,ax=ax1,alpha=0.5)
ax1.annotate('alumino-silicate',(np.mean(Tmelt-273.15),np.mean(np.sqrt(E/rho))+np.std(np.sqrt(E/rho))),color='g')

iminE=np.argmin(E)
imaxE=np.argmax(E)
print('xglass=',xglass[imaxE,:])
print('rho=',rho[imaxE])
print('E=',E[imaxE]*1.e-9)

data2=pd.DataFrame(np.transpose(np.array([rho,E*1.e-9])),columns=['rho','E'])
sns.kdeplot(data2,x='rho',y='E',color='g',fill=True,ax=ax2,alpha=0.5)
ax2.annotate('alumino-silicate',(np.mean(rho),np.mean(E*1.e-9)+np.std(E*1.e-9)),color='g')

data3=pd.DataFrame(np.transpose(np.array([Tg-273.15,Tmelt-273.15])),columns=['Tg','Tmelt'])
sns.kdeplot(data3,x='Tg',y='Tmelt',color='g',fill=True,ax=ax3,alpha=0.5)
ax3.annotate('alumino-silicate',(np.mean(Tg-273.15),np.mean(Tmelt-273.15)),color='g')

# -------------
# Boro-silicate
# -------------

xmol0=np.zeros(dbE.noxide)
xmol0[iB2O3]=0.1
xmol0[iSiO2]=0.6
xmol0[iNa2O]=0.1
xmol0[iCaO]=0.1
dx0=(1.-np.sum(xmol0))/3.
I=np.array([iB2O3,iSiO2,iNa2O,iCaO,dbE.noxide-1])
J=[i for i in np.array(range(dbE.noxide)) if i not in I]
xglass,Mmolar=dbE.familyrandomcomposition(Nglass,I,J,xmol0,dx0)

# Determination of the glass properties
# -------------------------------------
rho=dbrho.GlassDensity(nnmolvol,dbrho.oxide,xglass)
E=1.e9*dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,xglass)
Tmelt=dbTmelt.physicaly(nnTmelt.model.predict(xglass[:,:-1]).transpose()[0,:])
Tg=dbTannealing.physicaly(nnTannealing.model.predict(xglass).transpose()[0,:])
iminEsrho=np.argmin(E/rho)
imaxEsrho=np.argmax(E/rho)
print('xglass=',xglass[imaxEsrho,:])
print('rho=',rho[imaxEsrho])
print('E=',E[imaxEsrho]*1.e-9)

data1=pd.DataFrame(np.transpose(np.array([Tmelt-273.15,np.sqrt(E/rho)])),columns=['Tm','sqrt(E/rho)'])
sns.kdeplot(data1,x='Tm',y='sqrt(E/rho)',color='r',fill=True,ax=ax1,alpha=0.5)
ax1.annotate('boro-silicate',(np.mean(Tmelt-273.15),np.mean(np.sqrt(E/rho))+np.std(np.sqrt(E/rho))),color='r')

iminE=np.argmin(E)
imaxE=np.argmax(E)
print('xglass=',xglass[imaxE,:])
print('rho=',rho[imaxE])
print('E=',E[imaxE]*1.e-9)

data2=pd.DataFrame(np.transpose(np.array([rho,E*1.e-9])),columns=['rho','E'])
sns.kdeplot(data2,x='rho',y='E',color='r',fill=True,ax=ax2,alpha=0.5)
ax2.annotate('boro-silicate',(np.mean(rho),np.mean(E*1.e-9)+np.std(E*1.e-9)),color='r')

data3=pd.DataFrame(np.transpose(np.array([Tg-273.15,Tmelt-273.15])),columns=['Tg','Tmelt'])
sns.kdeplot(data3,x='Tg',y='Tmelt',color='r',fill=True,ax=ax3,alpha=0.5)
ax3.annotate('boro-silicate',(np.mean(Tg-273.15),np.mean(Tmelt-273.15)),color='r')

ax1.set_xlabel(r'$T_m$ (°C)')
ax1.set_ylabel(r'$\sqrt{E/\rho}$ (m/s)')
ax2.set_xlabel(r'$\rho$ (kg/m$^3$)')
ax2.set_ylabel(r'$E$ (GPa)')
ax3.set_xlabel(r'$T_g$ (°C)')
ax3.set_ylabel(r'$T_m$ (°C)')

plt.show()

# Saving of the figures

if (SAVEFIG):
    fig1.savefig(PATH+'sqrtEsrhovsTmchart.pdf',dpi=300,bbox_inches='tight')
    fig2.savefig(PATH+'Evsthochart.pdf',dpi=300,bbox_inches='tight')
    fig3.savefig(PATH+'TmvsTgchart.pdf',dpi=300,bbox_inches='tight')
#endif

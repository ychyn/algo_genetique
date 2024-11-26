#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:55:17 2023

Script used to train an ANN model to fit the molar volume obtained for the data-set on density.

The method is explained in [1].

Reference:

[1] F. Pigeonneau, M. Rondet, O. de Lataulade and E. Hachem (2024). Physical-informed deep learning prediction of solid and fluid mechanical properties of oxide glasses. J. Non-Cryst. Solids, under review, [[http://dx.doi.org/10.2139/ssrn.4997217]].

@author: fpigeonneau
"""
import numpy as np
import os
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from glassdata import GlassData
from network import NeuralNetwork

# Name of the database of glass
filedatabase='DataBase/rho20oxides.csv'
dbrho=GlassData(filedatabase)
dbrho.info()
dbrho.bounds()

# Determination of the molar volume
dbrho.oxidemolarmass()
dbrho.molarmass()
dbrho.y=dbrho.MolarM/dbrho.y

# Determination of the Neural network model
dbrho.normalize_y()
dbrho.split(0.6,0.2)
Nsample,Noxide=dbrho.shape()

# Determination of the training and validation sets
x_train,x_val=dbrho.x[dbrho.x_train:dbrho.x_valid],\
              dbrho.x[dbrho.x_valid:dbrho.x_test]
y_train,y_val=dbrho.y[dbrho.x_train:dbrho.x_valid],\
              dbrho.y[dbrho.x_valid:dbrho.x_test]

# Execution parameters
# --------------------

reload=True
nbepoch=5000
arch=[20,20,20]
batchsize=1024
errormolarvol=0.2
savefig=False
PATH='/home/fpigeonneau/ownCloud/Figures/MachineLearning/'
outlier=False
Cleaning=False
Saving=outlier or Cleaning

# ----------------------------
# Generation of neural network
# ----------------------------

nnmodel=NeuralNetwork(Noxide,arch,'gelu','linear')
nnmodel.compile(1.e-4)
nnmodel.ArchName(arch)
nnmodel.info()

# Training of the model
# ---------------------
modelfile='Models/nnmolarvol'+nnmodel.namearch+'.h5'

if (os.path.isfile(modelfile) and reload):
    nnmodel.load(modelfile)
#end if

nnmodel.fit(x_train,y_train,x_val,y_val,epochs=nbepoch,batch_size=batchsize)

# Plot training data
# ------------------
lossfig=PATH+'lossmolarvol'+nnmodel.namearch+'.pdf'
nnmodel.plot(lossfig,True)

# Saving of the model
# -------------------
nnmodel.save(modelfile)

# -----------------------
# Prediction of the model
# ----------------------

molarvol_nn_train=nnmodel.model.predict(x_train)
molarvol_nn_val=nnmodel.model.predict(x_val)
molarvol_nn_test=nnmodel.model.predict(dbrho.x[dbrho.x_test:Nsample-1])

# Computation of the Vickers hardness from the model
# --------------------------------------------------

molarvol_nn_train=dbrho.physicaly(molarvol_nn_train)
molarvol_nn_val=dbrho.physicaly(molarvol_nn_val)
molarvol_nn_test=dbrho.physicaly(molarvol_nn_test)

# Computation of the Vickers hardness from the data
# -------------------------------------------------

molarvol_actual_train=np.reshape(dbrho.physicaly(y_train),(-1,1))
molarvol_actual_val=np.reshape(dbrho.physicaly(y_val),(-1,1))
molarvol_actual_test=np.reshape(dbrho.physicaly(dbrho.y[dbrho.x_test:Nsample-1]),(-1,1))
ymin=np.min(np.concatenate([molarvol_actual_train,molarvol_actual_val,molarvol_actual_test]))
ymax=np.max(np.concatenate([molarvol_actual_train,molarvol_actual_val,molarvol_actual_test]))

# Determination of the R2 scores for the three sets of data
# ---------------------------------------------------------
r2_train=r2_score(molarvol_actual_train,molarvol_nn_train)
r2_val=r2_score(molarvol_actual_val,molarvol_nn_val)
r2_test=r2_score(molarvol_actual_test,molarvol_nn_test)

# Graphical plotting
# ------------------

plt.figure()
plt.plot(1.e6*molarvol_actual_train,1.e6*molarvol_nn_train,'ko')
plt.plot(1.e6*np.array([ymin,ymax]),1.e6*np.array([ymin,ymax]),'k',linewidth=2)
plt.plot(1.e6*np.array([ymin,ymax]),1.e6*np.array([ymin,ymax])*(1.+errormolarvol),'k-.',linewidth=2)
plt.plot(1.e6*np.array([ymin,ymax]),1.e6*np.array([ymin,ymax])*(1.-errormolarvol),'k-.',linewidth=2)

plt.xlabel(r'actual $V_{\mathrm{mol}}$',fontsize=12)
plt.ylabel(r'predicted $V_{\mathrm{mol}}$',fontsize=12)
plt.xlim((1.e6*ymin,1.e6*ymax))
plt.ylim((1.e6*ymin,1.e6*ymax))
plt.title(r'Training, $R^2$='+str(np.round(r2_train,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'predictedvsactualmolarvol'+namearch+'-train.pdf',dpi=300,bbox_inches='tight')
#endif

plt.figure()
plt.plot(1.e6*molarvol_actual_val,1.e6*molarvol_nn_val,'bo')
plt.plot(1.e6*np.array([ymin,ymax]),1.e6*np.array([ymin,ymax]),'b',linewidth=2)
plt.plot(1.e6*np.array([ymin,ymax]),1.e6*np.array([ymin,ymax])*(1.+errormolarvol),'b-.',linewidth=2)
plt.plot(1.e6*np.array([ymin,ymax]),1.e6*np.array([ymin,ymax])*(1.-errormolarvol),'b-.',linewidth=2)
plt.xlabel(r'actual $V_{\mathrm{mol}}$',fontsize=12)
plt.ylabel(r'predicted $V_{\mathrm{mol}}$',fontsize=12)
plt.xlim((1.e6*ymin,1.e6*ymax))
plt.ylim((1.e6*ymin,1.e6*ymax))
plt.title(r'Validation, $R^2$='+str(np.round(r2_val,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'predictedvsactualmolarvol'+namearch+'-val.pdf',dpi=300,bbox_inches='tight')
#endif

plt.figure()
plt.plot(1.e6*molarvol_actual_test,1.e6*molarvol_nn_test,'go')
plt.plot(1.e6*np.array([ymin,ymax]),1.e6*np.array([ymin,ymax]),'g',linewidth=2)
plt.plot(1.e6*np.array([ymin,ymax]),1.e6*np.array([ymin,ymax])*(1.+errormolarvol),'g-.',linewidth=2)
plt.plot(1.e6*np.array([ymin,ymax]),1.e6*np.array([ymin,ymax])*(1.-errormolarvol),'g-.',linewidth=2)
plt.xlabel(r'actual $V_{\mathrm{mol}}$',fontsize=12)
plt.ylabel(r'predicted $V_{\mathrm{mol}}$',fontsize=12)
plt.xlim((1.e6*ymin,1.e6*ymax))
plt.ylim((1.e6*ymin,1.e6*ymax))
plt.title(r'Test, $R^2$='+str(np.round(r2_test,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'predictedvsactualmolarvol'+namearch+'-test.pdf',dpi=300,bbox_inches='tight')
#endif

plt.show()

# Cleaning data
if (outlier):
    itodrop=np.argwhere(dbrho.y==np.max(dbrho.y))[:,0]
#end if

if (Cleaning):
    errortraining=(molarvol_nn_train-molarvol_actual_train)/molarvol_actual_train
    errorvalidation=(molarvol_nn_val-molarvol_actual_val)/molarvol_actual_val
    errortest=(molarvol_nn_test-molarvol_actual_test)/molarvol_actual_test

    itraining=np.argwhere(np.abs(errortraining)>errormolarvol)[:,0]
    ivalidation=np.argwhere(np.abs(errorvalidation)>errormolarvol)[:,0]
    itest=np.argwhere(np.abs(errortest)>errormolarvol)[:,0]
    itodrop=np.concatenate([itraining,ivalidation,itest])
#end if

if (Saving):
    # Molar volume nonormalized
    dbrho.y=dbrho.physicaly(dbrho.y)
    # Determination of the density
    dbrho.y=dbrho.MolarM/dbrho.y
    # Saving and remove of the worst data
    dbrho.savedata(filedatabase,itodrop)
# end if

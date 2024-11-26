#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:43:23 2023

Script used to train an ANN model to fit the atomic packing factor obtained for the data-set on Young's modulus. The method is explained in [1].

Reference:

[1] F. Pigeonneau, M. Rondet, O. de Lataulade and E. Hachem (2024). Physical-informed deep learning prediction of solid and fluid mechanical properties of oxide glasses. J. Non-Cryst. Solids, under review, [[http://dx.doi.org/10.2139/ssrn.4997217]].


@author: fpigeonneau
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
import os
from glassdata import GlassData
from network import NeuralNetwork

# ----------------------------------------
# Loading of the database of Young modulus
# ----------------------------------------

filedbE='DataBase/E20oxides.csv'
dbE=GlassData(filedbE)
dbE.bounds()

# ------------------------------
# Loading of dissociation energy
# ------------------------------

datadisso=pd.read_csv('dissociationenergy.csv')
G=np.zeros(dbE.nsample)
for i in range(dbE.nsample):
    G[i]=np.sum(datadisso['G'].values*dbE.x[i,:])
#end for

# Determination of E*Vmol/G unité SI
dbE.y=dbE.y/(2.*G)

# Determination of the Neural network model
dbE.normalize_y()
dbE.split(0.6,0.2)

# Execution parameters
# --------------------

reload=True
Nfold=0
Nepoch=5000
batchsize=1024
savefig=False
PATH='/home/fpigeonneau/ownCloud/Figures/MachineLearning/'
errorVt=0.2
Cleaning=False
outlier=False

# -----------------------------
# Generattion of neural network
# -----------------------------

arch=[20,20]
nnmodelEsG=NeuralNetwork(dbE.noxide,arch,'gelu','linear')
nnmodelEsG.ArchName(arch)
nnmodelEsG.compile(1.e-4)
nnmodelEsG.info()

# ---------------------
# Training of the model
# ---------------------

modelfile='Models/nnEsG'+nnmodelEsG.namearch+'.h5'
if (os.path.isfile(modelfile) and reload):
    nnmodelEsG.load(modelfile)
# #end if

tstart=time.time()
lossfile=PATH+'lossEsG'+namearch+'.pdf'

if (Nfold>0):
    # Training by cross validation
    # ----------------------------
    
    kfold=KFold(n_splits=Nfold,shuffle=True)
    
    ifold=0
    for train_index,val_index in kfold.split(dbE.x[dbE.x_train:dbE.x_test]):    
        # Determination of the training and validation sets
        # -------------------------------------------------
        
        x_train,x_val=dbE.x[train_index],dbE.x[val_index]
        y_train,y_val=dbE.y[train_index],dbE.y[val_index]
        
        # Training of the model
        # ---------------------
        
        nnmodelEsG.fit(x_train,y_train,x_val,y_val,epochs=500,batch_size=batchsize)
        
        # Plot training data
        # ------------------
        
        nnmodelEsG.plot(lossfile[:-4]+str(ifold)+'.pdf')
        
        # Saving of the model
        # -------------------
        
        nnmodelEsG.save(modelfile[:,-3]+str(ifold)+'.h5')
        
        # Incrementaion of ifold
        # ----------------------
        ifold+=1
    # end for
else:    

    # Determination of the training and validation sets
    x_train, x_val = dbE.x[dbE.x_train:dbE.x_valid],dbE.x[dbE.x_valid:dbE.x_test]
    y_train, y_val = dbE.y[dbE.x_train:dbE.x_valid],dbE.y[dbE.x_valid:dbE.x_test]

    # Training of the model
    # ---------------------
    
    nnmodelEsG.fit(x_train,y_train,x_val,y_val,epochs=Nepoch,batch_size=batchsize)

    # Plot training data
    # ------------------
    
    nnmodelEsG.plot(lossfile,False)
   
    # Saving of the model
    # -------------------

    nnmodelEsG.save(modelfile)
# end if

# -----------------------
# Prediction of the model
# ----------------------

EsG_nn_train=nnmodelEsG.model.predict(x_train).transpose()[0,:]
EsG_nn_val=nnmodelEsG.model.predict(x_val).transpose()[0,:]
EsG_nn_test=nnmodelEsG.model.predict(dbE.x[dbE.x_test:dbE.nsample-1]).transpose()[0,:]

# Renormalization of the result in physical dimension
# ---------------------------------------------------

EsG_nn_train=dbE.physicaly(EsG_nn_train)
EsG_nn_val=dbE.physicaly(EsG_nn_val)
EsG_nn_test=dbE.physicaly(EsG_nn_test)

# Computation of EsG from the data base
# ---------------------------------------

EsG_actual_train=dbE.physicaly(y_train)
EsG_actual_val=dbE.physicaly(y_val)
EsG_actual_test=dbE.physicaly(dbE.y[dbE.x_test:dbE.nsample-1])
ymin=np.min(np.concatenate([EsG_actual_train,EsG_actual_val,EsG_actual_test]))
ymax=np.max(np.concatenate([EsG_actual_train,EsG_actual_val,EsG_actual_test]))

# Determination of the R2 scores for the three sets of data
# ---------------------------------------------------------

r2_train=r2_score(EsG_actual_train,EsG_nn_train)
r2_val=r2_score(EsG_actual_val,EsG_nn_val)
r2_test=r2_score(EsG_actual_test,EsG_nn_test)

# ------------------
# Graphical plotting
# ------------------

plt.figure()
plt.plot(EsG_actual_train,EsG_nn_train,'ko')
plt.plot(np.array([ymin,ymax])*1e0,np.array([ymin,ymax])*1e0,'k',linewidth=2)
plt.plot(np.array([ymin,ymax])*1e0,np.array([ymin,ymax])*(1.+errorVt)*1e0,'k-.',linewidth=2)
plt.plot(np.array([ymin,ymax])*1e0,np.array([ymin,ymax])*(1.-errorVt)*1e0,'k-.',linewidth=2)
plt.xlabel('actual $V_t$',fontsize=14)
plt.ylabel('predicted $V_t$',fontsize=14)
plt.xlim((ymin*1e0,ymax*1e0))
plt.ylim((ymin*1e0,ymax*1e0))
plt.title(r'$R^2$='+str(np.round(r2_train,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'/predictedvsactualEsG'+namearch+'-train.pdf',dpi=300,bbox_inches='tight')
#endif

plt.figure()
plt.plot(EsG_actual_val*1e0,EsG_nn_val*1e0,'bo')
plt.plot(np.array([ymin,ymax])*1e0,np.array([ymin,ymax])*1e0,'b',linewidth=2)
plt.plot(np.array([ymin,ymax])*1e0,np.array([ymin,ymax])*(1.+errorVt)*1e0,'b-.',linewidth=2)
plt.plot(np.array([ymin,ymax])*1e0,np.array([ymin,ymax])*(1.-errorVt)*1e0,'b-.',linewidth=2)
plt.xlabel('actual $V_t$',fontsize=14)
plt.ylabel('predicted $V_t$',fontsize=14)
plt.xlim((ymin*1e0,ymax*1e0))
plt.ylim((ymin*1e0,ymax*1e0))
plt.title(r'$R^2$='+str(np.round(r2_val,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'predictedvsactualEsG'+namearch+'-val.pdf',dpi=300,bbox_inches='tight')
#endif

plt.figure()
plt.plot(EsG_actual_test*1e0,EsG_nn_test*1e0,'go',label='Test, R2='+str(r2_test))
plt.plot(np.array([ymin,ymax])*1e0,np.array([ymin,ymax])*1e0,'g',linewidth=2)
plt.plot(np.array([ymin,ymax])*1e0,np.array([ymin,ymax])*(1.+errorVt)*1e0,'g-.',linewidth=2)
plt.plot(np.array([ymin,ymax])*1e0,np.array([ymin,ymax])*(1.-errorVt)*1e0,'g-.',linewidth=2)
plt.xlabel('actual $V_t$',fontsize=14)
plt.ylabel('predicted $V_t$',fontsize=14)
plt.xlim((ymin*1e0,ymax*1e0))
plt.ylim((ymin*1e0,ymax*1e0))
plt.title(r'$R^2$='+str(np.round(r2_test,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'/predictedvsactualEsG'+namearch+'-test.pdf',dpi=300,bbox_inches='tight')
#endif
plt.show()

# -------------
# Cleaning data
# -------------

if (Cleaning):
    errortraining=(EsG_nn_train-EsG_actual_train)/EsG_actual_train
    errorvalidation=(EsG_nn_val-EsG_actual_val)/EsG_actual_val
    errortest=(EsG_nn_test-EsG_actual_test)/EsG_actual_test

    itraining=np.argwhere(np.abs(errortraining)>errorVt)[:,0]
    ivalidation=np.argwhere(np.abs(errorvalidation)>errorVt)[:,0]
    itest=np.argwhere(np.abs(errortest)>errorVt)[:,0]
    itodrop=np.concatenate([itraining,ivalidation,itest])

    # Molar volume nonormalized
    dbE.y=dbE.physicaly(dbE.y)
    
    # Back to E in GPa
    dbE.y=2.*dbE.y*G
    
    # Saving and remove of the worst data
    dbE.savedata('DataBase/Eclean.csv',itodrop)
# end if

# Outlier removal
if (outlier):
    itodrop=np.argwhere(dbE.y==np.max(dbE.y))[:,0]
    
    # Molar volume nonormalized
    dbE.y=dbE.physicaly(dbE.y)
    
    # Back to E in GPa
    dbE.y=2.*dbE.y*G
    
    # Saving and remove of the worst data
    dbE.savedata(filedbE,itodrop)
#end if

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:55:17 2023

Script achieving the training of an ANN network to fit a glass property with a data-set in the
directory DataBase. The model is saved in directory Models.

@author: fpigeonneau
"""

# Python modules

import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Our own modules

from glassdata import GlassData
from network import NeuralNetwork

# Name of the database of glass
filedatabase='DataBase/Tsoft20oxides.csv'
db=GlassData(filedatabase)
db.info()
db.bounds()

# Determination of the Neural network model
db.normalize_y()
db.split(0.6,0.2)
Nsample,Noxide=db.shape()

# Determination of the training and validation sets
x_train, x_val = db.x[db.x_train:db.x_valid],db.x[db.x_valid:db.x_test]
y_train, y_val = db.y[db.x_train:db.x_valid],db.y[db.x_valid:db.x_test]

# Parameters
# ----------

reload=True
Nfold=0
Nepoch=5000
batchsize=256
errormax=0.1
PATH='/home/fpigeonneau/ownCloud/Figures/MachineLearning/'
savefig=False
Cleaning=False

# ----------------------------
# Generation of neural network
# ----------------------------

arch=[20,20,20]
Nhidden=np.size(arch)
nnmodel=NeuralNetwork(Noxide,arch,'gelu','linear')
nnmodel.ArchName(arch)
nnmodel.compile(1.5e-4)
nnmodel.info()

# Training of the model
# ---------------------

modelfile='Models/nn'+db.nameproperty+nnmodel.namearch+'.h5'
if (os.path.isfile(modelfile) and reload):
    nnmodel.load(modelfile)
#end if

lossfig=PATH+'loss'+db.nameproperty+nnmodel.namearch+'.pdf'

if (Nfold>0):
    # Training by cross validation
    # ----------------------------
    
    kfold=KFold(n_splits=Nfold,shuffle=True)
    
    ifold=0
    for train_index,val_index in kfold.split(db.x[db.x_train:db.x_test]):
        # Determination of the training and validation sets
        # -------------------------------------------------
        
        x_train,x_val=db.x[train_index],db.x[val_index]
        y_train,y_val=db.y[train_index],db.y[val_index]
        
        # Training of the model
        # ---------------------
        
        nnmodel.fit(x_train,y_train,x_val,y_val,epochs=Nepoch,batch_size=batchsize)
        
        # Saving of the model
        # -------------------
        
        nnmodel.save(modelfile)
        
        # Incrementaion of ifold
        # ----------------------
        ifold+=1
    # end for
else:    

    # Determination of the training and validation sets
    x_train,x_val=db.x[db.x_train:db.x_valid],db.x[db.x_valid:db.x_test]
    y_train,y_val=db.y[db.x_train:db.x_valid],db.y[db.x_valid:db.x_test]

    # Training of the model
    # ---------------------
    
    nnmodel.fit(x_train,y_train,x_val,y_val,epochs=Nepoch,batch_size=batchsize)

    # Plot training data
    # ------------------
    
    nnmodel.plot(lossfig,False)
   
    # Saving of the model
    # -------------------

    nnmodel.save(modelfile)
# end if

# -----------------------
# Prediction of the model
# ----------------------

y_nn_train=nnmodel.model.predict(x_train)
y_nn_val=nnmodel.model.predict(x_val)
y_nn_test=nnmodel.model.predict(db.x[db.x_test:Nsample-1])

# Computation of the Vickers hardness from the model
# --------------------------------------------------

y_nn_train=db.physicaly(y_nn_train)
y_nn_val=db.physicaly(y_nn_val)
y_nn_test=db.physicaly(y_nn_test)

# Computation of the Vickers hardness from the data
# -------------------------------------------------

y_actual_train=np.reshape(db.physicaly(y_train),(-1,1))
y_actual_val=np.reshape(db.physicaly(y_val),(-1,1))
y_actual_test=np.reshape(db.physicaly(db.y[db.x_test:Nsample-1]),(-1,1))
ymin=np.min(np.concatenate([y_actual_train,y_actual_val,y_actual_test]))
ymax=np.max(np.concatenate([y_actual_train,y_actual_val,y_actual_test]))

# Determination of the R2 scores for the three sets of data
# ---------------------------------------------------------
r2_train=r2_score(y_actual_train,y_nn_train)
r2_val=r2_score(y_actual_val,y_nn_val)
r2_test=r2_score(y_actual_test,y_nn_test)

# Graphical plotting
# ------------------

plt.figure()
plt.plot(y_actual_train,y_nn_train,'ko')
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax]),'k',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.+errormax),'k-.',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.-errormax),'k-.',linewidth=2)
plt.xlabel('actual '+db.nameproperty,fontsize=12)
plt.ylabel('predicted '+db.nameproperty,fontsize=12)
plt.xlim((ymin,ymax))
plt.ylim((ymin,ymax))
plt.title(r'Training, $R^2$='+str(np.round(r2_train,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'predictedvsactual'+db.nameproperty+nnmodel.namearch+'-train.pdf',dpi=300,bbox_inches='tight')
#endif

plt.figure()
plt.plot(y_actual_val,y_nn_val,'bo')
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax]),'b',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.+errormax),'b-.',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.-errormax),'b-.',linewidth=2)
plt.xlabel('actual '+db.nameproperty,fontsize=12)
plt.ylabel('predicted '+db.nameproperty,fontsize=12)
plt.xlim((ymin,ymax))
plt.ylim((ymin,ymax))
plt.title(r'Validation, $R^2$='+str(np.round(r2_val,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'predictedvsactual'+db.nameproperty+nnmodel.namearch+'-val.pdf',dpi=300,bbox_inches='tight')
#endif

plt.figure()
plt.plot(y_actual_test,y_nn_test,'go')
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax]),'g',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.+errormax),'g-.',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.-errormax),'g-.',linewidth=2)
plt.xlabel('actual '+db.nameproperty,fontsize=12)
plt.ylabel('predicted '+db.nameproperty,fontsize=12)
plt.xlim((ymin,ymax))
plt.ylim((ymin,ymax))
plt.title(r'Test, $R^2$='+str(np.round(r2_test,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'predictedvsactual'+db.nameproperty+nnmodel.namearch+'-test.pdf',dpi=300,bbox_inches='tight')
#endif
plt.show()

# Cleaning data
# -------------

if (Cleaning):
    errortraining=(y_nn_train-y_actual_train)/y_actual_train
    errorvalidation=(y_nn_val-y_actual_val)/y_actual_val
    errortest=(y_nn_test-y_actual_test)/y_actual_test

    itraining=np.argwhere(np.abs(errortraining)>errormax)[:,0]
    ivalidation=np.argwhere(np.abs(errorvalidation)>errormax)[:,0]
    itest=np.argwhere(np.abs(errortest)>errormax)[:,0]
    itodrop=np.concatenate([itraining,ivalidation,itest])

    # Saving of the new database
    db.y=db.physicaly(db.y)
    db.savedata(filedatabase,itodrop)
# end if

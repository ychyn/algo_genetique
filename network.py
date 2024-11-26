#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 08:40:03 2023

Definition of a class network to create a neutral network.

@author: fpigeonneau
"""

# Generic imports
# ---------------

import os
import numpy             as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

# Keras-specific imports
from keras.models       import Sequential
from keras.optimizers   import Adam
from keras.layers       import Input,Dense
from keras.initializers import Orthogonal

# =============
# Network class
# =============

class NeuralNetwork():
    """
    Class used to define, fit an ANN to determine a glass property.

    Parameters:
    -----------
    shape: Integers corresponding of the number of features (= number of oxides) and
    number of samples in the data-set.
    arch: Array defining the number of layers and each associated neurals.
    actv: Activation function of hidden layers in ANN model.
    final: Function of the output layer.
    k_init: Kernel of initialization of parameters of the ANN.
    history: Information about the fitting of the ANN.
    namearch: Name of architecture used to save and call a model.

    Methods:
    --------

    build(self,shape,arch,actv,final): Build model.
    compile(self, lr=1.0e-3): Compile model.
    info(self): Print infos about model.
    fit(self,x_train,y_train,x_val,y_val,epochs,batch_size): Fitting model on data.
    plot(self,outputfile='loss.png',savefig=False): Plot accuracy and loss.
    save(self, filename): Save model to file
    load(self,filename): Load model from file
    ArchName(self,arch): Name summarizing the architecture of the model.

    """

    
    # -----------------------
    # Initialisation function
    # -----------------------
    
    def __init__(self, shape=None, arch=[32,32], actv='swish',final='softplus'):
        self.shape   = shape
        self.arch    = arch
        self.actv    = actv
        self.final   = final
        self.k_init  = 'orthogonal'
        self.history = None
        self.namearch=None
        if (shape is not None):
            self.build(shape,arch,actv,final)
        #end if
    # end __init__

    # -----------
    # Build model
    # -----------
    
    def build(self,shape,arch,actv,final):
        self.shape = shape
        self.arch  = arch
        self.actv  = actv
        self.final = final
        
        # Definition of the model
        # -----------------------
        self.model = Sequential()
        self.model.add(Input(shape=(shape,)))
        for i in range(len(self.arch)):
            self.model.add(Dense(self.arch[i],activation=self.actv,
                                 kernel_initializer=self.k_init))
        #end for
        self.model.add(Dense(1,activation=self.final))
    # end build
    
    # -------------
    # Compile model
    # -------------
    
    def compile(self, lr=1.0e-3):
        # Compile model
        self.model.compile(loss='mse',optimizer=Adam(learning_rate=lr),
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])
    # end compile
    
    # -----------------------
    # Print infos about model
    # -----------------------
    
    def info(self):
        print('**************')
        self.model.summary()
    # end info
    
    # ---------------------
    # Fitting model on data
    # ---------------------
    
    def fit(self,x_train,y_train,x_val,y_val,epochs,batch_size):
        self.history=self.model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,
                                    validation_data=(x_val,y_val),verbose=1)
    # end fit
    
    # ----------------------
    # Plot accuracy and loss
    # ----------------------
    
    def plot(self,outputfile='loss.png',savefig=False):
        if (self.history is None):
            print('**************')
            print('Error in network.plot()')
            print('Training history is empty')
            sys.exit()
        #end if
        train_loss = self.history.history['loss']
        valid_loss = self.history.history['val_loss']
        epochs     = range(len(train_loss))
        np.savetxt('loss.dat', np.transpose([epochs,train_loss,valid_loss]))
        plt.figure()
        plt.semilogy(epochs, train_loss, 'k', label='Training loss')
        plt.semilogy(epochs, valid_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if (savefig):
            plt.savefig(outputfile)
        #endif
        plt.show()
    # end plot
    
    # ------------------
    # Save model to file
    # ------------------
    
    def save(self, filename):
        self.model.save(filename)
    # end save
    
    # --------------------
    # Load model from file
    # --------------------
    
    def load(self,filename):
        if (not os.path.isfile(filename)):
            print('Could not find model file')
            sys.exit()
        #end if
        #self.model=self.model.load_model(filename,compile=True)
        self.model=tf.keras.models.load_model(filename)
    # end load

    # --------
    # ArchName
    # --------
    
    def ArchName(self,arch):
        
        Nhidden=np.size(arch)
        self.namearch=str(Nhidden)+'c'
        self.namearch+=str(arch[0])
        for i in range(1,Nhidden):
            if (arch[i]!=arch[i-1]):
                self.namearch+=str(arch[i])
            #end if
        #end for
    #end ArchName

# end class network

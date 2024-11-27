#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 08:40:03 2023

Definition of a class corresponding of a data set used to create a model of
neutral network.

@author: F. Pigeonneau
"""

# Generic imports
# ---------------

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import random
from molmass import Formula

# =============
# Dataset class
# =============

class GlassData():
    """
    Class GlassData

    Variables:
    ----------
    
    oxide: Array of string
    Names of the oxides of the glass composition.
    nameproperty: String
    Name of the property of the glasses.
    nsample: Integer
    Number of the glass.
    noxide: Integer
    Number of oxides constituting the glass composition.
    nbO: Array of integer of the noxide oxides
    number of oxygen in each oxide.
    cation: Array of string of the noxide oxides
    name of the cation in each oxide.
    nbcation: Array of integer of the noxide oxides
    number of cation in each oxide.
    Moxide: Array of float of the noxide oxides
    molar mass of each oxide.
    x: Array of float (nsample,noxide)
    Molar fraction of the noxide oxides of the nsample glass composition.
    y: Array of float of the nsample glass composition.
    Property of glass of the data set.
    xmin: Array of float of the noxide oxides
    Minimum of the molar fraction of each oxide (=0)
    xmax: Array of float of the noxide oxides
    Maximum of the molar fraction of each oxide in the data set.
    occurence: Array of integer
    Numbers of composition for which the molar fraction of each oxide is stricly different to zero.
    MolarM: Array of float of  the nsample glass composition.
    Molar mass of each glass composition in kg/mol.
    """
    
    # -----------------------
    # Initialisation function
    # -----------------------
    
    def __init__(self, filename=None):

        self.oxide=None
        self.nameproperty=None
        self.nsample=None
        self.noxide=None
        self.nbO=None
        self.cation=None
        self.nbcation=None
        self.Moxide=None
        self.x= None
        self.y= None
        self.xmin= None
        self.xmax= None
        self.occurence= None
        self.MolarM= None
        if (filename is not None):
            self.load(filename)
            self.info()
        # end if
    # end __init__
    
    # --------------------------------
    # Loading of dataset from csv file
    # --------------------------------
    
    def load(self,filename):
        # Reading of csv file
        data=pd.read_csv(filename)
        
        # Shuffling of data
        data=data.sample(frac=1)
        
        # Names of oxides
        self.oxide=data.columns[1:-1]
        
        # Name of the property
        self.nameproperty=data.columns[-1]
        
        # Molar fraction of oxides
        self.x=data.values[:,1:-1]
        
        # Values of the property
        self.y=data.values[:,-1]
        
    # end load
    
    #--------------------------
    # Print infos about dataset
    # ------------------------
    
    def info(self):
        # Determination of the numbers of rows and columns
        self.nsample, self.noxide = self.shape()

        print('**************')
        print('Finished loading dataset')
        print('Nb of samples: {}'.format(self.nsample))
        print('Nb of components: {}'.format(self.noxide))
    # end info

    # -----------------------------
    # Return shape of input dataset
    # -----------------------------
    
    def shape(self):
        if (self.x is not None):
            return self.x.shape[0], self.x.shape[1]
        else:
            print('**************************')
            print('Error in glassdata.shape()')
            print('Empty dataset')
            print('**************************')
            sys.exit()
        #end if
    # end shape

    def OxygenNumber(self):
        """
        Determination of the number of oxygen in the list of oxides constituting
        the glass composition.
        """
        
        self.nbO=np.zeros(self.noxide)
        for i in range(self.noxide):
            if (self.oxide[i].find('O')>0):
                if len(self.oxide[i])==(self.oxide[i].find('O')+1):
                    self.nbO[i]=1
                else:
                    self.nbO[i]=int(self.oxide[i][-1])
                #end if
            #end if
        #end for
    #end OxygenNumber

    def namenumbercation(self):
        
        # ---------------------------------------
        # Determination of the name of the cation
        # ---------------------------------------
        
        self.cation=[]
        for i in range(self.noxide):
            if (self.nbO[i]==1):
                self.cation=np.append(self.cation,self.oxide[i][:-1])
            else:
                self.cation=np.append(self.cation,self.oxide[i][:-2])
            #end if
        #end for
        
        self.nbcation=np.ones(self.noxide)
        for i in range(self.noxide):
            for j in self.cation[i]:
                if (j.isdigit()):
                    self.nbcation[i]=np.float64(j)
                #end if
            #end for
            if (self.nbcation[i]>1.):
                self.cation[i]=self.cation[i][:-1]
            #end if
        #end for
    #end namenumbercation

    # --------------
    # oxidemolarmass
    # --------------
    
    def oxidemolarmass(self):
        """
        Molar mass of each oxide in kg/mol.
        """
        self.Moxide=np.zeros(self.noxide)
        for i in range(self.noxide):
            f=Formula(self.oxide[i])
            self.Moxide[i]=f.mass*1.e-3
        #end for
    #end oxidemolarmass
    
    # -------------------------
    # Molar mass for each glass
    # -------------------------
    
    def molarmass(self):
        self.MolarM=np.zeros(self.nsample)
        for i in range(self.nsample):
            self.MolarM[i]=np.sum(self.x[i,:]*self.Moxide)
        #end for
    #end molarmass
    
    # --------------------
    # Bounds of each oxide
    # --------------------
    
    def bounds(self):
        self.xmin=np.zeros(self.noxide)
        self.xmax=np.zeros(self.noxide)
        self.occurence=np.zeros(self.noxide)
        for i in range(self.noxide):
            listarg=np.argwhere(self.x[:,i]!=0.)
            nozerovalue=self.x[listarg,i]
            self.xmin[i]=np.min(nozerovalue)
            self.xmax[i]=np.max(nozerovalue)
            self.occurence[i]=np.size(nozerovalue)
        #end for
    #end bounds
    
    # ----------------------
    # Normalize of x dataset
    # ----------------------
    
    def normalize_x(self):
        #  Determination of the mean and standard deviation of x
        # -----------------------------------------------------
        self.x_avg = np.average(self.x,axis=0)
        self.x_std = np.std(self.x,axis=0)
        
        # Normalization of y
        # ------------------
        self.x = (self.x - self.x_avg)/self.x_std
        self.normalized_x = True
    #end normalize_x
    
    # ----------------------
    # Normalize of y dataset
    # ----------------------
    
    def normalize_y(self):
        # Determination of the mean and standard deviation of y
        # -----------------------------------------------------
        self.y_avg = np.average(self.y)
        self.y_std = np.std(self.y)

        # Normalization of y
        # ------------------
        self.y = (self.y - self.y_avg)/self.y_std
        self.normalized_y = True
    #end normalize_y

    # -------------------------------------------------------
    # Define split indices into training, validation and test
    # -------------------------------------------------------
    
    def split(self,train,valid):
        # Determination of the fraction of data test
        test=1.-train-valid
        if (test<0.):
            print('Error in split.')
            sys.exit()
        #end if
        
        # Compute sizes
        n_rows, n_cols = self.shape()
        self.x_train   = 0
        self.x_valid   = math.floor(n_rows*train) + 0
        self.x_test    = math.floor(n_rows*valid) + self.x_valid

        # Print infos
        print('**************')
        print('Finished splitting dataset')
        print('Training   indices: {} to {}'.format(self.x_train, self.x_valid-1))
        print('Validation indices: {} to {}'.format(self.x_valid, self.x_test-1))
        print('Testing    indices: {} to {}'.format(self.x_test,n_rows-1))
    #end split

    # --------------------------
    # Back to physical unit of x
    # --------------------------
    
    def physicalx(self,x):
        return self.x_avg+self.x_std*x
    #end physicalx
    
    # --------------------------
    # Back to physical unit of y
    # --------------------------
    
    def physicaly(self,y):
        return self.y_avg+self.y_std*y
    #end physicaly

    # --------
    # savedata
    # --------
    
    def savedata(self,filesavedata,listtodrop=None):
        """
        Saving of the data set in a csv file with a removal of a list of samples if it exists.

        Parameters
        ----------
        filesavedata: String
        Name of the file of the data set.
        listtodrop: Array of integers
        Indexes of samples to remove of the data set.
        
        """
        
        # Preparation of data
        y=np.reshape(self.y,(-1,1))
        xy=np.append(self.x,y,axis=1)
        data=pd.DataFrame(xy,columns=np.append(self.oxide,self.nameproperty))
        
        # Drop of the data if it is needed
        if (listtodrop is not None):
            data=data.drop(listtodrop)
        #end if
        
        # Saving the data in the csv file
        data.to_csv(filesavedata)
    #end savedata

    # -----------
    # xglasstoxdb
    # -----------
    
    def xglasstoxdb(self,oxide,xglass):
        """
        Copy the array of the glass composition xglass with a list of oxide in the 
        formatted array of the dataset.

        Parameters
        ----------
        oxide: Array of string
        List of oxides with a composition given by xglass
        xglass: Array of floats (Nglass,Noxide)
        Compositions of the Nglass with the list of oxide.

        Return
        ------
        xdb: Array of floats ((Nglass,self.noxide)
        Compositions of the Nglass on the list of the data set.
        
        """
        
        # Determination of the numbers of glass compositions and oxides
        Nglass=np.size(xglass,0)
        Noxide=np.size(xglass,1)
    
        # Initialisation of the molar fraction for the database
        xdb=np.zeros((Nglass,self.noxide))
    
        # Copy in the array of oxide of the xoxide of the composition of the database
        for i in range(Nglass):
            for j in range(Noxide):
                if (oxide[j] in self.oxide):
                    idb=self.oxide.get_loc(oxide[j])
                    xdb[i,idb]=xglass[i,j]
                else:
                    print('The oxide ',oxide[j],' is not in the list of oxides of the database.')
                    sys.exit()
                #endif
            #end for
        #end for
        
        # Return of the composition in the glass samples
        return xdb
    # end xglasstoxdb

    # -------------
    #  datacleaning
    # -------------

    def datacleaning(self,filename,xtotal,probamin,probamax,xminor,minoxidefraction,filteringoxide,Plot):
        """
        Cleaning of the dataset gathered in the filename extracted from Interglad. Here, all
        unexpected characters and lines are removed. Alphanumeric characters are transformed into float.
        
        Parameters
        ----------
        filename : String
        Name of the database built from the database Interglad V8.
        xtotal : Float
        Bound admited in the sum of oxides for each glass sample.
        probamin : Float
        Minimum of the probability of occurence of oxide or proprety.
        probamax : Float
        Maximum of the probability of occurence of oxide or proprety.
        xminor : Float
        Minimum value of the oxide to be considered to be significant.
        minoxidefraction : Float
        Minimum fraction of glasses with a specific oxide in the composition
        Plot : Boolean
        Parameter giving the choice to plot the pdf of the property.
        
        Returns
        -------
        self.oxide : Array of strings
        List of the names of oxides.
        self.x : Array of dimension 2 of float 
        The molar fraction of glass samples.
        self.y : Array of float
        Property of the glass samples.
        """
        
        # Reading of Interglad data file 
        data=pd.read_csv(filename,skiprows=27)
        
        # Dropping of the five first columns
        data=data.drop(data.columns[:5],axis=1)
        
        # Conversion of NaN in 0
        data=data.apply(pd.to_numeric,errors='coerce').replace(np.nan,0,regex=True)
        
        # Dropping duplicated data
        data=data.drop_duplicates()
        
        # Convert data in numpy
        xoxidey=np.array(data,dtype=np.float64)
        
        # Values of the property in an array y
        self.y=xoxidey[:,-1]
        
        # Molar fraction (mol %) in an array xoxide and transformation in molar fraction
        self.x=xoxidey[:,:-1]*1.e-2
        
        print('Before cleaning, the number of data are ',np.size(self.y))
        
        # Constitution of the list of oxides with the removal of the first blank character
        self.oxide=np.array(data.columns.values,dtype=str)[:-1]
        self.noxide=np.size(self.oxide)
        
        # Removal of oxide not enough represented in the database
        listoxide=[]
        for i in range(self.noxide):
            # Remove of the blank character at the beginning of the string
            self.oxide[i]=self.oxide[i][1:]
            # Determination of the fraction of glasses
            oxideproportion=np.size(np.argwhere(self.x[:,i]!=0.))/np.size(self.y)
            if (oxideproportion<minoxidefraction):
                listoxide=np.append(listoxide,i)
                print('Oxide ',oxide[i],' is removed of the composition.')
            #end if
        #end for
        listoxide=np.int64(listoxide)
        self.oxide=np.delete(self.oxide,listoxide)
        self.x=np.delete(self.x,listoxide,axis=1)
        self.noxide=np.size(self.oxide)
        print('Number of oxides after removal of minor representation is ',self.noxide)
        
        # Data cleaning
        self.thresholdxtotal(xtotal)
        
        # Data filtering to keep the most representatif data of property y
        self.FilteringProperty(probamin,probamax,Plot)
        
        # Data filtering to keep the most representatif data of oxides
        if (filteringoxide):
            for i in range(np.size(oxide)):
                self.FilteringOxide(probamin,probamax,i,Plot)
            #end for
        #end if
        
        # Removal of minor oxide or not enough represented
        listoxide=[]
        for i in range(self.noxide):
            xmax=np.max(self.x[:,i])
            print('For ',self.oxide[i],', xmax=',xmax)
            
            # Determination of the fraction of glasse
            glasswithoxide=np.size(np.argwhere(self.x[:,i]!=0.))
            oxideproportion=glasswithoxide/np.size(self.y)
            if (xmax>0.):
                if (xmax<xminor):
                    listoxide=np.append(listoxide,i)
                #end if
            else:
                listoxide=np.append(listoxide,i)
            #end if
        #end for
        listoxide=np.int64(listoxide)
        
        # Determination of the new list of oxide and the new composition
        self.oxide=np.delete(self.oxide,listoxide)
        self.x=np.delete(self.x,listoxide,axis=1)
        self.info()
    # end datacleaning

    # ---------------
    # thresholdxtotal
    # ---------------

    def thresholdxtotal(self,xtotal):
        """
        Cleaning of the database keeping only the composition for which the sum of
        molar fractions is strictly greater than xtotal and less than 1.
    
        Parameters
        ----------
        xtotal : Float
        Bound admited in the sum of oxides for each glass sample.

        """
        Ndata=np.size(self.y)
        Noxide=np.size(self.x,1)
        xoxidenew=[]
        ynew=[]
        for i in range(Ndata):
            if ((np.sum(self.x[i,:])>=xtotal) and (np.sum(self.x[i,:])<=1.0) and (self.y[i]>0.)):
                ynew=np.append(ynew,self.y[i])
                xoxidenew=np.append(xoxidenew,self.x[i,:],axis=0)
            #end if
        #end for
        print('Numbers of data after x>xtotal: ',np.size(ynew))
            
        xoxidenew=np.reshape(xoxidenew,(-1,Noxide))

        # Copy in the x and y of the data set
        self.x=np.copy(xoxidenew)
        self.y=np.copy(ynew)
    #end thresholdxtotal

    # -----------------
    # FilteringProperty
    # -----------------

    def FilteringProperty(self,probamin,probamax,Plot):
        """
        Cleaning of database by keeping glass samples for which the probability of
        occurrence is in tha range [probamin,probamax].
        
        Parameters
        ----------
        xoxide : Array of 2 dimensions of float
        Composition in molar fraction of glasses.
        y : Array of float
        Property of glasses with composition given by xoxide.
        probamin : Float
        Minimum of the probability of the property y.
        probamax : Float
        Maximum of the probability of the property y.
        Plot : Boolean
        Parameter giving the choice to plot the pdf of the property.
        
        Returns
        -------
        xoxidenew : Array of 2 dimensions
        Composition in molar fraction of glasses.
        ynew : Array of float
        Property of the set of data.
        
        """
    
        # Determination of histogram
        # --------------------------
    
        N=np.size(self.y)
        Noxide=np.size(self.x,1)
        Nbins=int(np.sqrt(N))

        # Determination of the probability density function
        # -------------------------------------------------
    
        pdfproperty,bin_edges=np.histogram(self.y,bins=Nbins,range=(np.min(self.y),np.max(self.y)),\
                                           density=True)
        binsize=bin_edges[1]-bin_edges[0]
        mid_points=bin_edges[:-1]+0.5*binsize

        if (Plot):
            plt.figure()
            plt.bar(mid_points,pdfproperty,width=binsize,color='k',edgecolor='k',\
                    linewidth=1,fill=False,label='Initial pdf')
        #end if
        
        # Determination of the probability function
        # -----------------------------------------
        
        P=np.zeros(Nbins)
        P[0]=pdfproperty[0]*binsize
        for i in range(1,Nbins):
            P[i]+=P[i-1]+pdfproperty[i]*binsize
        #end for

        # Searching of minval for which P<probamin
        # ----------------------------------------
        
        iprobamin=np.max(np.argwhere(P<probamin))
        minval=mid_points[iprobamin]
        print('minval of y: ',minval)
    
        # Searching of maxval for which P>probamax
        # ----------------------------------------
    
        iprobamax=np.min(np.argwhere(P>probamax))
        maxval=mid_points[iprobamax]
        print('maxval of y: ',maxval)
    
        xoxidenew=[]
        ynew=[]
        for i in range(N):
            if ((self.y[i]>=minval)and(self.y[i]<=maxval)):
                ynew=np.append(ynew,self.y[i])
                xoxidenew=np.append(xoxidenew,self.x[i,:],axis=0)
            #end if
        #end for
        xoxidenew=np.reshape(xoxidenew,(-1,Noxide))
        print('Numbers of data after filtering: ',np.size(ynew))
    
        if (Plot):
            N=np.size(ynew)
            Nbins=int(np.sqrt(N))
            pdfproperty,bin_edges=np.histogram(ynew,bins=Nbins,\
                                               range=(np.min(ynew),np.max(ynew)),density=True)
            binsize=bin_edges[1]-bin_edges[0]
            mid_points=bin_edges[:-1]+0.5*binsize
            plt.figure()
            plt.bar(mid_points,pdfproperty,width=binsize,color='b',edgecolor='b',\
                    linewidth=1,fill=False,label='final pdf')
            plt.xlabel(self.nameproperty,fontsize=14)
            plt.ylabel(r'PDF of '+self.nameproperty,fontsize=14)
            plt.legend(loc=0,fontsize=12)
            plt.savefig('pdf'+self.nameproperty+'.png',dpi=300,bbox_inches='tight')
            plt.show()
        #end if

        # Save in x and y the new list of data set
        self.x=np.copy(xoxidenew)
        self.y=np.copy(ynew)
    #end FilteringProperty

    # --------------
    # FilteringOxide
    # --------------

    def FilteringOxide(self,probamin,probamax,ioxide,Plot):
        """
        Cleaning of database by keeping glass samples for which the probability of
        occurrence of oxide ioxide is in the range [probamin,probamax].
    
        Parameters
        ----------
        xoxide : Array of 2 dimensions of float
        Composition in molar fraction of glasses.
        y : Array of float
        Property of glasses with composition given by xoxide.
        probamin : Float
        Minimum of the probability of the oxide xoxide[:,ioxide].
        probamax : Float
        Maximum of the probability of the oxide xoxide[:,ioxide].
        ioxide : Int
        Index of the oxide in xoxide.
        Plot : Boolean
        Parameter giving the choice to plot the pdf of the property.
        
        Returns
        -------
        xoxidenew : Array of 2 dimensions
        Composition in molar fraction of glasses.
        ynew : Array of float
        Property of the set of data.
        
        """
        # Nb de donnees
        N=np.size(self.y)
    
        # Nb. d'oxides
        Noxide=np.size(self.x,1)
    
        # Determination of the probability density function of oxide ioxide
        # -----------------------------------------------------------------
    
        listarg=np.argwhere(self.x[:,ioxide]!=0.)
        nozerovalue=self.x[listarg,ioxide]
        minxoxide=np.min(nozerovalue)
        maxxoxide=np.max(nozerovalue)
        Nbins=int(np.sqrt(np.size(nozerovalue)))
        print(minxoxide,maxxoxide,Nbins)
        pdfxoxide,bin_edges=np.histogram(nozerovalue,bins=Nbins,range=(minxoxide,maxxoxide),\
                                         density=True)
        binsize=bin_edges[1]-bin_edges[0]
        mid_points=bin_edges[:-1]+0.5*binsize
        
        if Plot:
            plt.figure()
            plt.bar(mid_points,pdfxoxide,width=binsize,color='k',edgecolor='k',\
                    linewidth=1,fill=False,label='Initial pdf')
        #end if
        
        # Determination of the probability function
        # -----------------------------------------
        
        P=np.zeros(Nbins)
        P[0]=pdfxoxide[0]*binsize
        for i in range(1,Nbins):
            P[i]+=P[i-1]+pdfxoxide[i]*binsize
        #end for
        
        # Searching of minval for which P<probamin
        # ----------------------------------------
        if (P[0]>probamin):
            iprobamin=0
        else:
            iprobamin=np.max(np.argwhere(P<probamin))
        #end if
        minval=mid_points[iprobamin]
        print('minval of x[',ioxide,']: ',minval)
    
        # Searching of maxval for which P>probamax
        # ----------------------------------------
        
        iprobamax=np.min(np.argwhere(P>probamax))
        maxval=mid_points[iprobamax]
        print('maxval of x[',ioxide,']: ',maxval)

        # Filtering of the data set
        # -------------------------
        
        xoxidenew=[]
        ynew=[]
        for i in range(N):
            if ((self.x[i,ioxide]>=minval) and (self.x[i,ioxide]<=maxval) or self.x[i,ioxide]==0.):
                ynew=np.append(ynew,self.y[i])
                xoxidenew=np.append(xoxidenew,self.x[i,:],axis=0)
            #end if
        #end for
        xoxidenew=np.reshape(xoxidenew,(-1,Noxide))
        print('Numbers of data after filtering: ',np.size(ynew))
    
        if (Plot):
            listarg=np.argwhere(xoxidenew[:,ioxide]!=0.)
            nozerovalue=xoxide[listarg,ioxide]
            minxoxide=np.min(nozerovalue)
            maxxoxide=np.max(nozerovalue)
            Nbins=int(np.sqrt(np.size(nozerovalue)))
            if (Nbins>1):
                pdfxoxide,bin_edges=np.histogram(nozerovalue,bins=Nbins,range=(minxoxide,maxxoxide),\
                                                 density=True)
                binsize=bin_edges[1]-bin_edges[0]
                mid_points=bin_edges[:-1]+0.5*binsize
                plt.bar(mid_points,pdfxoxide,width=binsize,color='b',edgecolor='b',\
                        linewidth=1,fill=False,label='final pdf')
            #end if
            plt.show()
        #end if

        # Saving of the new dataset:
        self.x=np.copy(xoxidenew)
        self.y=np.copy(ynew)
    #end FilteringOxide

    # --------
    # pdfoxide
    # --------
    
    def pdfoxide(self,ioxide,density,Plot,filename):
        """
        Determination of the pdf of the oxide[ioxide].
        
        Parameters
        ----------
        ioxide : Integer
        index of the oxide for which the pdf is determined.
        density : boolean
        Variable to specify if the pdf is normalized to one.
        Plot : Boolean
        If Plot=True, the pdf is plotted.
        filename : string
        Name of the figure with the pdf.
        
        Returns
        -------
        None.
        """
        
        # Determination of the probability density function of oxide ioxide
        # -----------------------------------------------------------------
        
        listarg=np.argwhere(self.x[:,ioxide]!=0.)
        nozerovalue=self.x[listarg,ioxide]
        minxoxide=np.min(nozerovalue)
        maxxoxide=np.max(nozerovalue)
        Nbins=int(np.sqrt(np.size(nozerovalue)))
        print('For oxide ',self.oxide[ioxide],' ',minxoxide,maxxoxide,np.size(nozerovalue))
        pdfxoxide,bin_edges=np.histogram(nozerovalue,bins=Nbins,range=(minxoxide,maxxoxide),\
                                         density=density)
        binsize=bin_edges[1]-bin_edges[0]
        mid_points=bin_edges[:-1]+0.5*binsize
    
        if (Plot):
            plt.figure()
            plt.bar(mid_points,pdfxoxide,width=binsize,color='k',edgecolor='k',\
                    linewidth=1,fill=False,label=self.oxide[ioxide])
            plt.xlabel('Molar fraction of '+self.oxide[ioxide],fontsize=12)
            if (density):
                plt.ylabel('propa. density funct.',fontsize=12)
            else:
                plt.ylabel('histogram',fontsize=12)
            #end if
            plt.legend(loc=0)
            plt.savefig(filename,dpi=300,bbox_inches='tight')
        #end if
    # end pdfoxide

    # familyrandomcomposition
    # -----------------------
    
    def familyrandomcomposition(self,Nglass,I,J,xmol0,dx0):
        """
        Determination of random composition of Nglass composition centered on 
        a composition constituted of a list I of a sub-set oxides with centered
        composition xmol0.
        
        Parameters
        ----------
        Nglass : Integer
        Number of glass composition.
        I : Array of integers
        Sub-set of indexes correponding of a nominal composition of glasses.
        J : Array of integer corresponding to the complementary of I.
        Sub-set of indexes correponding of the complementary oxides.
        xmol0 : Array of float
        Nominal composition of glass.
        dx0 : Float
        Initial range of oxide fraction.
        Moxide : Array of float
        Molar mass of the oxides.

        Returns
        -------
        xglass : Array of float, dimension (Nglass,Noxide)
        Molar fraction of oxides for the Nglass compositions.
        Mmolar : Array of float of Nglass
        Molar mass of each glass composition.
        """

        # Determination of molar mass of oxides
        # -------------------------------------
        if (self.Moxide is None):
            self.oxidemolarmass()
        #end if
        
        xglass=np.zeros((Nglass,np.size(xmol0)))
        Mmolar=np.zeros(Nglass)

        for n in range(Nglass):
            # Determination of glass composition
            # ----------------------------------
            x=0.
            dx=dx0
            for i in range(np.size(I)-1):
                dx=np.min([1.-x-xmol0[I[i]],dx])
                if (xmol0[I[i]]!=0.):
                    xglass[n,I[i]]=xmol0[I[i]]+dx*np.random.uniform(-1.,1.)
                else:
                    xglass[n,I[i]]=dx*np.random.rand()
                #endif
                x+=xglass[n,I[i]]
            #end for
            
            random.shuffle(J)
            for i in range(np.size(J)):
                dx=np.min([1.-x,dx])
                xglass[n,J[i]]=dx*np.random.rand()
                x+=xglass[n,J[i]]
            #endfor
            xglass[n,:]=xglass[n,:]/np.sum(xglass[n,:])
        
            # Molar mass of glasses
            # ---------------------
            Mmolar[n]=np.sum(xglass[n,:]*self.Moxide)
        #end for
    
        # Return of xglass and Mmolar
        return xglass,Mmolar
    #end familyrandomcomposition

    # -----------------
    # randomcomposition
    # -----------------
    
    def randomcomposition(self,Nglass,xmax):
        """
        Determination of random composition of Nglass composition.
        
        Parameters
        ----------
        Nglass : Integer
        Number of glass composition.
        xmax : Array of float
        Maximum value of each oxide found in the data-set invelved in the ANN
        model.

        Returns
        -------
        xglass : Array of float, dimension (Nglass,Noxide)
        Molar fraction of oxides for the Nglass compositions.
        Mmolar : Array of float of Nglass
        Molar mass of each glass composition.

        """
        
        # Determination of molar mass of oxides
        # -------------------------------------
        if (self.Moxide is None):
            self.oxidemolarmass()
        #end if
        
        xglass=np.zeros((Nglass,self.noxide))
        Mmolar=np.zeros(Nglass)
        I=np.array(range(self.noxide-1))
        for n in range(Nglass):
            random.shuffle(I)
            x=0.
            for i in range(np.size(I)):
                xglass[n,I[i]]=np.random.rand()*np.min([1.-x,xmax[I[i]]])
                x+=xglass[n,I[i]]
            #end for
            xglass[n,:]=xglass[n,:]/np.sum(xglass[n,:])
        
            # Molar mass of glasses
            # ---------------------
            Mmolar[n]=np.sum(xglass[n,:]*self.Moxide)
        #end for
    
        # Return of xglass and Mmolar
        return xglass,Mmolar
    #end randomcomposition

    def better_random_composition(self,Nglass,xmin,xmax):
        if (self.Moxide is None):
            self.oxidemolarmass()
        #end if
        xmax = xmax.copy()
        xmax[-1] = 0
        xglass=np.zeros((Nglass,self.noxide))
        Mmolar=np.zeros(Nglass)
        for n in range(Nglass):
            power = np.random.random() * 100
            weights = np.random.random(self.noxide)
            weights **= power
            deltas = np.array(xmax) - np.array(xmin)
            total = np.sum(xmin)
            xglass[n, :] = xmin
            to_add = deltas * weights
            while to_add < (1 - total):
                weights **= 0.5
                to_add = deltas * weights
            to_add = (to_add / np.sum(to_add)) * (1 - total)
            xglass[n,:] += to_add
            # Molar mass of glasses
            # ---------------------
            Mmolar[n]=np.sum(xglass[n,:]*self.Moxide)
        #end for
    
        # Return of xglass and Mmolar
        return xglass,Mmolar
    #end randomcomposition


    # ------------
    # GlassDensity
    # ------------
    
    def GlassDensity(self,nnmodel,oxide,x):
        """
        Determination of the glass density from the molar volume.
        
        Parameters
        ----------
    
        nnmodel : ANN model on Vt.
        oxide : Array of string
        Oxide names.
        x : Array of two dimensions (N,Noxide)
        molar fraction of glass composition.

        Returns
        -------
    
        rho : Array of float
        Glass density.

        """
        # Copy of the glass composition in the format of data-set of db
        xdb=self.xglasstoxdb(oxide,x)

        if (self.Moxide is None):
            self.oxidemolarmass()
        #end if

        # Determination of the molar mass
        Mmolar=np.zeros(np.size(x,0))
        for i in range(np.size(x,0)):
            Mmolar[i]=np.sum(xdb[i,:]*self.Moxide)
        #end for
    
        # Determination of the glass density
        rho=Mmolar/self.physicaly(nnmodel.model.predict(xdb).transpose()[0,:])
    
        # Return to the density
        return rho
    # end GlassDensity

    # ------------
    # YoungModulus
    # ------------
    
    def YoungModulus(self,nnmodel,datadisso,oxide,x):
        """
        Determination of the Young's modulus as a function of a composition of
        glass with the list of oxide and molar composition x.

        Parameters
        ----------
        db : Data-set used to determine E
        nnmodel : Nearal-network model to predict E
        datadisso : dataframe
        Data gathered the dissosiation energy of oxides.
        oxide : Array of strings
        Names of oxides.
        x : Array of two dimensions (N,Noxide)
        molar fraction of glass composition.

        Returns
        -------
        Array of N floats
        Young's modulus.
        """
        # Size of the data-set
        # --------------------
        N=np.size(x,0)
    
        # Determination of the atomic packing factor
        # ------------------------------------------
        xdb=self.xglasstoxdb(oxide,x)
        VtVm=self.physicaly(nnmodel.model.predict(xdb).transpose()[0,:])

        # Determination of the dissociation energy of glasses
        # ---------------------------------------------------
        G=np.zeros(N)
        for i in range(N):
            G[i]=np.sum(datadisso['G'].values*xdb[i,:])
        #end for

        # Determination of E and return of the result
        # -------------------------------------------
        return 2.*VtVm*G
    #end YoungModulus

# end class GlassData

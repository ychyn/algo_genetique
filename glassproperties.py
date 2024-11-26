# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 07:55 2024

Functions to determine various properties of glass or glass forming liquid.

@author: Franck Pigeonneau (CEMEF/CFL)
"""

import numpy as np
from molmass import Formula

# -----
# muVFT
# -----

def muVFT(T,Amu,Bmu,Tmu):
    """
    
    Determination of the dynamic viscosity of glass forming liquid using VFT's law given
    by

    mu=Amu*exp(Bmu/(T-Tmu)),

    Paremeters
    ----------
    T : Float
        Temperature in (K)
    
    Amu : Float
          Parameter of VFT's (Pa.s)
    Bmu : Float
          Parameter of VFT's law (K)
    Tmu : Float
          Parameter of VFT's law (K)

    Returns
    -------
    Dynamic viscosity in Pa.s

    """

    return Amu*np.exp(Bmu/(np.float64(T)-Tmu))
#end muVFT

# -----
# Henry
# -----

def Henry(A,B,T):
    """
    Solubility of gas in a glass forming liquid from the Henry's law given by

    L=A*exp(B/T)

    The unit of L is mol/(m^3Pa^beta) with beta the exponent in pressure depenence:

    C=L*P^beta.

    Parameters
    ----------
    A : Float
        Parameter of Henry's in mol/(m^3Pa^beta).
    B : Float
        Parameter of Henry's in K
    T : Float
        Temperature in K.

    Returns
    -------
    Gas solubility of a gas species.

    """
    T=np.float64(T)
    return A*np.exp(B/T)
#end Henry

# -----------
# diffusivity
# -----------

def diffusivity(A,B,T):
    """
    Diffusivity of gas in a glass forming liquid from the law given by

    D=A*exp(-B/T)

    The unit of D is m^2/s.

    Parameters
    ----------
    A : Float
        Parameter of D in m^2/s.
    B : Float
        Parameter of D in K
    T : Float
        Temperature in K.

    Returns
    -------
    Diffusivity of a gas species.

    """

    T=np.float64(T)
    return A*np.exp(-B/T)
#end diffusivity

# ----
# Kequi
# -----

def Kequi(A,B,T):
    """
    Computation of the equilibrium constant of a chemical reaction of 
    oxidation-reduction in glass forming liquid.

    Parameters:
    -----------

    A : Float
        Exponential factor corresponding to $-\Delta S/R$ without unit.
    B : Float
        Exponential factor corresponding to $-\Delta H/R$ in K.
    T : Float
        Temperature in K.

    Returns
    -------
    Equilibrium constant.

    """
    T = np.float64(T)
    return np.exp(A-B/T)
#end Kequi

# ---------------
# VFTcoefficients
# ---------------

def VFTcoefficients(Tm,Ts,Tg):
    """
    This function determines the three coefficients, A, B and T0 of the VFT's law
    
    log(eta)=A+B/(T-T0)
    
    The temperatures are given in Celsius or in Kelvin. The coefficient A is determined to
    have the viscosity in Poisseuille (Pa.s).

    Parameters
    ----------
    Tm : Float
        Temperature for which the log(eta)=1.
    Ts : Float
        Temperature for which the log(eta)=6.65.
    Tg : Float
        Temperature for which the log(eta)=12.

    Returns
    -------
    A, B and T0.

    """
    
    # Computation of T0
    # -----------------
    c=5.65/11.
    T0=((Tm-Ts)*Tg-c*Ts*(Tm-Tg))/(Tm-Ts-c*(Tm-Tg))
    
    # Computation of B
    # ----------------
    B=11.*(Tg-T0)*(Tm-T0)/(Tm-Tg)
    
    # Computation of A
    # ----------------
    A=1.-B/(Tm-T0)
    
    return A,B,T0
#end VFTcoefficients

# -----
# Tsoft
# -----

def Tsoft(Amu,Bmu,Tmu):
    """
    Computation of the Littletown temperature based on the VFT's law of a
    glass forming liquid given by

    mu=Amu*exp(Bmu/(T-Tmu)).

    Parameters
    ----------
    Amu : Float
          Parameter of VFT's (Pa.s)
    Bmu : Float
          Parameter of VFT's law (K)
    Tmu : Float
          Parameter of VFT's law (K)

    Returns
    -------
    Softening temperature in K.

    """
    return Tmu+Bmu/np.log(10.**6.65/Amu)
#end Tsoft

# ------
# Tglass
# ------

def Tglass(Amu,Bmu,Tmu):
    """
    Computation of the glass transition temperature based on the VFT's law of a
    glass forming liquid given by

    mu=Amu*exp(Bmu/(T-Tmu)).

    Parameters
    ----------
    Amu : Float
          Parameter of VFT's (Pa.s)
    Bmu : Float
          Parameter of VFT's law (K)
    Tmu : Float
          Parameter of VFT's law (K)

    Returns
    -------
    Glass transition temperature in K.

    """
    
    return Tmu+Bmu/np.log(10.**12/Amu)
#end Tglass

# ---------
# MolarMass
# ---------

def MolarMass(name):
    """
    Molar mass of a chemical molecule given by name in kg/mol.

    Parameters
    ----------
    
    name : string corresponding to the name of a chemical component.
    
    Returns
    -------
    
    Float equal to the molar mass in kg/mol.
    
    """
    
    f=Formula(name)
    
    # return of the molar mass in kg/mol
    return f.mass*1.e-3
#end MolarMass

# ------------
# PoissonRatio
# ------------

def PoissonRatio(dbE,nnmodelEsG,db,nnmodel,oxide,x):
    """

    Determination of the Poisson's ratio from the Makishima-Mackensie's model.

    Parameters:
    -----------
    dbE: Dataset of the Young's modulus
    nnmodelEsG: Artificial neural networks on the atomic packing factor.
    db: Dataset of the Poisson's ratio.
    nnmodel: Artificial neural networks on alpha factor.
    oxide: array of strings
           List of oxide names.
    x: Array of molar fraction.

    Returns:
    --------
    Poisson's ratio.
    """

    # Determination of the atomic packing factor from the ANN model created from the Young's modulus
    
    xdb=dbE.xglasstoxdb(oxide,x)
    Vt=dbE.physicaly(nnmodelEsG.model.predict(xdb).transpose()[0,:])

    # Determination of alpha
    # ----------------------
    
    xdb=db.xglasstoxdb(oxide,x)
    alpha=db.physicaly(nnmodel.model.predict(xdb).transpose()[0,:])

    # Return to the Poisson's ratio
    # -----------------------------
    
    return 0.5-1./(6.*alpha*Vt)
#end PoissonRatio

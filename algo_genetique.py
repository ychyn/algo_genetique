import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from random import randint

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

# # Algo genetique

# ## Variables utiles

labels = dbrho.oxide
N_oxides = len(labels)
available_mat = ['SiO2', 'Al2O3', 'MgO', 'CaO', 'Na2O', 'K2O','ZnO', 'TiO2']
prop_label = ['rho','E','Tg','Tmelt']
columns = list(labels)+prop_label+['F']

#Contraintes

xmaxt=np.array([dbrho.xmax,dbE.xmax,dbTannealing.xmax,np.append(dbTmelt.xmax,1.),dbTliq.xmax])
xmax=np.zeros(dbrho.noxide)
for i in range(dbrho.noxide):
    if dbrho.oxide[i] in available_mat:
        xmax[i]=np.min(xmaxt[:,i])

xmin = np.zeros(dbrho.noxide)
xmin[list(dbrho.oxide).index('SiO2')] = 0.35
xmin[list(dbrho.oxide).index('Na2O')] = 0.1

#Parametres fitness function

weight=[0,0.3,0,0.7]
minimize=[True,False,False,True]

N_generations = 10
N_population = 1000

survivor_rate = 0.5
parent_rate = 0.2
child_rate = 0.4
mutation_rate = 0
immigration_rate = 1 - survivor_rate - child_rate

#population = np.zeros((N_population,len(labels) + len(prop_label) + 1))

N_parents = int(parent_rate * N_population)
N_childs = int(child_rate * N_population)
N_mutants = int(0.1 * N_population)

epsilon = 0.05

# ## Creation de generations

def prop_calculation(composition):
    rho=dbrho.GlassDensity(nnmolvol,dbrho.oxide,composition)
    E=dbE.YoungModulus(nnmodelEsG,datadisso,dbE.oxide,composition)
    Tg=dbTannealing.physicaly(nnTannealing.model.predict(composition).transpose()[0,:])
    Tmelt=dbTmelt.physicaly(nnTmelt.model.predict(composition[:,:-1]).transpose()[0,:])
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

# Trie la population par F decroissant et renvoie cette population triée avec une nuovelle colonne qui represente 
# le fitness de chaque composition.

def stack_by_f(population,properties,F):
    population_info = np.column_stack((population,properties,F))
    sorted_arr = population_info[population_info[:, -1].argsort()][::-1]
    return sorted_arr

def init_properties(population):
    prop = prop_calculation(population)
    normalized_prop = normalize(prop)
    F = fitness_func(normalized_prop,weight,minimize)
    sorted_arr = stack_by_f(population, prop, F)
    return sorted_arr

def compute_properties(generation):
    population_sorted = init_properties(generation[:, :20])
    return population_sorted

def init_pop(N_population):
    population,_=dbrho.better_random_composition(N_population,xmin,xmax)
    population = init_properties(population)
    return population

old_generation = init_pop(N_population)

def population_selection(generation):
    survivors = generation[:int(N_population*survivor_rate)]
    parents = generation[:N_parents]
    #to_be_mutated = sorted_population[int(N_population*survivor_rate):int(N_population*survivor_rate)+int(N_population*mutation_rate)]
    return survivors,parents

def crossover (parents) :
    #Dans parents chaque individu est représenté par 20 premiers floats et le dernier est la valeur de fitness
    childs = np.array([[0.] * len(parents[0])] * N_childs)
    for i in range (N_childs) :
        i1 = randint(0, N_parents-1)
        i2 = randint(0, N_parents-2)
        if i2 == i1 : #problème si deux fois le même parent !!
            i2 += 1
        w1 = parents[i1, -1] / (parents[i1, -1] + parents[i2, -1]) #poids du parent 1
        w2 = parents[i2, -1] / (parents[i1, -1] + parents[i2, -1]) #poids du parent 2
        childs[i] = (w1 * parents[i1] + w2 * parents[i2]) #moyenne pondérée
    return (childs)

def mutation (mutants) :
    #les mutants sont les meilleures compositions entre 40% et 50%
    for j in range (N_mutants) :
        iplus = randint(0, 19) #choix de l'oxyde qui gagne epsilon
        imoins = randint(0, 19) #choix de l'oxyde qui perd epsilon
        if imoins == iplus : #problème si deux fois le même oxyde !!
            imoins = (1 + imoins)%19
        mutantPlus = mutants[j, iplus]
        mutantMoins = mutants[j, imoins]
        if (mutantMoins > epsilon + xmin[j]) and (mutantPlus < xmax[j] - epsilon) :
            mutants[j, iplus] = mutantPlus + epsilon
            mutants[j, imoins] = mutantMoins - epsilon
    return (mutants)

def new_generation(old_generation):
    survivors,parents = population_selection(old_generation)
    child = crossover(parents)
    immigrants = init_pop(N_population - (len(survivors) + len(child)))
    new_population = np.vstack((np.vstack((survivors,child)),immigrants))
    new_population = compute_properties(new_population)
    return new_population

def evolution(generation,N):
    for _ in range(N):
        generation = new_generation(generation)
    return generation

initial_pop = init_pop(N_population)
compositions = evolution(initial_pop,N_generations)

df = pd.DataFrame(compositions,columns=columns)

df.to_csv('generation_final.csv')

df
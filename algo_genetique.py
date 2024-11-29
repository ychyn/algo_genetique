import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from random import randint

# Modules with own classes
# ------------------------

from glassdata import GlassData
from network import NeuralNetwork


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

def default_population_selection(generation):
    survivors = generation[:int(N_population*survivor_rate)]
    parents = generation[:N_parents]
    #to_be_mutated = sorted_population[int(N_population*survivor_rate):int(N_population*survivor_rate)+int(N_population*mutation_rate)]
    return survivors,parents

def default_crossover (parents) :
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

def default_mutation (mutants, xmin, xmax) :
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

class EvolutionModel():

    def __init__(self):
        self.dbrho = None
        self.nnmolvol = None
        self.dbE = None
        self.dbTannealing = None
        self.nnmodelEsG = None
        self.dbTmelt = None
        self.nnTmelt = None
        self.dbTliq = None
        self.nnTliq = None
        self.datadisso = None
        self.generation = None
        self.xmin = None
        self.xmax = None

        # Functions
        self.crossover = None
        self.mutation = None
        self.population_selection = None

    def load(self):

        # ---------------------------------------
        # Data-set on rho and ANN on molar volume
        # ---------------------------------------
        
        # Dataset of rho
        filedbrho='DataBase/rho20oxides.csv'
        self.dbrho=GlassData(filedbrho)
        self.dbrho.info()
        self.dbrho.bounds()

        # Determination of the molar volume
        self.dbrho.oxidemolarmass()
        self.dbrho.molarmass()
        self.dbrho.y=self.dbrho.MolarM/self.dbrho.y
        self.dbrho.normalize_y()

        # Loading of the ANN model
        arch=[20,20,20]
        self.nnmolvol=NeuralNetwork(self.dbrho.noxide,arch,'gelu','linear')
        self.nnmolvol.compile(3.e-4)
        self.nnmolvol.ArchName(arch)
        self.nnmolvol.load('Models/nnmolarvol'+self.nnmolvol.namearch+'.h5')
        self.nnmolvol.info()

        # ------------------------------------------------
        # Data-set on Young's modulus and ANN on Vt=E/(2G)
        # ------------------------------------------------

        filedbE='DataBase/E20oxides.csv'
        self.dbE=GlassData(filedbE)
        self.dbE.info()
        self.dbE.bounds()

        # ------------------------------
        # Loading of dissociation energy
        # ------------------------------

        self.datadisso=pd.read_csv('dissociationenergy.csv')
        G=np.zeros(self.dbE.nsample)
        for i in range(self.dbE.nsample):
            G[i]=np.sum(self.datadisso['G'].values*self.dbE.x[i,:])
        #end for

        # Determination of E/G and normalization
        self.dbE.y=self.dbE.y/(2.*G)
        self.dbE.normalize_y()

        # ------------------------------
        # Loading of the ANN model on Vt
        # ------------------------------

        arch=[20,20,20]
        self.nnmodelEsG=NeuralNetwork(self.dbE.noxide,arch,'gelu','linear')
        self.nnmodelEsG.compile(1.e-4)
        self.nnmodelEsG.ArchName(arch)
        self.nnmodelEsG.load('Models/nnEsG'+self.nnmodelEsG.namearch+'.h5')
        self.nnmodelEsG.info()

        # ---------------------------------------
        # Data-set on Tannealing=Tg and ANN model
        # ---------------------------------------

        # Data-set of Tannealing
        filedbTannealing='DataBase/Tannealing20oxides.csv'
        self.dbTannealing=GlassData(filedbTannealing)
        self.dbTannealing.bounds()
        self.dbTannealing.normalize_y()

        # ANN model on Tannealing
        # -----------------------
        arch=[20,20,20]
        self.nnTannealing=NeuralNetwork(self.dbTannealing.noxide,arch,'gelu','linear')
        self.nnTannealing.compile(3.e-4)
        self.nnTannealing.ArchName(arch)
        self.nnTannealing.info()
        self.nnTannealing.load('Models/nn'+self.dbTannealing.nameproperty+self.nnTannealing.namearch+'.h5')

        # Data-set on Tmelt
        # -----------------

        filedbTmelt='DataBase/Tmelt19oxides.csv'
        self.dbTmelt=GlassData(filedbTmelt)
        self.dbTmelt.info()
        self.dbTmelt.bounds()
        self.dbTmelt.normalize_y()

        # ANN model on Tmelt
        # ------------------
        arch=[20,20,20]
        self.nnTmelt=NeuralNetwork(self.dbTmelt.noxide,arch,'gelu','linear')
        self.nnTmelt.compile(3.e-4)
        self.nnTmelt.ArchName(arch)
        self.nnTmelt.load('Models/nn'+self.dbTmelt.nameproperty+self.nnTmelt.namearch+'.h5')
        self.nnTmelt.info()

        # ------------------------------
        # Data-set on Tliq and ANN model
        # ------------------------------

        filedbTliq='DataBase\Tsoft20oxides.csv'
        self.dbTliq=GlassData(filedbTliq)
        self.dbTliq.info()
        self.dbTliq.bounds()
        self.dbTliq.normalize_y()

        # ANN model on Tliq
        # -----------------

        arch=[32,32,32,32]
        self.nnTliq=NeuralNetwork(self.dbTliq.noxide,arch,'gelu','linear')
        self.nnTliq.compile(3.e-4)
        self.nnTliq.ArchName(arch)
        #modelfile='Models\nn'+dbTliq.nameproperty+nnTliq.namearch+'.h5'
        modelfile='Models/nnTsoft3c20.h5'
        self.nnTliq.load(modelfile)
        self.nnTliq.info()

        self.xmin = np.zeros(self.dbrho.noxide)
        self.xmax = np.ones(self.dbrho.noxide)

        # Function

        self.population_selection = default_population_selection
        self.crossover = default_crossover
        self.mutation = default_mutation

    def prop_calculation(self, composition):
        rho=self.dbrho.GlassDensity(self.nnmolvol,self.dbrho.oxide,composition)
        E=self.dbE.YoungModulus(self.nnmodelEsG,self.datadisso,self.dbE.oxide,composition)
        Tg=self.dbTannealing.physicaly(self.nnTannealing.model.predict(composition).transpose()[0,:])
        Tmelt=self.dbTmelt.physicaly(self.nnTmelt.model.predict(composition[:,:-1]).transpose()[0,:])
        return np.vstack((rho,E,Tg,Tmelt)).transpose()

    def normalize(self, prop):
        return (prop - prop.min(axis=0))/(prop.max(axis=0)-prop.min(axis=0))

    #prop est une array avec les proprietes du verre normalisées, weight est le poids qu'on accorde
    #à chacune des proprietes, et minize est une liste de booléens selon qu'on veuille minimiser
    #ou maximiser une certaine variable

    def fitness_func(self, prop_normalized,weight,minimize):
        rating = np.zeros(prop_normalized.shape[0])
        for i in range(len(weight)):
            if minimize[i]:
                rating += (1-prop_normalized[:,i])*weight[i]
            else:
                rating += prop_normalized[:,i]*weight[i]
        return rating

    # Trie la population par F decroissant et renvoie cette population triée avec une nuovelle colonne qui represente 
    # le fitness de chaque composition.

    def stack_by_f(self, population,properties,F):
        population_info = np.column_stack((population,properties,F))
        sorted_arr = population_info[population_info[:, -1].argsort()][::-1]
        return sorted_arr

    def init_properties(self, population):
        prop = self.prop_calculation(population)
        normalized_prop = self.normalize(prop)
        F = self.fitness_func(normalized_prop,weight,minimize)
        sorted_arr = self.stack_by_f(population, prop, F)
        return sorted_arr

    def compute_properties(self, generation):
        population_sorted = self.init_properties(generation[:, :20])
        return population_sorted

    def init_pop(self, N_population):
        population,_=self.dbrho.better_random_composition(N_population,self.xmin,self.xmax)
        population = self.init_properties(population)
        self.generation = population
        return population

    def next_generation(self, old_generation):
        survivors,parents = self.population_selection(old_generation)
        child = self.crossover(parents)
        immigrants = self.init_pop(N_population - (len(survivors) + len(child)))
        new_population = np.vstack((np.vstack((survivors,child)),immigrants))
        new_population = self.compute_properties(new_population)
        return new_population

    def evolution(self,N):
        for _ in range(N):
            self.generation = self.next_generation(self.generation)
        return self.generation

# ------------------------------------------
# Determination of the bounds for each oxide
# ------------------------------------------

# # Algo genetique

# ## Variables utiles

data = EvolutionModel()
data.load()

labels = data.dbrho.oxide
N_oxides = len(labels)
available_mat = ['SiO2', 'Al2O3', 'MgO', 'CaO', 'Na2O', 'K2O','ZnO', 'TiO2']
prop_label = ['rho','E','Tg','Tmelt']
columns = list(labels)+prop_label+['F']

#Contraintes

xmaxt=np.array([data.dbrho.xmax,data.dbE.xmax,data.dbTannealing.xmax,np.append(data.dbTmelt.xmax,1.),data.dbTliq.xmax])
xmax=np.zeros(data.dbrho.noxide)
for i in range(data.dbrho.noxide):
    if data.dbrho.oxide[i] in available_mat:
        xmax[i]=np.min(xmaxt[:,i])

xmin = np.zeros(data.dbrho.noxide)
xmin[list(data.dbrho.oxide).index('SiO2')] = 0.35
xmin[list(data.dbrho.oxide).index('Na2O')] = 0.1

data.xmin = xmin
data.xmax = xmax

#population = np.zeros((N_population,len(labels) + len(prop_label) + 1))

N_parents = int(parent_rate * N_population)
N_childs = int(child_rate * N_population)
N_mutants = int(0.1 * N_population)

epsilon = 0.05

# ## Creation de generations

initial_pop = data.init_pop(N_population)
final_pop = data.evolution(N_generations)

df = pd.DataFrame(final_pop,columns=columns)

df.to_csv('generation_final.csv')

df
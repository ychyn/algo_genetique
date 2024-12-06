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

weight=[0,0.35,0,0.65]
minimize=[True,False,False,True]
penalties_onMin = [-np.inf, -np.inf, -np.inf, -np.inf]
penalties_onMax = [np.inf, np.inf, np.inf, np.inf]

N_generations = 10
N_population = 1000

survivor_rate = 0.1
parent_rate = 0.5
child_rate = 0.7
mutation_rate = 0
immigration_rate = 1 - survivor_rate - child_rate

strategies = ['2by2','tournament']
N_tournament = 5
child_strategy = strategies[1]

def parent_choice (parents,strategie) :
    if strategie =='2by2' and N_parents>= 2*N_childs:
        dads = parents[::2]
        moms = parents[1::2]
        return (np.array(moms[:N_childs]),np.array(dads[:N_childs]))    
    if strategie =='tournament':
        dads =[]
        moms =[]
        for i in range(N_childs):
            t = np.array([parents[i,:] for i in np.random.choice(N_parents,N_childs)])
            dad = t[np.argmax(t[:N_tournament,-1])]
            mom = t[np.argmax(t[N_tournament:,-1])]
            dads.append(dad)
            moms.append(mom)
        return (np.array(moms),np.array(dads))
    #Fallback
    t = np.array([parents[i] for i in np.random.choice(N_parents,N_childs*2)])
    return (np.array(t[N_childs:]),np.array(t[:N_childs]))

def default_population_selection(generation):
    survivors = generation[:int(N_population*survivor_rate)]
    dads,moms = parent_choice(generation[:N_parents],child_strategy)
    #to_be_mutated = sorted_population[int(N_population*survivor_rate):int(N_population*survivor_rate)+int(N_population*mutation_rate)]
    return survivors,dads,moms 

fondants = [False, False, False, True, True, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] # diminuent Tm
durability = [True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] #augmentent E
other = [False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False]

def default_crossover (mom, dad) :
    childs = np.zeros((len(mom), len(mom[0])))
    for i in range (N_childs) :
        if mom[i, 21] < dad[i, 21] : #choix du meilleur E
            bestE = dad[i]
        else :
            bestE = mom[i]
        if mom[i, 23] < dad[i, 23] : #choix du meilleur Tm
            bestTm = mom[i]
        else :
            bestTm = dad[i]
        childs[i][fondants] = bestTm[fondants]
        childs[i][durability] = bestE[durability]
        childs[i][other] = (mom[i][other] + dad[i][other])/2
        sum = np.sum(childs[i,1:])
        childs[i,1:] = (1-childs[i,0]) * childs[i,1:] / sum
    return (childs)

N_parents = int(parent_rate * N_population)
N_childs = int(child_rate * N_population)
N_mutants = int(0.1 * N_population)

epsilon = 0.05

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

def bettermutation (mutants, xmin, xmax) :
    for j in range (N_mutants) :
        travel = np.random.rand(20) - 0.5
        travel =travel/np.linalg.norm(travel)*epsilon
        mutants[j,:20] += travel
        for _ in range(5):
            mutants[j,:20] = np.clip(mutants[j,:20],xmin*np.sum(mutants[j,:20]),xmax*np.sum(mutants[j,:20]))
            mutants[j,:20] = mutants[j,:20]/np.sum(mutants[j,:20])
    return (mutants)

class EvolutionModel():

    def __init__(self):
        self.dbrho = None
        self.dbE = None
        self.dbTannealing = None
        self.dbTmelt = None
        self.dbTliq = None
        self.datadisso = None

        self.nnmolvol = None
        self.nnmodelEsG = None
        self.nnTmelt = None
        self.nnTliq = None

        self.xmin = None
        self.xmax = None
        self.prop_min = None
        self.prop_max = None
        self.penalties_onMin_normalized = None
        self.penalties_onMax_normalized = None
        
        self.generation = None

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

        self.prop_min = np.array([min(self.dbrho.y),min(self.dbE.y),min(self.dbTannealing.y),min(self.dbTmelt.y)])
        self.prop_max = np.array([max(self.dbrho.y),max(self.dbE.y),max(self.dbTannealing.y),max(self.dbTmelt.y)])

        self.penalties_onMin_normalized = self.normalize(penalties_onMin)
        self.penalties_onMax_normalized = self.normalize(penalties_onMax)

    def prop_calculation(self, composition):
        rho=self.dbrho.GlassDensity(self.nnmolvol,self.dbrho.oxide,composition)
        E=self.dbE.YoungModulus(self.nnmodelEsG,self.datadisso,self.dbE.oxide,composition)
        Tg=self.dbTannealing.physicaly(self.nnTannealing.model.predict(composition).transpose()[0,:])
        Tmelt=self.dbTmelt.physicaly(self.nnTmelt.model.predict(composition[:,:-1]).transpose()[0,:])
        return np.vstack((rho,E,Tg,Tmelt)).transpose()

    def normalize(self, prop):
        return (prop - self.prop_min)/(self.prop_max - self.prop_min)

    # prop est une array avec les proprietes du verre normalisées, weight est le poids qu'on accorde
    # à chacune des proprietes, et minize est une liste de booléens selon qu'on veuille minimiser
    # ou maximiser une certaine variable

    def fitness(self, property_normalized):
        rating = 0
        #print(penalties_onMin_normalized, property_normalized, penalties_onMax_normalized)
        #print(np.all(penalties_onMin_normalized <= property_normalized) and np.all(penalties_onMax_normalized >= property_normalized))
        if np.all(self.penalties_onMin_normalized <= property_normalized) and np.all(self.penalties_onMax_normalized >= property_normalized):
            for i in range(len(weight)):
                if minimize[i]:
                    rating += (1-property_normalized[i])*weight[i]
                else:
                    rating += property_normalized[i]*weight[i]
        return rating

    def fitness_func(self, prop_normalized):
        return np.apply_along_axis(self.fitness, 1, prop_normalized)

    # Trie la population par F decroissant et renvoie cette population triée avec une nuovelle colonne qui represente 
    # le fitness de chaque composition.

    def stack_by_f(self, population,properties,F):
        population_info = np.column_stack((population,properties,F))
        sorted_arr = population_info[population_info[:, -1].argsort()][::-1]
        return sorted_arr

    def init_properties(self, population):
        prop = self.prop_calculation(population)
        normalized_prop = self.normalize(prop)
        F = self.fitness_func(normalized_prop)
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

    def new_generation(self, old_generation):
        survivors,dads,moms = self.population_selection(old_generation)
        child = bettermutation(self.crossover( dads, moms) , self.xmin, self.xmax)
        immigrants = self.init_pop(N_population - (len(survivors) + len(child)))
        new_population = np.vstack((np.vstack((survivors,child)),immigrants))
        new_population = self.compute_properties(new_population)
        return new_population

    def evolution(self,N):
        for _ in range(N):
            self.generation = self.new_generation(self.generation)
        return self.generation

data = EvolutionModel()
data.load()

available_mat = ['SiO2', 'Al2O3', 'MgO', 'CaO', 'Na2O', 'K2O','ZnO', 'TiO2']

def get_xmax(data, available_mat):
    # Contraintes

    xmaxt=np.array([data.dbrho.xmax,data.dbE.xmax,data.dbTannealing.xmax,np.append(data.dbTmelt.xmax,1.),data.dbTliq.xmax])
    xmax=np.zeros(data.dbrho.noxide)
    for i in range(data.dbrho.noxide):
        if data.dbrho.oxide[i] in available_mat:
            xmax[i]=np.min(xmaxt[:,i])
    
    return xmax

xmax = get_xmax(data, available_mat)

xmin = np.zeros(data.dbrho.noxide)
xmin[list(data.dbrho.oxide).index('SiO2')] = 0.5
xmin[list(data.dbrho.oxide).index('Na2O')] = 0.1

data.xmin = xmin
data.xmax = xmax

# Creation de generations

initial_pop = data.init_pop(N_population)
final_pop = data.evolution(N_generations)

labels = data.dbrho.oxide
prop_label = ['rho','E','Tg','Tmelt']
columns = list(labels)+prop_label+['F']

df = pd.DataFrame(final_pop, columns = columns)

df.to_csv('generation_final.csv')

df
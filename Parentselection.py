from random import randint
import numpy as np

N_population = 1000

N_parents = int(0.1 * N_population)
N_enfants = int(0.4 * N_population)

def Parentcontest (Population,strategie) :
    #Dans parents chaque individu est représenté par 21 floats dont le dernier est la valeur de fitness
    Parents = Population[:N_parents]
    enfants = np.array([[0.] * 21] * N_enfants)
    if strategie =='Contest':
        papas = np.random.choice(Parents,10)
        

    return (enfants)


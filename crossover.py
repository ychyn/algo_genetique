from random import randint
import numpy as np

N_population = 1000

N_parents = int(0.1 * N_population)
N_enfants = int(0.4 * N_population)

def crossover (parents) :
    #Dans parents chaque individu est représenté par 21 floats dont le dernier est la valeur de fitness
    enfants = np.array([[0.] * 21] * N_enfants)
    for i in range (N_enfants) :
        i1 = randint(0, N_parents-1)
        i2 = randint(0, N_parents-2)
        if i2 == i1 : #problème si deux fois le même parent !!
            i2 += 1
        w1 = parents[i1, 20] / (parents[i1, 20] + parents[i2, 20]) #poids du parent 1
        w2 = parents[i2, 20] / (parents[i1, 20] + parents[i2, 20]) #poids du parent 2
        enfants[i] = (w1 * parents[i1] + w2 * parents[i2]) #moyenne pondérée
    return (enfants)

N_mutants = int(0.1 * N_population)
epsilon = 0.05
complementEpsilon = 1 - epsilon

def mutation (mutants) :
    #les mutants sont les meilleures compositions entre 40% et 50%
    for j in range (N_mutants) :
        iplus = randint(0, 19) #choix de l'oxyde qui gagne epsilon
        imoins = randint(0, 19) #choix de l'oxyde qui perd epsilon
        if imoins == iplus : #problème si deux fois le même oxyde !!
            imoins = (1 + imoins)%19
        mutantPlus = mutants[j, iplus]
        mutantMoins = mutants[j, imoins]
        if (mutantMoins > epsilon) and (mutantPlus < complementEpsilon) :
            mutants[j, iplus] = mutantPlus + epsilon
            mutants[j, imoins] = mutantMoins - epsilon
    return (mutants)
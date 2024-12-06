from random import randint

N_population = 10000

N_parents = 0.1 * N_population
N_enfants = 0.4 * N_population

def crossover (parents, fitness) :
    enfants = []
    for i in range (N_enfants) :
        i1 = randint(0, N_parents-1)
        i2 = randint(0, N_parents-2)
        if i2 == i1 : #problème si deux fois le même parent !!
            i2 += 1
        w1 = fitness[i1] / (fitness[i1] + fitness[i2]) #poids du parent 1
        w2 = fitness[i2] / (fitness[i1] + fitness[i2]) #poids du parent 2
        enfants.append(w1 * parents[i1] + w2 * parents[i2]) #moyenne pondérée
    return (enfants)

#N_mutants = 0.1 * N_population
N_nonMutants = 0.9 * N_population
epsilon = 0.05

def mutation (pop) :
    for j in range (N_nonMutants, N_population) :
        iplus = randint(0, 19) #choix de l'oxyde qui gagne epsilon
        imoins = randint(0, 18) #choix de l'oxyde qui perd epsilon
        if imoins >= iplus : #problème si deux fois le même oxyde !!
            imoins += 1
        pop[j, iplus] = pop[j, iplus] + epsilon
        pop[j, imoins] = pop[j, imoins] - epsilon
    return (pop)
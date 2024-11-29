from random import randint
import numpy as np

N_population = 1000

N_parents = int(0.1 * N_population)
N_childs = int(0.4 * N_population)

# def crossover (parents) :
#     #Dans parents chaque individu est représenté par 21 floats dont le dernier est la valeur de fitness
#     childs = np.array([[0.] * 21] * N_childs)
#     for i in range (N_childs) :
#         i1 = randint(0, N_parents-1)
#         i2 = randint(0, N_parents-2)
#         if i2 == i1 : #problème si deux fois le même parent !!
#             i2 += 1
#         w1 = parents[i1, 20] / (parents[i1, 20] + parents[i2, 20]) #poids du parent 1
#         w2 = parents[i2, 20] / (parents[i1, 20] + parents[i2, 20]) #poids du parent 2
#         childs[i] = (w1 * parents[i1] + w2 * parents[i2]) #moyenne pondérée
#     return (childs)

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


# températures de fusion des oxydes (en °C) : 
# [1610, 2045, 2852, 2580]
# dureté des oxydes :
# [7, 9, 5.5]
# TiO2 est un agent nucléant utilisé pour les plaques vitrocéramiques (très faible coeff de dilatation thermique).
# source : L'élémentarium

# fondants = ['Na2O', 'K2O', 'MgO']
# stabilisants = ['CaO', 'ZnO']
# ZnO augmente l'élasticité
# source : https://lasirene.e-monsite.com/pages/le-verre/-.html

# formateurs = ['SiO2']
# SiO2 augmente la dureté du verre
# fondants = ['Na2O', 'K2O', 'MgO']
# stabilisants = ['CaO', 'ZnO']
# source : https://infovitrail.com/contenu.php/fr/d/---la-composition-du-verre/e9b609c9-91f5-4a08-86a6-6112dc12b66d

# fondants = ['Na2O', 'K2O'] + ['CaO', 'MgO'] (mais moins bien que les deux premiers)
# modificateurs = ['CaO', 'MgO', 'Al2O3'] #augmentent les propriétés de durabilité chimique et mécanique (E ??)
# formateurs = ['SiO2'] #essentiels
# source : Franck

fondants = [False, False, False, True, True, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] # diminuent Tm
durability = [True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False] #augmentent E
other = [False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False]

# def crossover (mom, dad) :
#     childs = np.zeros((len(mom), len(mom[0])))
#     for i in range (N_childs) :
#         if mom[i, 21] < dad[i, 21] : #choix du meilleur E
#             bestE = dad[i]
#         else :
#             bestE = mom[i]
#         if mom[i, 23] < dad[i, 23] : #choix du meilleur Tm
#             bestTm = mom[i]
#         else :
#             bestTm = dad[i]
#         childs[i][fondants] = bestTm[fondants]
#         childs[i][durability] = bestE[durability]
#         childs[i][other] = (mom[i][other] + dad[i][other])/2
#         sum = np.sum(childs[i])
#         childs[i] = childs[i] / sum
#     return (childs)

def crossover (mom, dad) :
    childs = np.zeros((len(mom), len(mom[0])))
    for i in range (N_childs) :
        wMomTm = mom[i, 23] / (mom[i, 23] + dad[i, 23])
        wDadTm = dad[i, 23] / (mom[i, 23] + dad[i, 23])
        wMomE = mom[i, 21] / (mom[i, 21] + dad[i, 21])
        wDadE = dad[i, 21] / (mom[i, 21] + dad[i, 21])
        childs[i][fondants] = wMomTm * mom[i][fondants] + wDadTm * dad[i][fondants]
        childs[i][durability] = wMomE * mom[i][durability] + wDadE * dad[i][durability]
        childs[i][other] = (mom[i][other] + dad[i][other])/2
    return (childs)
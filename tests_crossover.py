from random import randint

N_population = 10000

N_parents = 0.1 * N_population
N_enfants = 0.4 * N_population

from crossover import crossover

N_nonMutants = 0.9 * N_population
epsilon = 0.05

from crossover import mutation

from random import randint
import numpy as np

N_population = 1000

N_parents = int(0.2 * N_population)
N_enfants = int(0.1 * N_population)

def parentcontest (parents,strategie) :
    #Dans parents chaque individu est représenté par 21 floats dont le dernier est la valeur de fitness
    if strategie =='China' and N_parents>= 2*N_enfants:
        dads = parents[::2]
        moms = parents[1::2]
        return (np.array(moms[:N_enfants]),np.array(dads[:N_enfants]))    
    if strategie =='tournament':
        N_tournament =5
        dads =[]
        moms =[]
        for i in range(N_enfants):
            t = np.array([parents[i,:] for i in np.random.choice(N_parents,N_tournament*2)])
            dad = t[np.argmax(t[:N_tournament,-1])]
            mom = t[np.argmax(t[N_tournament:,-1])]
            dads.append(dad)
            moms.append(mom)
        return (np.array(moms),np.array(dads))
    #Fallback
    return (np.random.shuffle(parents)[:N_enfants],np.random.shuffle(parents)[:N_enfants])


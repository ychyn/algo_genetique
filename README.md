# algo_genetique

Ce code fait partie de notre mini-projet du MIG Verre 2024. L'objectif du mini-projet est l'obtention d'une composition de verre avec des qualités avantageuses pour une application potentiellement industrielle.

<!--"une application potentiellement industrielle" :D -->

On utilise l'outil de prédiction des propriétés du verre à travers des réseaux de neurones codé par Franck Pigeonneau (qui constitue la plus grande partie de ce repo). On a codé l'algorithme génétique qui va, en utilisant ce réseau de neurones nous permettre d'obtenir une composition optimale dont on essaiera la fabrication dans le four situé à l'Ecole des Mines à Sophia Antipolis.

## La liste de fonctions

### Fichier glassdata.py

- `GlassData.better_random_composition(self,Nglass,xmin,xmax)`  
Renvoie `Nglass` couples du type `(composition, masse_molaire)` où la composition est représentée par un tableau de fractions molaires de chaque oxyde. Ces fractions vérifient `xmin[i] <= composition[i] <= xmax[i]`.

### Fichier algo_genetique.ipynb

Information générale :

Dans cette fichier les générations sont représentés par des tableaux 2D où chaque ligne est de la forme :

| Composition | Propriétés | Score de minimisation |
| --- | --- | --- |
| 'SiO2', 'B2O3', 'Al2O3', 'MgO', 'CaO', 'BaO', 'Li2O', 'Na2O', 'K2O', 'ZnO', 'SrO', 'PbO', 'Fe2O3', 'Y2O3', 'La2O3', 'TiO2', 'GeO2', 'ZrO2', 'P2O5', 'V2O5' | 'rho', 'E', 'Tg', 'Tmelt' | 'F' |

Le tableau représentant chaque génération est trié dans l'ordre décroissant des 'F'.

Les constantes :

- `available_mat = ['SiO2', 'Al2O3', 'MgO', 'CaO', 'Na2O', 'K2O','ZnO', 'TiO2']`  
La liste des oxydes disponibles.
- `xmin`  
La liste de fractions molaires minimales à utiliser.
- `xmax`  
La liste de fractions molaires maximales à utiliser.

Les fonctions :

- `prop_calculation(composition)`  
Renvoie les propriétés de la composition donnée sous la forme d'un tableau où chaque ligne correspond aux valeurs $\rho ,\, E ,\, T_g ,\, T_m$ du verre.
- `normalize(prop)`  
Normalise le tableau de propriétés du verre.
- `fitness_func(prop_normalized,weight,minimize)`  
Renvoie un tableau de valeurs de fitness associés aux propriétés. Les paramètres sont :
    - `prop` est une array avec les propriétés normalisées du verre.
    - `weight` est le poids qu'on accorde à chacune des propriétés.
    - `minimize` est une liste de booléens qui vaut `True` si on veut minimiser la variable et `False` si on veut la maximiser.
- `stack_by_f(population,properties,F)`  
Renvoie la génération composée de la population, ses propriétés et les valeurs de F correspondantes.
- `init_properties(population)`  
Renvoie la génération correspondant à la population donnée.
- `compute_properties(generation)`  
Recalcule les propriétés et la valeur de F pour la génération donnée, puis la renvoie.
- `init_pop(N_population)`  
Renvoie une génération aléatoire de la taille donnée.
- `population_selection(generation)`  
Renvoie 2 tableaux qui correspondent aux survivants et aux parents.
- `crossover (parents)`  
Renvoie les enfants en combinant des parents.
- `mutation (mutants)`  
Renvoie les individus en les mutant.
- `new_generation(old_generation)`  
Renvoie la génération suivante.
- `evolution(generation,N)`  
Renvoie la génération obtenue après `N` cycles à partir de la génération donnée.

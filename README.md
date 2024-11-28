# algo_genetique

Ce code fait partie de notre mini-projet du MIG Verre 2024. L'objectif du mini-projet est l'obtention d'une composition de verre avec des qualités avantageuses pour une application potentiellement industrielle.

On utilise l'outil de prédiction des propriétés du verre à travers des réseaux de neurones codé par Franck Pigeonneau (qui constitue la plus grande partie de ce repo). On a codé l'algorithme génétique qui va, en utilisant ce réseau de neurones tenter d'obtenir une composition optimale dont on essaiera la fabrication dans le four situé à l'Ecole des Mines à Sophia Antipolis.

## La liste de fonctions

### Fichier glassdata.py

- `GlassData.better_random_composition(self,Nglass,xmin,xmax)`  
Renvoie `Nglass` couples du type `(composition, masse_molaire)` où la composition est représentée par un tableau de fractions molaires de chaque oxyde. Ces fractions vérifient `xmin[i] <= composition[i] <= xmax[i]`

### Fichier algo_genetique.ipynb

Les constantes :

- `available_mat = ['SiO2', 'Al2O3', 'MgO', 'CaO', 'Na2O', 'K2O','ZnO', 'TiO2']`  
La liste des oxydes disponibles
- `xmin`  
La liste de fractions molaires minimales à utiliser
- `xmax`  
La liste de fractions molaires maximales à utiliser

Les fonctions :

- `init_pop(N_population)`  
Renvoie la génération 0 sous la forme d'un tableau de compositions de taille `N_population`
- `prop_calculation(population)`  
Renvoie les propriétés de la population sous la forme d'un tableau où chaque ligne correspond aux valeurs $\rho ,\, E ,\, T_g ,\, T_m$ du verre
- `normalize(prop)`  
Normalise le tableau de propriétés du verre
- `fitness_func(prop_normalized,weight,minimize)`  
Renvoie un tableau de valeurs de fitness associés à chaque verre. Les paramètres sont :
    - `prop` est une array avec les propriétés du verre normalisées
    - `weight` est le poids qu'on accorde à chacune des propriétés
    - `minimize` est une liste de booléens qui vaut `True` si on veut minimiser la variable et `False` si on veut la maximiser
- `sort_by_f(population,F)`  
Renvoie la population triée dans l'ordre croissant de F

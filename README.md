# algo_genetique

Ce code fait partie de notre mini-projet du MIG Verre 2024. L'objectif du mini-projet est l'obtention d'une composition de verre avec des qualités avantageuses pour une application potentiellement industrielle.

<!--"une application potentiellement industrielle" :D -->

On utilise l'outil de prédiction des propriétés du verre à travers des réseaux de neurones codé par Franck Pigeonneau (qui constitue la plus grande partie de ce repo). On a codé l'algorithme génétique qui va, en utilisant ce réseau de neurones nous permettre d'obtenir une composition optimale dont on essaiera la fabrication dans le four situé à l'Ecole des Mines à Sophia Antipolis.

## La liste de fonctions

### Fichier glassdata.py

- `GlassData.better_random_composition(self,Nglass,xmin,xmax)`  
Renvoie `Nglass` couples du type `(composition, masse_molaire)` où la composition est représentée par un tableau de fractions molaires de chaque oxyde. Ces fractions vérifient `xmin[i] <= composition[i] <= xmax[i]`.

### Fichier algo_genetique.py

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

La classe `EvolutionModel`  

Cette classe possède comme attributs :

Les bases de données correspondantes à $\rho ,\, E ,\, T_g ,\, T_m ,\, T_l ,\, D_0$ :

- `self.dbrho`
- `self.dbE`
- `self.dbTannealing`
- `self.dbTmelt`
- `self.dbTliq`
- `self.datadisso`

Les modèles qui prédisent les propriétés $V_m ,\, \frac{E}{2G} (= V_t) ,\, T_m ,\, T_l$ :

- `self.nnmolvol`
- `self.nnmodelEsG`
- `self.nnTmelt`
- `self.nnTliq`

Les tableaux représentant les contraintes sur les composants à utiliser :

- `self.xmin`
- `self.xmax`

Les penalités pour fitness function:

- `self.penalties_onMin_normalized`
- `self.penalties_onMax_normalized`

Les fonctions utilisées lors de passage d'une génération à l'autre :

- `self.crossover`
- `self.mutation`
- `self.population_selection`

Elle possède les méthodes suivantes :

<!-- TO DO -->

- `load(self)`  
Charge les bases de données et les modèles ainsi que les fonction de passage de base
- `prop_calculation(self, composition)`  
Renvoie les propriétés de la composition donnée sous la forme d'un tableau où chaque ligne correspond aux valeurs $\rho ,\, E ,\, T_g ,\, T_m$ du verre.
- `normalize(self, prop)`  
Normalise le tableau de propriétés du verre.
- `fitness(self, property_normalized)`  
Renvoie la valeur de fitness pour les propriétés donnés.
- `fitness_func(self, prop_normalized)`  
Renvoie les valeurs de fitness pour le tableau de propriétés donné.
- `init_properties(self, population)`  
Renvoie la génération correspondant à la population donnée, triée par les valeurs de fitness.
- `update_properties(self, generation)`  
Recalcule les propriétés pour la génération donnée, la trie par les valeurs de fitness et la renvoie.
- `init_pop(self, N_population)`  
Renvoie une génération aléatoire de la taille donnée.
- `new_generation(self, old_generation)`  
Renvoie la génération suivante.
- `evolution(self, generation, N)`  
Renvoie la génération obtenue après N cycles à partir de la génération donnée.

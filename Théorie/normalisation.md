# La normalisation

## C'est quoi ?
Normaliser c'est transformer les valeurs numériques d'un jeu de données pour les ramener à une échelle comparable.
On change leur "unité".

ex: 
| Taille (cm) |Poids (kg)|
|----------|------------|
|170            | 60|
|180             |80|
|160           |  55|

Les tailles sont en centaines, le poids en dizaines.
Le risque c'est que dans certains algorithmes, la variable "taille" domine les calculs simplement car les valeurs sont plus grandes.

L'objectif est clair:
- Mettre toutes les variables à la même echelle pour éviter les biais dus aux unités.
- Faciliter l'entrainement de modèles car les gradients convergent plus vite.
- Rendre les différences ou les similarités significatives (utiles pour les algos de clustering)

## Les principales méthodes de normalisation
**1. Min-max scaling (normalisation simple)**

$$ x' = {(x - xmin) \over (xmax - xmin)} $$

- ramène les valeurs entre 0 et 1
- présèrve la forme de la distribution

Très utilisée pour les réseaux de neurones.

ex:  
min = 10  
max = 20  
|valeur |     valeur normalisée|
|--------|-----------|
|10     |     (10-10)/(20-10)=0|
|15     |     (15-10)/(20-10)=0.5|
|20     |     (20-10)/(20-10)=1|

**2. Standardisation (z-score normalization)**

$$ x' = {(x - \mu)\over \alpha} $$

où   
$\mu$ = moyenne  
$\alpha$ = écart-type  

- Donne une variable centrée sur 0 et avec une dispersion (écart-type) de 1

Très utilisée dans les modèle linéaires et les algos sensibles à la variance.

ex:  
$$\mu $$ = 10  
$$\alpha $$ = 2
|Valeur|      z-score|
|---------|-----------|
|10   |       (10-10)/2 = 0|
|12   |       (12-10)/2 = 1|
|8     |      (8-10)/2 = -1|

**3. Mean Normalization**

$$ x' = {(x - \mu)\over(xmax - xmin)} $$

- Centrage sur la moyenne (pas forcément 0-1)
- Moins fréquent, mais utile quand on souhaite centrer tout en gardant l'échelle

**4. Unit Vector / L2 Normalization**

$$ x' = {x\over||x||} $$

- On divise par la norme du vecteur

Très utilisé dans les modèles de texte

ex:  
Vecteur: [3, 4]   
Norme = $\sqrt{3^2 + 4^2} = 5$  

Normalisé = [3/5, 4/5] = [0.6, 0.8]

## Quand utiliser quelle méthode ?

Min-Max: 
    intervalle [0, 1] 
    réseaux de neurones, images, etc

Standardisation z-score
    intervalle: centré sur 0, écart-type de 1
    Régression, PCA, SVM

Unit vector
    intervalle: norme = 1
    NLP, cosine similarity

Mean normalization
    intervalle: [-1, 1]
    Données centrées mais bornées

## Exemples concrets

z-score:
    Données : [5, 10, 15]

    moyenne = (5+10+15)/3 = 10

    écart-type ≈ 4.08
    → z-scores :
    (5−10)/4.08 = −1.22
    (10−10)/4.08 = 0
    (15−10)/4.08 = +1.22



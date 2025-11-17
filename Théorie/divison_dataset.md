# Pourquoi les séries temporelles sont spéciales

Dataset classique: les lignes sont indépendantes    
Dataset temporel: la valeur du jour t dépend de t-1 etc.

On ne mélange pas un dataset temporel.

# Structure

ex: 
| jour | prix |
|--------|----------|
| 0 | 100 |  
| 1 | 102 |  
| 2 | 101 |  
| 3 | 108 |  
| 4 | 110 |  
| 5 | 115 |  

Un réseau de neurones ne peut pas travailler lignes par ligne. Il a besoin d'avoir les informations de x lignes en entrée. => des **séquences**

ex :
```
[100, 102, 101] -> 108
[102, 101, 108] -> 110
[101, 108, 110] -> 115

On prend 3 points en entée et on prédit le point suivant. C'est le sliding window.
```

# Comment découper un dataset temporel ?
## 1. Séparer train/validation/test
Tout ce qui est futur ne doit jamais être visible durant l'entrainement.

70% pour l'entrainement   
20% pour la validation    
10% pour les tests  

```
train_size = int(len(data) * 0.7)

train_data = data[:train_size] # Du début du dataset jusqu'à 70% 
val_data = data[train_size:] # De 70% du dataset jusqu'à la fin
```

## Créer des séquences (window)
supposons un `window_size = 3`

Entrée : [100, 102, 101]
Target : 108

Entrée : [102, 101, 108]
Target : 110

Entrée : [101, 108, 110]
Target : 115

Attention de créer les windows **après** la séparation entrainement/validation.    
Car une séquence pourrait se retrouver dans le dataset d'entrainement mais contenir des informations de la validation. Ca constituerait de la fuite de données.

```
def create_sequences(data, window_size=3):
    X = []
    y = []
    for i in range(len(data) - window_size):
        seq = data[i : i + window_size]    # fenêtre glissante
        # dans le cas où la valeur suivante correspond à la valeur à deviner
        target = data[i + window_size]     # valeur suivante
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

```

# Reshape pour LSTM
Un LSTM ne prend jamais un simple tableau 2D, il a besoin d'une entrée avec 3 dimensions
Les LSTM attendent une forme : (batch_size, sequence_length, nb_features)
où :
- batch_size: Nombre d'échantillons envoyés au réseau à la fois
- sequence_length: La longueur de chaque séquence temporelle, la window (ici, 3)
- nb_features: Nombre de colonnes dans le dataset

Avec NumPy, shape[0] représente le nombre de séquences, et shape[1] la taille des windows.

```
print(X_train.shape)
--> (4, 3)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val   = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

print(X_train.shape)
--> (4, 3, 1) # Il y a 4 séquences, chacunes dure 3 pas de temps, et à chaque window on observe 1 caractéristique
```


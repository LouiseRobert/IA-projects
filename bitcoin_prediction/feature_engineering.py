###
# Feature engineering pas à pas pour transformer notre dataset de bitcoin en dataset digeste pour un modèle.
###

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == "__main__":
    torch.cuda.empty_cache()

    print(f"{datetime.now()} : Chargement et préparation des données.")


    df = pd.read_csv("../dataset/bitcoin/btcusd_1-min_data.csv")

    # On convertis le timestamp en datetime et on trie le dataset
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # On met Timestamp en index du dataframe
    df.set_index('Timestamp', inplace=True)

    print(df.shape)
    # => (7280317, 5) Donc 5 colonnes et 7280317 lignes

    print(df.tail())
        
    #                          Open      High       Low     Close    Volume
    # Timestamp
    # 2025-11-04 23:53:00  101583.0  101687.0  101583.0  101676.0  8.483488
    # 2025-11-04 23:54:00  101647.0  101663.0  101607.0  101610.0  1.060988
    # 2025-11-04 23:55:00  101610.0  101667.0  101608.0  101638.0  0.123431
    # 2025-11-04 23:56:00  101639.0  101686.0  101637.0  101638.0  1.115895
    # 2025-11-04 23:57:00  101634.0  101634.0  101560.0  101627.0  0.945285

    # les valeurs de prix sont très grandes et très différentes des valeurs de volume
    # On pourrait supprimer les colonnes High et Low, elles ne nous intéressent pas.
    # On pourrait aussi supprimer la colonne close et la remplacer par la variation entre open et close.

    # La normalisation Min-Max me semble la plus pertinente mais à condition de normaliser aussi la colonne Volume.
    # Il faudrait trouver un moyen de pondérer les colonnes qui concernent le prix et celles qui concernent le volume.
    # Par exemple en multipliant par 10 celles du prix, pour inciter à ne pas trop accorder de poids au Volume.

    print(f"{datetime.now()} : Resample complet et ajout de la variation horaire.")

    # On ne récupère qu'une ligne par heure car c'est suffisant
    df = df.resample('1h').agg({
        'Open': 'first',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Caclul de la variation en pourcentage
    df['Variation'] = ((df['Close'] / df['Open'])-1)*100

    # Suppression des colonnes dont on ne se sert pas
    df = df.drop("Close", axis=1)

    print(df.tail())

    #                          Open      Volume  Variation
    # Timestamp
    # 2025-11-04 19:00:00  101306.0  294.845561  -0.830158
    # 2025-11-04 20:00:00  100463.0  517.080809   0.236903
    # 2025-11-04 21:00:00  100701.0  440.281391  -0.436937
    # 2025-11-04 22:00:00  100199.0  345.298771   1.042925
    # 2025-11-04 23:00:00  101266.0   96.336257   0.356487

    print(f"{datetime.now()} : Normalisation de la colonne Open.")

    # On applique le Min-Max à la colonne Open
    minmax = MinMaxScaler()
    
    df['Open'] = minmax.fit_transform(df[['Open']]) # Double crochet car prend en entrée un dataframe et non une série, on lui passe un dataframe (N, 1)
    # Le fit ne fait que lire les données et calculer les min et max
    # alors que le transform applique la transformation et renvoie une nouvelle série.

    print(df.tail())
    #                          Open      Volume  Variation
    # Timestamp
    # 2025-11-04 19:00:00  0.803410  294.845561  -0.830158
    # 2025-11-04 20:00:00  0.796724  517.080809   0.236903
    # 2025-11-04 21:00:00  0.798612  440.281391  -0.436937
    # 2025-11-04 22:00:00  0.794631  345.298771   1.042925
    # 2025-11-04 23:00:00  0.803093   96.336257   0.356487

    # Maintenant il faudrait normaliser la colonne volume aussi, on pourrait lui appliquer un min max
    # ou bien lui appliquer une normalisation z-score pour mettre en avant les moments où le volume est bas (il serait négatif)
    # le volume est "moyen" à 0 et fort en positif.

    print(f"{datetime.now()} : Normalisation de la colonne Volume.")
    
    # On applique la normalisation z-score à la colonne Volume
    zscore = StandardScaler()

    df['Volume'] = zscore.fit_transform(df[['Volume']])

    print(df.tail())

    #                          Open    Volume  Variation
    # Timestamp
    # 2025-11-04 19:00:00  0.803410 -0.024385  -0.830158
    # 2025-11-04 20:00:00  0.796724  0.368422   0.236903
    # 2025-11-04 21:00:00  0.798612  0.232677  -0.436937
    # 2025-11-04 22:00:00  0.794631  0.064792   1.042925
    # 2025-11-04 23:00:00  0.803093 -0.375256   0.356487
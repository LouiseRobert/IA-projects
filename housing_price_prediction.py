
"""
Prédiction des prix immobiliers en Californie
Auteur : Inconnu
Description :
Ce code n'est pas le mien, je n'ai fait qu'une étude sur celui-ci pour comprendre le fonctionnement du machine learning
"""

import torch # pytorch pour le modèle de prédiction
import torch.nn as nn # nn pour neural network, c'est notre réseau de neuronne
import torch.optim as optim # optim pour l'optimiseur, c'est lui qui ajuste les résultats lors de l'entrainement pour apprendre
from sklearn.model_selection import train_test_split # fonction permettant de couper le jeu de données en deux, pour l'entrainement et pour l'évaluation
from sklearn.datasets import fetch_california_housing # le dataset qu'on va utiliser sur l'immobilier californien
from sklearn.preprocessing import StandardScaler # Fonction pour normaliser le jeu de données (pour faciliter l'apprentissage)
import matplotlib.pyplot as plt # pour la visualisation des données
import seaborn as sns # seaborn est aussi pour la visualisation, mais en plus évolué

# Charger les données
data = fetch_california_housing(as_frame=True) # on place dans data les données immobilières
df = data.frame # on place dans df nos données transformées en dataframe

print(df.head()) # affiche le début du dataframe
print(df.shape) # affiche la "forme" du dataframe, sous la forme (lignes, colonnes)

X = df.drop("MedHouseVal", axis=1).values # Dans X on place les valeur de la dataframe sans la colonne de valeur des maisons (axis=1 agit sur les colonnes)
y = df["MedHouseVal"].values.reshape(-1, 1) # on met dans y les valeurs de prix des maisons qu'on a transformé en matrices (avec reshape)

# Visualisation de la corrélation
plt.figure(figsize=(8,6)) # on va créer la heatmap
sns.heatmap(df.corr(), cmap='coolwarm', annot=False) # ce sera une heatmap, avec comme données le résultat de la fonction de corrélation
plt.title("Corrélation entre les variables") # On donne un titre à notre figure
plt.show() # on affiche la figure

# Normalisation (important pour l'apprentissage)
scaler_X = StandardScaler() # création d'un objet de normalisation pour les variables d'entrées
scaler_y = StandardScaler() # création d'un objet de normalisation pour la variable cible
X = scaler_X.fit_transform(X) # fit (calcule les statistiques) et transform (normalise les données) sur les données d'entrées, toutes les colonnes ont une moyenne de 0 et une variance de 1
y = scaler_y.fit_transform(y) # idem sur les données cibles

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # découpage du jeu de données tel que: X_train et y_train = 80% et X_test et y_test = 20%

# Conversion en tenseurs PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # création du device pour exécuter ce code sur ma carte graphique
X_train = torch.tensor(X_train, dtype=torch.float32).to(device) # On envoie les données dans la mémoire du device
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Définir le modèle (régression linéaire)
model = nn.Linear(X_train.shape[1], 1).to(device) # création de notre réseau de neuronnes avec une couche linéaire, et en lui donnant son échantillon d'entrainement en entrée et en lui demandant 1 sortie

# Définir la fonction de coût et l’optimiseur
criterion = nn.MSELoss() # dans criterion on va créer notre fonction de cout (Mean Squared Error), qui nous permet d'évaluer à quel point le modèle est performant
optimizer = optim.Adam(model.parameters(), lr=0.01) # On utilise l'optimizer Adam auquel on donne les paramètres de notre modèle, ces parametres sont les "poids" et c'est eux que doit ajuster l'optimizer, le learning rate est à 0;01

# Entraînement
epochs = 500 # On va faire 500 tours d'entrainement
for epoch in range(epochs): 
    model.train() # On passe en mode entrainement sur le model
    optimizer.zero_grad() # remet à 0 les gradient des poids accumulés avant chaque epoch, sinon ils s'accumuleraient

    outputs = model(X_train) # On entraine le modèle sur l'echantillon X_train et on stock la sortie de l'entrainement ???
    loss = criterion(outputs, y_train) # compare les résultats de l'entrainement (outputs) avec les vrais résultats (y_train) et calcule la perte
    loss.backward() # Backpropagation : PyTorch calcule automatiquement les dérivées de la perte par rapport à chaque poids du modèle. (c'est loss.backward() et non model.backward() car on part de la perte pour remonter au model)
    optimizer.step() # met à jour les poids du model en suivant les gradients calculés précédement
    if (epoch+1) % 100 == 0: # Toutes les 100 epoch on affiche où on en est dans l'entrainement
        print(f"Époque {epoch+1}/{epochs}, Perte : {loss.item():.4f}")

# Évaluation
model.eval() # on passe en mode évaluation 
with torch.no_grad(): # désactive le calcul des gradients, pour faire des calculs plus rapides, on ne va pas faire de backpropagation (on arrete d'apprendre)
    y_pred = model(X_test) # Fait des prédictions sur les données de test
    test_loss = criterion(y_pred, y_test).item() # calcule la perte sur les données de test et item() extrait la valeur numérique du tenseur

print(f"\nPerte (MSE) sur test : {test_loss:.4f}") # on affiche notre loss effective sur l'évaluation

# Revenir aux valeurs d'origine (pour tracer)
y_test_cpu = scaler_y.inverse_transform(y_test.cpu()) # On avait normalisé les valeurs, on les remet à leur échelle d'origine pour pouvoir les visualiser
y_pred_cpu = scaler_y.inverse_transform(y_pred.cpu())

# Visualisation
plt.figure(figsize=(6,6)) # on prépare la prochaine figure
plt.scatter(y_test_cpu, y_pred_cpu, alpha=0.6) # Affiche un nuage de points entre : les vraies valeurs (y_test_cpu) les valeurs prédites (y_pred_cpu) Si ton modèle est bon, les points se rapprochent de la diagonale rouge [0,5] → [0,5].
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Régression linéaire PyTorch sur GPU")
plt.plot([0, 5], [0, 5], color='red') 
plt.show()

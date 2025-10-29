"""
Iris growth classifier
Auteur : Louise ROBERT
Description :
Classifieur de variété d'iris selon des critères physiques comme la taille des pétales.
"""

import torch # pytorch pour le modèle de prédiction
import torch.nn as nn # nn pour neural network, c'est notre réseau de neuronne
import torch.nn.functional as F
import torch.optim as optim # optim pour l'optimiseur, c'est lui qui ajuste les résultats lors de l'entrainement pour apprendre
from sklearn.model_selection import train_test_split # fonction permettant de couper le jeu de données en deux, pour l'entrainement et pour l'évaluation
from sklearn.preprocessing import StandardScaler # Pour standardiser les données
from sklearn.preprocessing import LabelEncoder # pour encoder mes labels de fleurs

import matplotlib.pyplot as plt # pour la visualisation des données
import seaborn as sns # seaborn est aussi pour la visualisation, mais en plus évolué
import pandas as pd
import pickle # pour sauvegarder le scaler et l'encodeur

import datetime

datas = pd.read_csv("dataset/iris/IRIS.csv")
# print(type(dataframe['sepal_length']))

X = datas[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = datas['species']

### Standardisation des données

X_scaler = StandardScaler()
y_encoder = LabelEncoder()

y = y_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

X_scaler.fit(X_train)

X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print(f"Taille du jeu de train : {X_train.shape}")
print(f"Taille du jeu de test: {X_test.shape}")

sns.violinplot(data=datas)
# plt.show()


### Convertir en tenseur pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # création du device pour exécuter ce code sur ma carte graphique

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

class IrisIdentifier(nn.Module):
    """
    On va définir 3 couches Linear (donc de régression linéaire) et une fonction ReLU (Rectified Linear Unit) qui n'est pas une fonction linéaire.
    ReLU est une fonction d'activation
    Dans Forward on altene Linear avec ReLU pour approximer une fonction non linéaire complexe.
    """
    def __init__(self):
        super(IrisIdentifier, self).__init__()

        # Définition des couches du réseau
        self.couche1 = nn.Linear(4, 8) # 4 parametres d'entrées vont donner 8 neuronnes cachés
        self.couche2 = nn.Linear(8, 8) # Ces 8 neuronnes vont en donner 8 autres
        self.couche3 = nn.Linear(8, 3) # Ces 8 neuronnes vont donner 3 sorties possibles (les 3 espèces d'iris possibles)

        # Fonction d'activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.couche1(x)
        x = self.relu(x)

        x = self.couche2(x)
        x = self.relu(x)

        x = self.couche3(x)
        ## Pas de relu car la fonction de loss s'en occupe automatiquement

        return x
    
model = IrisIdentifier().to(device)

"""
Notre criterion va calculer la quantification mathématique de l'erreur de notre modèle.
CrossEntropyLoss est une fonction adaptée à la classification
On aurait pu utiliser MSE (Mean Squared Error), MAE (Mean Absolute Error) ou Huber Loss (mélange MSE et MAE)
"""
criterion = nn.CrossEntropyLoss()

"""
L'optimizer va modifier les poids selon la Loss calculée 
Il existe d'autres optimizers: SGD, RMSProp, AdamW
"""
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 95
model.train()

for i in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)

    loss.backward()

    optimizer.step()

    if (i+1) % 50 == 0:
        print(f"Époque {i+1}/{epochs}, Perte : {loss.item():.4f}")

# évaluation
model.eval()

with torch.no_grad():
    y_pred = model(X_test)

    predictions = torch.argmax(y_pred, dim=1)

    accuracy = (predictions == y_test).float().mean()

    test_loss = criterion(y_pred,y_test).item()

    print(f"Précision sur l'évaluation : {accuracy:.2f}")
    print(f"Perte sur l'évaluation : {test_loss:.4f}")


# Enregistrement des poids du model
torch.save(model.state_dict(), "iris_model.pth")

# Enregistrement du scaler et de l'encodeur, pour pouvoir re normaliser mes futures données

with open("iris_scaler.pkl", "wb") as f:
    pickle.dump(X_scaler, f)

with open("iris_encoder.pkl", "wb") as f:
    pickle.dump(y_encoder, f)


#### Réutilisation du modèle sauvegardé

model2 = IrisIdentifier().to(device)
model2.load_state_dict(torch.load("iris_model.pth"))
model2.eval()

with open("iris_scaler.pkl", "rb") as f:
    X_scaler = pickle.load(f)

with open("iris_encoder.pkl", "rb") as f:
    y_encoder = pickle.load(f)

# exemple:

sample = [[5.1,3.5,1.4,0.2]] # double crochet car matrice 1x4

sample_scaler = X_scaler.transform(sample) # standardisation des données sample
sample_tensor = torch.tensor(sample_scaler, dtype=torch.float32).to(device) # transformation des données sample en tensor car c'est ça que comprend les modèles pytorch

with torch.no_grad(): 
    output = model2(sample_tensor) # prédiction avec le modèle sauvegardé
    probabilites = F.softmax(output, dim=1) # convertit les scores en probabilités
    prediction = torch.argmax(output, dim=1).item() 

predicted_label = y_encoder.inverse_transform([prediction])[0]
confiance = probabilites[0, prediction].item()

print(f"L'iris prédit est : {predicted_label}")
print(f"Confiance du modèle : {confiance*100:.2f}%")

for i, variete in enumerate(y_encoder.classes_):
    print(f"{variete} : {probabilites[0, i]*100:.2f}%")


"""
Taille du jeu de train : (120, 4)
Taille du jeu de test: (30, 4)
Époque 50/1000, Perte : 0.2632
Époque 100/1000, Perte : 0.0255
Époque 150/1000, Perte : 0.0127
Époque 200/1000, Perte : 0.0068
Époque 250/1000, Perte : 0.0039
Époque 300/1000, Perte : 0.0024
Époque 350/1000, Perte : 0.0016
Époque 400/1000, Perte : 0.0012
Époque 450/1000, Perte : 0.0009
Époque 500/1000, Perte : 0.0007
Époque 550/1000, Perte : 0.0005
Époque 600/1000, Perte : 0.0004
Époque 650/1000, Perte : 0.0003
Époque 700/1000, Perte : 0.0003
Époque 750/1000, Perte : 0.0002
Époque 800/1000, Perte : 0.0002
Époque 850/1000, Perte : 0.0002
Époque 900/1000, Perte : 0.0002
Époque 950/1000, Perte : 0.0001
Époque 1000/1000, Perte : 0.0001
Précision sur l'évaluation : 0.87
Perte sur l'évaluation : 1.0757 <== enorme overfitting car trop d'epoch pour un si petit dataset

Taille du jeu de train : (120, 4)
Taille du jeu de test: (30, 4)
Époque 50/95, Perte : 0.2317
Précision sur l'évaluation : 0.90
Perte sur l'évaluation : 0.2011
L'iris prédit est : Iris-setosa
Confiance du modèle : 99.96%
Iris-setosa : 99.96%
Iris-versicolor : 0.04%
Iris-virginica : 0.00%
"""

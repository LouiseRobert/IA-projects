"""
Prédiction du prix de cloture de l'or à J+1
Auteur : Louise ROBERT
Description :

"""

import torch # pytorch pour le modèle de prédiction
import torch.nn as nn # nn pour neural network, c'est notre réseau de neuronne
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim # optim pour l'optimiseur, c'est lui qui ajuste les résultats lors de l'entrainement pour apprendre
from sklearn.model_selection import train_test_split # fonction permettant de couper le jeu de données en deux, pour l'entrainement et pour l'évaluation
from sklearn.preprocessing import StandardScaler # Pour standardiser les données
from sklearn.preprocessing import LabelEncoder # pour encoder mes labels de fleurs
import numpy as np
import matplotlib.pyplot as plt # pour la visualisation des données
import seaborn as sns # seaborn est aussi pour la visualisation, mais en plus évolué
import pandas as pd
import pickle # pour sauvegarder le scaler et l'encodeur

import datetime

datas = pd.read_csv("../dataset/gold/Gold_Spot_historical_data.csv")

# On convertis Date en datetime et on trie le dataset
datas['Date'] = pd.to_datetime(datas['Date'], utc=True)
datas = datas.sort_values('Date').reset_index(drop=True)

# On met Date en index du dataframe
datas.set_index('Date', inplace=True)

# print(datas.isna().sum()) # 0 NaN donc on est bons 

# Conversion des colonnes numériques
colonnes_numeriques = ['Open','High','Low','Close','Volume']
datas[colonnes_numeriques] = datas[colonnes_numeriques].apply(pd.to_numeric, errors='coerce')

"""
sns.set_theme(style="darkgrid")
plt.figure(figsize=(12,6))
sns.lineplot(x=datas.index, y=datas['Close'])
plt.title("Cours de l'or (USD)")
plt.xlabel("Date")
plt.ylabel("Prix de cloture")
plt.show()
"""

### On crée notre label cible

datas['Target'] = datas['Close'].shift(-1) - datas['Close']  # target = variation à 1 jours
datas = datas.dropna(subset=['Target'])      # la dernière ligne n'a pas de target

### Feature engineering

# Rendement
datas['Return'] = datas['Close'] .pct_change() 

# Différences intra-journée
datas['High_low_diff'] = datas['High'] - datas['Low']
datas['Open_close_diff'] = datas['Open'] - datas['Close'].shift(1)

# Volatilité quotidienne relative
datas['Volatility'] = (datas['High'] - datas['Low']) / datas['Open']

# Moyennes mobiles
datas['SMA_5'] = datas['Close'].shift(1).rolling(window=5).mean()
datas['SMA_10'] = datas['Close'].shift(1).rolling(window=10).mean()
datas['SMA_20'] = datas['Close'].shift(1).rolling(window=20).mean()

datas['Volume_MA_5'] = datas['Volume'].rolling(window=5).mean()

# Indicateur basique
datas['Close_minus_SMA10'] = datas['Close'].shift(1) - datas['SMA_10']

# On normalise en pourcentages
datas['Open_pct'] = datas['Open'].pct_change()
datas['High_pct'] = datas['High'].pct_change()
datas['Low_pct'] = datas['Low'].pct_change()
datas['Volume_pct'] = datas['Volume'].pct_change()

# on remplace les infinis par des NaN
datas.replace([np.inf, -np.inf], np.nan, inplace=True)
# On supprime les NaN
datas = datas.dropna()

print(datas.shape)
print(datas.columns)
print(datas.tail())

# les features que notre modèle va prendre en entrées
features = [
    'Open_pct','High_pct','Low_pct','Volume_pct',
    'Return','High_low_diff','Open_close_diff','Volatility',
    'SMA_5','SMA_10','SMA_20','Volume_MA_5','Close_minus_SMA10'
]

# Split entrainement et validation selon une date (pour éviter le data leakage)

split_date = '2023-12-31'
X_train = datas[datas.index <= split_date]
X_test = datas[datas.index > split_date]

y_train = X_train['Target']
y_test = X_test['Target']

# On garde les colonnes qui nous interessent
X_train = X_train[features]
X_test = X_test[features]

### Scaling (fit sur train uniquement)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# On normalise aussi sur la target
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler,f)

### Convertir en Tenseurs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

Xtr = torch.tensor(X_train_scaled, dtype=torch.float32)
Ytr = torch.tensor(y_train_scaled, dtype=torch.float32)
# Ytr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) # ajoute 1 dimension au vecteur 1D y_train
Xte = torch.tensor(X_test_scaled, dtype=torch.float32)
# Yte = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
Yte = torch.tensor(y_test_scaled, dtype=torch.float32)

batch_size = 64
train_ds = TensorDataset(Xtr, Ytr)
test_ds = TensorDataset(Xte, Yte)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

### Création de mon modèle de réseau de neuronne
## MLP : Multi Layer Perceptron
# Réseau de neuronnes classique entièrement connecté (linear -> activation -> Dropout) et prend un vecteur de features en entrée
# Mais ne sait pas capturer les dépendances temporelles longues
# Fonctionne si les features encapsulent des données historiques comme des moyennes mobiles ou un rendement...

# LSTM : Long Short-term Memory
# Type de Recurrent Neural Network concu pour traiter les séries temporelles
# Il prend en entrée une séquence et apprend à mémoriser l'ordre des données
# Mais il est plus complexe à entrainer et est plus lent

class SimpleMLP(nn.Module):
    def __init__(self, n_features, hidden_sizes=[128,64], dropout=0.1):
        super().__init__()
        layers = []

        in_dim = n_features # entrées : nombre de colonnes de mon dataset
        for h in hidden_sizes: # Parcours des deux couches du réseau (une couche de 128 et une couche de 64 neuronnes)
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout)) # évite le surapprentissage. Désactive un petit % (ici 10%) de neuronnes pour éviter l'over fitting
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # sortie : prix
        self.net = nn.Sequential(*layers) # Permet d'enchainer toutes les couches du réseau 
    def forward(self, x):
        return self.net(x)
    
model = SimpleMLP(n_features=Xtr.shape[1], hidden_sizes=[256,128,64], dropout=0.1).to(device)

# Loss / Optimizer / Scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Scheduler d'apprentissage automatique, il réduit le learning rate de moitié si la validation loss ne s'améliore pas pendant 5 epoch
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5) 

#### Entrainement
n_epochs = 60
best_val_loss = float('inf') # on initialise à l'infini pour considérer le premier modèle comme le meilleur.
patience = 20
patience_counter = 0

for epoch in range(1, n_epochs+1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train = sum(train_losses)/len(train_losses)
    
    # validation 
    model.eval()
    with torch.no_grad():
        val_losses = []
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_losses.append(criterion(pred, yb).item()) # on stocke toutes les losses pour en faire la moyenne plus tard
        avg_val = sum(val_losses)/len(val_losses)
    scheduler.step(avg_val)
    
    # Si la loss actuelle (moyenne) est meilleure que la loss précédente on sauve le modèle comme le meilleur
    if avg_val < best_val_loss - 1e-9: # 
        best_val_loss = avg_val
        torch.save(model.state_dict(), "best_mlp.pth")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train loss: {avg_train:.6f} | Val loss: {avg_val:.6f} | Best: {best_val_loss:.6f}")
    
    if patience_counter >= patience: # interrompt l'entrainement quand le modèle à appris tout ce qu'il pouvait
        print("Early stopping (patience dépassée)")
        break

# --- Charger le meilleur modèle et prédire sur le test ---
model.load_state_dict(torch.load("best_mlp.pth", map_location=device))
model.eval()
with torch.no_grad():
    # y_pred = model(Xte.to(device)).cpu().numpy().squeeze()
    y_pred_scaled = model(Xte.to(device)).detach().cpu().numpy()
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

# --- Évaluation (RMSE, MAPE, R2) ---
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score
rmse = root_mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: en moyenne le modèle se trompe de {rmse:.4f} dollars sur la cible.")
print(f"Test MAPE: en moyenne le modèle se trompe de {mape:.4%} par rapport à la cible.")
print(f"Test R2 (mesure statistique de la qualité du modèle (1 = parfait, 0 = nul)): {r2:.4f}")

plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test.values, label='True')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.title("True vs Predicted (Test Set)")
plt.show()

# --- Plot prédictions vs vérité (test period) ---
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,6))
# plt.plot(y_test.index, y_test.values, label='True Close', linewidth=1)
# plt.plot(y_test.index, y_pred, label='Predicted Close (MLP)', linewidth=1)
# plt.title("Prédiction Close t+1 - Test set (MLP)")
# plt.xlabel("Date")
# plt.ylabel("Prix (USD)")
# plt.legend()
# plt.grid(True)
# plt.show()

# --- Résidus / histogramme (optionnel) ---
# residuals = y_test.values - y_pred
# plt.figure(figsize=(10,4))
# plt.hist(residuals, bins=60)
# plt.title("Distribution des résidus (True - Pred)")
# plt.show()

"""
Epoch 001 | Train loss: 1975143.125000 | Val loss: 1891331.343750 | Best: 1891331.343750
Epoch 005 | Train loss: 297205.015032 | Val loss: 256681.591406 | Best: 256681.591406
Epoch 010 | Train loss: 55500.885532 | Val loss: 46945.061523 | Best: 46945.061523
Epoch 015 | Train loss: 29735.442964 | Val loss: 21757.758838 | Best: 21757.758838
Epoch 020 | Train loss: 19635.760136 | Val loss: 10328.207959 | Best: 10328.207959
Epoch 025 | Train loss: 14605.441802 | Val loss: 5720.341394 | Best: 5720.341394
Epoch 030 | Train loss: 13493.223552 | Val loss: 3732.376166 | Best: 3732.376166
Epoch 035 | Train loss: 11747.888165 | Val loss: 2897.340448 | Best: 2875.983490
Epoch 040 | Train loss: 11112.887875 | Val loss: 2061.563913 | Best: 2010.746777
Epoch 045 | Train loss: 11038.391997 | Val loss: 1435.382727 | Best: 1435.382727
Epoch 050 | Train loss: 10360.374271 | Val loss: 1140.134822 | Best: 1140.134822
Epoch 055 | Train loss: 9919.492432 | Val loss: 1056.656738 | Best: 1016.363094
Epoch 060 | Train loss: 9835.942142 | Val loss: 843.330461 | Best: 843.330461
Test RMSE: 29.0908
Test MAPE: 1.8368%
Test R2: 0.9983

Epoch 001 | Train loss: 1977042.553797 | Val loss: 1890390.687500 | Best: 1890390.687500
Epoch 005 | Train loss: 297590.019976 | Val loss: 237917.525781 | Best: 237917.525781
Epoch 010 | Train loss: 57595.287134 | Val loss: 40623.530176 | Best: 40623.530176
Epoch 015 | Train loss: 28921.329744 | Val loss: 18067.229150 | Best: 18067.229150
Epoch 020 | Train loss: 19805.565665 | Val loss: 9390.010083 | Best: 9390.010083
Epoch 025 | Train loss: 16019.981112 | Val loss: 6058.235229 | Best: 6058.235229
Epoch 030 | Train loss: 14342.842013 | Val loss: 4162.190875 | Best: 4162.190875
Epoch 035 | Train loss: 12582.355166 | Val loss: 3245.004352 | Best: 3245.004352
Epoch 040 | Train loss: 12430.935028 | Val loss: 2783.387830 | Best: 2783.387830
Epoch 045 | Train loss: 12167.065862 | Val loss: 2286.180676 | Best: 2286.180676
Epoch 050 | Train loss: 11092.009840 | Val loss: 1907.074643 | Best: 1907.074643
Epoch 055 | Train loss: 10221.820999 | Val loss: 1793.584109 | Best: 1777.308569
Epoch 060 | Train loss: 11125.601612 | Val loss: 1624.537234 | Best: 1596.240863
Test RMSE: 39.9911
Test MAPE: 2.5358%
Test R2: 0.9967


Epoch 001 | Train loss: 1975811.049051 | Val loss: 1887730.268750 | Best: 1887730.268750
Epoch 005 | Train loss: 258312.779866 | Val loss: 201547.596094 | Best: 201547.596094
Epoch 010 | Train loss: 57274.776948 | Val loss: 40868.601172 | Best: 40868.601172
Epoch 015 | Train loss: 28876.520619 | Val loss: 17693.542383 | Best: 17693.542383
Epoch 020 | Train loss: 19222.778778 | Val loss: 9424.087927 | Best: 9424.087927
Epoch 025 | Train loss: 15740.049409 | Val loss: 5798.369031 | Best: 5798.369031
Epoch 030 | Train loss: 13292.837452 | Val loss: 3905.504889 | Best: 3905.504889
Epoch 035 | Train loss: 12792.979727 | Val loss: 2929.038049 | Best: 2929.038049
Epoch 040 | Train loss: 12140.771206 | Val loss: 2371.238007 | Best: 2371.238007
Epoch 045 | Train loss: 11080.315436 | Val loss: 2033.507544 | Best: 2033.507544
Epoch 050 | Train loss: 10986.788129 | Val loss: 1845.522214 | Best: 1833.505835
Epoch 055 | Train loss: 10666.754209 | Val loss: 1611.781201 | Best: 1611.781201
Epoch 060 | Train loss: 10653.532882 | Val loss: 1698.694513 | Best: 1541.272723
Test RMSE: 39.3043
Test MAPE: 2.3813%
Test R2: 0.9968

Epoch 001 | Train loss: 1.008464 | Val loss: 7.086710 | Best: 7.086710
Epoch 005 | Train loss: 0.992412 | Val loss: 6.247897 | Best: 6.247897
Epoch 010 | Train loss: 0.967518 | Val loss: 7.079223 | Best: 6.247897
Epoch 015 | Train loss: 0.924331 | Val loss: 7.441147 | Best: 6.247897
Epoch 020 | Train loss: 0.888926 | Val loss: 6.810045 | Best: 6.247897
Epoch 025 | Train loss: 0.871313 | Val loss: 6.740522 | Best: 6.247897
Early stopping (patience dépassée)
Test RMSE: en moyenne le modèle se trompe de 35.4754 dollars sur la cible.
Test MAPE: en moyenne le modèle se trompe de 116.0347% par arpport à la cible.
Test R2 (mesure statistique de la qualité du modèle (1 = parfait, 0 = nul)): -0.0038
"""
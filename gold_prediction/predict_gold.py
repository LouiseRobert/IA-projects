import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
# from gold_price_prediction import SimpleMLP

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
    

datas = pd.read_csv("dataset/gold/Gold_Spot_historical_data.csv")

# on lui redonne les features qu'on lui avait ajouté 
### Feature engineering

# Rendement
datas['Return'] = datas['Close'].pct_change() 

# Différences intra-journée
datas['High_low_diff'] = datas['High'] - datas['Low']
datas['Open_close_diff'] = datas['Open'] - datas['Close']

# Volatilité quotidienne relative
datas['Volatility'] = (datas['High'] - datas['Low']) / datas['Open']

# Moyennes mobiles
datas['SMA_5'] = datas['Close'].rolling(window=5).mean()
datas['SMA_10'] = datas['Close'].rolling(window=10).mean()
datas['SMA_20'] = datas['Close'].rolling(window=20).mean()

datas['Volume_MA_5'] = datas['Volume'].rolling(window=5).mean()

# Indicateur basique
datas['Close_minus_SMA10'] = datas['Close'] - datas['SMA_10']

datas = datas.dropna()

features = [
    'Open','High','Low','Volume',
    'Return','High_low_diff','Open_close_diff','Volatility',
    'SMA_5','SMA_10','SMA_20','Volume_MA_5','Close_minus_SMA10'
]


# on récupère le dernier jour
input = datas.tail(1)

print(input)

# === 2️⃣ Charger le modèle entraîné ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_features = len(features)  # à remplacer par le même nombre qu’à l’entraînement !
model = SimpleMLP(n_features=n_features)
model.load_state_dict(torch.load("best_mlp.pth", map_location=device))
model.to(device)
model.eval()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

x = input[features].values
x_scaled = scaler.transform(x)
x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(x_tensor).cpu().item()

print(f"Prix de clôture prédit dans 5 jours : {y_pred:.2f} USD")

"""
Prix de clôture prédit dans 5 jours (31/10/2025): 4040.32 USD

Prix de clôture prédit dans 3 jours (04/11/2025): 3860.59 USD
"""
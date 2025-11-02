import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle

SEQ_LEN = 20  # même longueur qu'à l'entraînement

# ---------------------------------------------------------------------
# 1. Définir le modèle LSTM (même architecture qu'à l'entraînement)
# ---------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ---------------------------------------------------------------------
# 2. Charger les scalers et le modèle
# ---------------------------------------------------------------------
with open("scaler_x.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open("scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

n_features = 13  # nombre de features que tu as utilisé
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMModel(n_features).to(device)
model.load_state_dict(torch.load("ltsm_gold_model.pth", map_location=device))
model.eval()

# ---------------------------------------------------------------------
# 3. Charger les dernières données
# ---------------------------------------------------------------------
datas = pd.read_csv("../dataset/gold/Gold_Spot_historical_data.csv")
datas['Date'] = pd.to_datetime(datas['Date'], utc=True)
datas = datas.sort_values('Date').reset_index(drop=True)
datas.set_index('Date', inplace=True)

# Feature engineering (mêmes features que lors de l'entraînement)
datas['Return'] = datas['Close'].pct_change()
datas['High_low_diff'] = datas['High'] - datas['Low']
datas['Open_close_diff'] = datas['Open'] - datas['Close'].shift(1)
datas['Volatility'] = (datas['High'] - datas['Low']) / datas['Open']
datas['SMA_5'] = datas['Close'].shift(1).rolling(window=5).mean()
datas['SMA_10'] = datas['Close'].shift(1).rolling(window=10).mean()
datas['SMA_20'] = datas['Close'].shift(1).rolling(window=20).mean()
datas['Volume_MA_5'] = datas['Volume'].rolling(window=5).mean()
datas['Close_minus_SMA10'] = datas['Close'].shift(1) - datas['SMA_10']
datas['Open_pct'] = datas['Open'].pct_change()
datas['High_pct'] = datas['High'].pct_change()
datas['Low_pct'] = datas['Low'].pct_change()
datas['Volume_pct'] = datas['Volume'].pct_change()

datas.replace([np.inf, -np.inf], np.nan, inplace=True)
datas = datas.dropna()

# ---------------------------------------------------------------------
# 4. Préparer les séquences pour prédiction
# ---------------------------------------------------------------------
features = [
    'Open_pct','High_pct','Low_pct','Volume_pct',
    'Return','High_low_diff','Open_close_diff','Volatility',
    'SMA_5','SMA_10','SMA_20','Volume_MA_5','Close_minus_SMA10'
]

X = datas[features].values
X_scaled = scaler_X.transform(X)

# Créer la dernière séquence
X_last_seq = X_scaled[-SEQ_LEN:]          # prend les 10 derniers jours
X_last_seq = np.expand_dims(X_last_seq, axis=0)  # forme (1, seq_len, n_features)
X_last_seq_t = torch.tensor(X_last_seq, dtype=torch.float32).to(device)

# ---------------------------------------------------------------------
# 5. Prédiction
# ---------------------------------------------------------------------
with torch.no_grad():
    y_pred_scaled = model(X_last_seq_t).cpu().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

print(f"Prix prédit de l'or dans 5 jours : {y_pred[0,0]:.2f} dollars")

"""
Prix prédit de l'or dans 5 jours : 3863.78 dollars
"""

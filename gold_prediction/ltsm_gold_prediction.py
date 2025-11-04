import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import random
import pickle
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


SEQ_LEN = 20

# Fonction pour calcul de l'inducateur RSI
def rsi(close_series, window = 14):
    """
    Fonction de calcul du RSI.

    :@param close_series: pandas.Series, colone des prix de cloture
    :@param window: Int, Fenetre temporelle du calcul
    :@return: pandas.Series, Relative Strength Index
    """
    # calcule la différence d'un jour à l'autre
    delta = close_series.diff()

    # On isole les jours positifs et les jours négatifs
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()

    # ratio
    rs = gain / loss

    rsi = 100 - (100 / (1 + rs))

    return rsi

# ---------------------------------------------------------------------
# 1. Chargement et préparation des données
# ---------------------------------------------------------------------

datas = pd.read_csv("../dataset/gold/Gold_Spot_historical_data.csv")

# On convertis Date en datetime et on trie le dataset
datas['Date'] = pd.to_datetime(datas['Date'], utc=True)
datas = datas.sort_values('Date').reset_index(drop=True)

# On met Date en index du dataframe
datas.set_index('Date', inplace=True)

# Conversion des colonnes numériques
colonnes_numeriques = ['Open','High','Low','Close','Volume']
datas[colonnes_numeriques] = datas[colonnes_numeriques].apply(pd.to_numeric, errors='coerce')

### On crée notre label cible

datas['Target'] = datas['Close'].shift(-5)   # target = Prix de cloture à j+5
datas = datas.dropna(subset=['Target'])      # les dernière ligne n'a pas de target

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

datas['RSI'] = rsi(datas['Close'].shift(1))

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
    'SMA_5','SMA_10','SMA_20','Volume_MA_5','Close_minus_SMA10', 'RSI'
]

# Suppose que ta colonne cible s'appelle 'Target' (prix de clôture de J+5)
# features = [c for c in datas.columns if c != "Target"]
X = datas[features].values
y = datas["Target"].values.reshape(-1, 1)

# Normalisation
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ---------------------------------------------------------------------
# 2. Création des séquences temporelles
# ---------------------------------------------------------------------

def create_sequences(X, y, seq_len=10):
    """
    Transforme les données en séquences glissantes
    seq_len : longueur de la fenetre temporelle (10 jours)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)


X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len=SEQ_LEN)

# Split train/test
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Conversion en tenseurs PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

# ---------------------------------------------------------------------
# 3. Définition du modèle LSTM
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
        self.fc = nn.Linear(hidden_size, 1) # dernière couche qui transforme en une seule valeur

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # on garde la dernière sortie temporelle
        out = self.fc(out)
        return out

n_features = X_train.shape[2]
model = LSTMModel(n_features).to(device)

# ---------------------------------------------------------------------
# 4. Entraînement
# ---------------------------------------------------------------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()

    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        val_pred = model(X_test_t)
        val_loss = criterion(val_pred, y_test_t)
        print(f"Epoch {epoch:03d} | Train loss: {loss.item():.6f} | Val loss: {val_loss.item():.6f}")

# ---------------------------------------------------------------------
# 5. Évaluation
# ---------------------------------------------------------------------

model.eval()

with torch.no_grad():
    # Assure-toi que les données sont sur le même device que le modèle
    X_test_t = X_test_t.to(device)

    # Prédiction sur le même device
    y_pred_scaled = model(X_test_t).detach().cpu().numpy()

y_true = scaler_y.inverse_transform(y_test_t.cpu().numpy().reshape(-1, 1))

y_pred = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
r2 = r2_score(y_true, y_pred)

print(f"Test RMSE: en moyenne le modèle se trompe de {rmse:.4f} dollars sur la cible.")
print(f"Test MAPE: en moyenne le modèle se trompe de {mape:.2f}% par rapport à la cible.")
print(f"Test R2 (mesure statistique de la qualité du modèle (1 = parfait, 0 = nul)): {r2:.4f}")

# Sauvegarde des fichiers
# with open("ltsm_gold_model_rsi.pth", "wb") as f:
#     torch.save(model.state_dict(), f)

# with open("scaler_x_rsi.pkl", "wb") as f:
#     pickle.dump(scaler_X, f)

# with open("scaler_y_rsi.pkl", "wb") as f:
#     pickle.dump(scaler_y, f)


# ---------------------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------------------

plt.figure(figsize=(12,6))
plt.plot(y_true, label="True")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("LSTM - True vs Predicted (Test Set)")
plt.show()


"""
Epoch 000 | Train loss: 0.052288 | Val loss: 0.223242
Epoch 010 | Train loss: 0.016097 | Val loss: 0.055340
Epoch 020 | Train loss: 0.011174 | Val loss: 0.096760
Epoch 030 | Train loss: 0.008149 | Val loss: 0.047607
Epoch 040 | Train loss: 0.004626 | Val loss: 0.029734
Test RMSE: en moyenne le modèle se trompe de 184.7403 dollars sur la cible.
Test MAPE: en moyenne le modèle se trompe de 5.48% par rapport à la cible.
Test R2 (mesure statistique de la qualité du modèle (1 = parfait, 0 = nul)): 0.8981

Epoch 000 | Train loss: 0.111873 | Val loss: 0.347410
Epoch 010 | Train loss: 0.011651 | Val loss: 0.057181
Epoch 020 | Train loss: 0.010637 | Val loss: 0.092907
Epoch 030 | Train loss: 0.010043 | Val loss: 0.080076
Epoch 040 | Train loss: 0.007946 | Val loss: 0.039869
Epoch 050 | Train loss: 0.004690 | Val loss: 0.030475
Test RMSE: en moyenne le modèle se trompe de 154.0829 dollars sur la cible.
Test MAPE: en moyenne le modèle se trompe de 5.44% par rapport à la cible.
Test R2 (mesure statistique de la qualité du modèle (1 = parfait, 0 = nul)): 0.9291

Epoch 000 | Train loss: 0.025301 | Val loss: 0.150953
Epoch 010 | Train loss: 0.011391 | Val loss: 0.068377
Epoch 020 | Train loss: 0.007640 | Val loss: 0.054967
Epoch 030 | Train loss: 0.001710 | Val loss: 0.004787
Epoch 040 | Train loss: 0.000985 | Val loss: 0.009446
Epoch 050 | Train loss: 0.000520 | Val loss: 0.000909
Epoch 060 | Train loss: 0.000238 | Val loss: 0.000720
Epoch 070 | Train loss: 0.000190 | Val loss: 0.000683
Epoch 080 | Train loss: 0.000161 | Val loss: 0.001563
Epoch 090 | Train loss: 0.000139 | Val loss: 0.001492
Test RMSE: en moyenne le modèle se trompe de 192.7991 dollars sur la cible.
Test MAPE: en moyenne le modèle se trompe de 4.54% par rapport à la cible.
Test R2 (mesure statistique de la qualité du modèle (1 = parfait, 0 = nul)): 0.8890

Epoch 000 | Train loss: 0.051031 | Val loss: 0.218622
Epoch 010 | Train loss: 0.015985 | Val loss: 0.053582
Epoch 020 | Train loss: 0.011110 | Val loss: 0.096100
Epoch 030 | Train loss: 0.007915 | Val loss: 0.042599
Epoch 040 | Train loss: 0.003946 | Val loss: 0.024862
Epoch 050 | Train loss: 0.000831 | Val loss: 0.006128
Test RMSE: en moyenne le modèle se trompe de 173.3215 dollars sur la cible.
Test MAPE: en moyenne le modèle se trompe de 5.60% par rapport à la cible.
Test R2 (mesure statistique de la qualité du modèle (1 = parfait, 0 = nul)): 0.9103

Entrainement avec RSI:
Epoch 000 | Train loss: 0.035486 | Val loss: 0.182498
Epoch 010 | Train loss: 0.014199 | Val loss: 0.071479
Epoch 020 | Train loss: 0.011327 | Val loss: 0.095845
Epoch 030 | Train loss: 0.008330 | Val loss: 0.045628
Epoch 040 | Train loss: 0.001164 | Val loss: 0.001055
Test RMSE: en moyenne le modèle se trompe de 146.5278 dollars sur la cible.
Test MAPE: en moyenne le modèle se trompe de 3.65% par rapport à la cible.
Test R2 (mesure statistique de la qualité du modèle (1 = parfait, 0 = nul)): 0.9369

Epoch 000 | Train loss: 0.035486 | Val loss: 0.182500
Epoch 010 | Train loss: 0.014197 | Val loss: 0.071470
Epoch 020 | Train loss: 0.011325 | Val loss: 0.095836
Epoch 030 | Train loss: 0.008327 | Val loss: 0.045598
Epoch 040 | Train loss: 0.001171 | Val loss: 0.001083
Test RMSE: en moyenne le modèle se trompe de 148.9133 dollars sur la cible.
Test MAPE: en moyenne le modèle se trompe de 3.77% par rapport à la cible.
Test R2 (mesure statistique de la qualité du modèle (1 = parfait, 0 = nul)): 0.9348
"""
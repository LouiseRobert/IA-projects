import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import random
import pickle
from datetime import datetime
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

SEQ_LEN = 120
EPOCH = 10

class WindowDataset(Dataset):
    def __init__(self, X, y, seq_len, step=1, dtype=np.float32):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.step = step
        self.starts = np.arange(0, len(X) - seq_len, step)
        self.dtype = dtype

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x = self.X[s:s+self.seq_len].astype(self.dtype)
        y = self.y[s + self.seq_len].astype(self.dtype)
        return torch.from_numpy(x), torch.tensor(y)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 1. Chargement et préparation des données
    # ---------------------------------------------------------------------
    print(f"{datetime.now()} : Chargement et préparation des données.")


    df = pd.read_csv("../dataset/bitcoin/btcusd_1-min_data.csv")

    # On convertis le timestamp en datetime et on trie le dataset
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    dg = df.sort_values('Timestamp').reset_index(drop=True)

    # On met Timestamp en index du dataframe
    df.set_index('Timestamp', inplace=True)

    datas = df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()  # supprime les bougies incomplètes

    print(f"Données resamplées : {datas.shape[0]} lignes")

    # Conversion des colonnes numériques
    colonnes_numeriques = ['Open','High','Low','Close','Volume']
    datas[colonnes_numeriques] = datas[colonnes_numeriques].apply(pd.to_numeric, errors='coerce')

    ### On crée notre label cible

    datas['Target'] = datas['Close'].shift(-120)   # target = Prix de cloture à j+5 (120 heures en 5 jour)
    datas = datas.dropna(subset=['Target'])      # les dernière ligne n'a pas de target

    ### Feature engineering

    # Rendement
    datas['Return'] = datas['Close'].pct_change() 

    # Différences intra-journée
    datas['High_low_diff'] = datas['High'] - datas['Low']
    datas['Open_close_diff'] = datas['Open'] - datas['Close'].shift(1)

    # Volatilité quotidienne relative
    datas['Volatility'] = (datas['High'] - datas['Low']) / datas['Open']

    # Moyennes mobiles
    datas['SMA_5'] = datas['Close'].shift(1).rolling(window=24).mean()
    datas['SMA_10'] = datas['Close'].shift(1).rolling(window=48).mean()
    datas['SMA_20'] = datas['Close'].shift(1).rolling(window=96).mean()

    datas['Volume_MA_5'] = datas['Volume'].rolling(window=5).mean()

    # Indicateur basique
    datas['Close_minus_SMA10'] = datas['Close'].shift(1) - datas['SMA_10']

    # datas['RSI'] = rsi(datas['Close'].shift(1))

    # datas['MACD'] = macd(datas['Close'].shift(1))
    # datas['MACD_signal'] = datas['MACD'].ewm(span=9, adjust=False).mean()
    # datas['MACD_hist'] = datas['MACD'] - datas['MACD_signal']

    # Bollinger bands
    # bol = bollinger_bands(datas['Close'].shift(1))
    # datas['Bollinger_lower'] = bol['Bollinger_lower']
    # datas['Bollinger_upper'] = bol['Bollinger_upper']

    # Ratio de position du prix dans les bandes
    # datas['Bollinger_pos'] = (datas['Close'].shift(1) - datas['Bollinger_lower']) / (datas['Bollinger_upper'] - datas['Bollinger_lower'])

    # On normalise en pourcentages
    # datas['Open_pct'] = datas['Open'].pct_change()
    # datas['High_pct'] = datas['High'].pct_change()
    # datas['Low_pct'] = datas['Low'].pct_change()
    # datas['Volume_pct'] = datas['Volume'].pct_change()

    # on remplace les infinis par des NaN
    datas.replace([np.inf, -np.inf], np.nan, inplace=True)
    # On supprime les NaN
    datas = datas.dropna()

    print(datas.shape)
    print(datas.columns)
    print(datas.tail())

    # les features que notre modèle va prendre en entrées
    features = [
        'Return','High_low_diff','Open_close_diff','Volatility',
        'SMA_5','SMA_10','SMA_20','Volume_MA_5','Close_minus_SMA10'
    ]

   # On découpe d'abord le dataset en train/test AVANT le scaling (important)
    split = int(0.8 * len(datas))
    train_df = datas.iloc[:split]
    test_df  = datas.iloc[split:]

    # Séparation X / y
    X_train = train_df[features].values
    y_train = train_df["Target"].values.reshape(-1, 1)

    X_test  = test_df[features].values
    y_test  = test_df["Target"].values.reshape(-1, 1)

    # ---------------------------------------------------------------------
    # 3. Normalisation SANS fuite d'information
    # ---------------------------------------------------------------------
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    # ---------------------------------------------------------------------
    # 4. Création des séquences temporelles
    # ---------------------------------------------------------------------
    print(f"{datetime.now()} : Création des séquences temporelles.")

    train_ds = WindowDataset(X_train_scaled, y_train_scaled, seq_len=SEQ_LEN, step=1)
    test_ds  = WindowDataset(X_test_scaled, y_test_scaled, seq_len=SEQ_LEN, step=1)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    print(f"Train sequences : {len(train_loader.dataset)}, Test sequences : {len(test_loader.dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------------------------------------------------
    # 3. Définition du modèle LSTM
    # ---------------------------------------------------------------------
    print(f"{datetime.now()} : Définition du modèle.")

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

    sample_x, _ = train_ds[0]
    n_features = sample_x.shape[1]

    model = LSTMModel(n_features).to(device)

    # ---------------------------------------------------------------------
    # 4. Entrainement
    # ---------------------------------------------------------------------
    print(f"{datetime.now()} : Entrainement.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(EPOCH):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCH} - Train loss: {total_loss/len(train_loader):.6f}")

    # ---------------------------------------------------------------------
    # 5. Évaluation
    # ---------------------------------------------------------------------
    print(f"{datetime.now()} : Évaluation du modèle.")

    model.eval()

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Prédiction
            y_pred_batch = model(X_batch)

            # Stockage CPU
            y_true_all.append(y_batch.cpu().numpy())
            y_pred_all.append(y_pred_batch.cpu().numpy())

    # Concaténation de tous les batches
    y_true_all = np.concatenate(y_true_all).reshape(-1, 1)
    y_pred_all = np.concatenate(y_pred_all).reshape(-1, 1)

    # Inverse scaling
    y_true = scaler_y.inverse_transform(y_true_all)
    y_pred = scaler_y.inverse_transform(y_pred_all)

    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"Test RMSE : {rmse:.4f} dollars en moyenne d'erreur.")
    print(f"Test MAPE : {mape:.2f}% d'erreur moyenne relative.")
    print(f"Test R²   : {r2:.4f} (1 = parfait, 0 = nul).")

    # ---------------------------------------------------------------------
    # 6. Sauvegarde du modèle et des scalers
    # ---------------------------------------------------------------------
    print(f"{datetime.now()} : Sauvegarde du modèle et des scalers.")

    torch.save(model.state_dict(), "bitcoin_predicter.pth")

    with open("scaler_x.pkl", "wb") as f:
        pickle.dump(scaler_X, f)

    with open("scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    # ---------------------------------------------------------------------
    # 7. Visualisation
    # ---------------------------------------------------------------------
    print(f"{datetime.now()} : Visualisation des résultats.")

    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label="Valeur réelle", linewidth=1.2)
    plt.plot(y_pred, label="Prédiction LSTM", linewidth=1.2, alpha=0.8)
    plt.title("LSTM - Valeurs réelles vs Prédictions (jeu de test)")
    plt.xlabel("Temps (index test)")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Zoom sur les 1000 derniers points pour voir le détail local
    plt.figure(figsize=(14, 6))
    plt.plot(y_true[-1000:], label="Valeur réelle (zoom)", linewidth=1.2)
    plt.plot(y_pred[-1000:], label="Prédiction LSTM (zoom)", linewidth=1.2, alpha=0.8)
    plt.title("Zoom sur les 1000 dernières prédictions")
    plt.xlabel("Temps (index test)")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

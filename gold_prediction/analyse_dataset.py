import numpy as np
import pandas as pd
import torch
import random
    
    
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

# RSI
window = 14
delta = datas['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
rs = gain / loss
datas['RSI'] = 100 - (100 / (1 + rs))

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
print(datas['RSI'].tail())

print(type(datas['Close']))
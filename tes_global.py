import numpy as np, os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

MODEL_FILE = "global_model.h5"   
DATA_FILE = "ina.csv"            
FIXED_INPUT_DIM = 3
PERCENTILE = 95

def load_and_prep(path):
    df = pd.read_csv(path).dropna()
    df_num = df.select_dtypes(include=[np.number])
    X = df_num.values.astype(np.float32)
    # pad/truncate
    if X.shape[1] < FIXED_INPUT_DIM:
        pad = FIXED_INPUT_DIM - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad), dtype=np.float32)])
    elif X.shape[1] > FIXED_INPUT_DIM:
        X = X[:, :FIXED_INPUT_DIM]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    return Xs

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"{MODEL_FILE} not found. Run server.py first.")

print(f" Loading global model: {MODEL_FILE}")
model = tf.keras.models.load_model(MODEL_FILE, compile=False)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found.")

X = load_and_prep(DATA_FILE)
print(f" Loaded {DATA_FILE}: {X.shape[0]} samples, {X.shape[1]} features\n")

recon = model.predict(X, verbose=0)
mse = np.mean(np.square(X - recon), axis=1)

mean_mse = np.nanmean(mse)
p95 = np.nanpercentile(mse, PERCENTILE)

anoms = (mse > p95).sum()
print(f" Mean reconstruction MSE  : {mean_mse:.6f}")
print(f" {PERCENTILE}th percentile MSE      : {p95:.6f}\n")
print(f" Detected anomalies: {anoms}/{len(mse)} ({anoms/len(mse)*100:.2f}%)")

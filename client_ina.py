import os, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


DATA_FILE = "ina.csv"           
OUT_WEIGHTS = "client3_weights.pkl"
FIXED_INPUT_DIM = 3
EPOCHS = 5
BATCH = 32


def load_and_prep(path):
    df = pd.read_csv(path).dropna()
    
    df_num = df.select_dtypes(include=[np.number])
    X = df_num.values.astype(np.float32)
    
    if X.shape[1] < FIXED_INPUT_DIM:
        pad = FIXED_INPUT_DIM - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], pad), dtype=np.float32)])
    elif X.shape[1] > FIXED_INPUT_DIM:
        X = X[:, :FIXED_INPUT_DIM]
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    return Xs, scaler

def run_client():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found in folder.")
    print(f" Loading {DATA_FILE}")
    X, scaler = load_and_prep(DATA_FILE)
    print(f" → {X.shape[0]} samples, {X.shape[1]} features after prep")

    
    model = load_model("base_model.h5", compile=False)
    model.compile(optimizer="adam", loss="mse")

    
    model.fit(X, X, epochs=EPOCHS, batch_size=BATCH, verbose=0)
    print("✔ Local training done")

    
    payload = {"weights": model.get_weights(), "samples": X.shape[0]}
    with open(OUT_WEIGHTS, "wb") as f:
        pickle.dump(payload, f)
    print(f" Saved local weights → {OUT_WEIGHTS}")

if __name__ == "__main__":
    run_client()

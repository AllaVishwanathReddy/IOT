# client_template.py
import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


DATA_FILE = "dht.csv"  # change per client

df = pd.read_csv(DATA_FILE).dropna()
X = df.select_dtypes(include=["number"]).values

# Ensure all have same column count
FIXED_INPUT_DIM = 4
if X.shape[1] < FIXED_INPUT_DIM:
    pad = FIXED_INPUT_DIM - X.shape[1]
    X = np.hstack([X, np.zeros((X.shape[0], pad))])
elif X.shape[1] > FIXED_INPUT_DIM:
    X = X[:, :FIXED_INPUT_DIM]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Define lightweight autoencoder for anomaly detection

input_dim = 4  # must match FIXED_INPUT_DIM
encoding_dim = 3  # smaller hidden layer

autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(encoding_dim, activation="relu"),
    tf.keras.layers.Dense(input_dim, activation="sigmoid")
])
autoencoder.compile(optimizer="adam", loss="mse")


print(f"✅ Local model ready for {DATA_FILE}")


class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return autoencoder.get_weights()

    def fit(self, parameters, config):
        autoencoder.set_weights(parameters)
        autoencoder.fit(X_scaled, X_scaled, epochs=5, batch_size=16, verbose=0)
        print(f"🔁 Retrained locally for {DATA_FILE}")
        return autoencoder.get_weights(), len(X_scaled), {}

    def evaluate(self, parameters, config):
        autoencoder.set_weights(parameters)
        loss = autoencoder.evaluate(X_scaled, X_scaled, verbose=0)
        print(f"📊 Local loss for {DATA_FILE}: {loss:.4f}")
        return loss, len(X_scaled), {"loss": loss}


SERVER = "127.0.0.1:8080"
print(f"🚀 Connecting to FL server for {DATA_FILE}")
fl.client.start_numpy_client(server_address=SERVER, client=FLClient())


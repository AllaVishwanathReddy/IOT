import tensorflow as tf
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Input 
from tensorflow.keras.optimizers import Adam 

FIXED_INPUT_DIM = 3
ENCODING_DIM = 2
OUTFILE = "base_model.h5"

def build_autoencoder(input_dim=FIXED_INPUT_DIM, encoding_dim=ENCODING_DIM):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(encoding_dim, activation="relu", name="bottleneck"),
        Dense(16, activation="relu"),
        Dense(input_dim, activation="linear")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

if __name__ == "__main__":
    m = build_autoencoder()
    m.save(OUTFILE)
    print(f" Saved base model to {OUTFILE}")

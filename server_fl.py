import pickle, os
import numpy as np
from tensorflow.keras.models import load_model

CLIENT_FILES = ["client1_weights.pkl", "client2_weights.pkl","client3_weights.pkl"]
NUM_ROUNDS = 3
BASE_MODEL_FILE = "base_model.h5"
OUT_GLOBAL = "global_model.h5"

def fedavg(weight_list, sample_counts):
    total = sum(sample_counts)
    avg_weights = []
    for layer_i in range(len(weight_list[0])):
        weighted_sum = sum((sample_counts[c] / total) * weight_list[c][layer_i]
                           for c in range(len(weight_list)))
        avg_weights.append(weighted_sum)
    return avg_weights

def run_server(num_rounds=NUM_ROUNDS):
    if not os.path.exists(BASE_MODEL_FILE):
        raise FileNotFoundError(f"{BASE_MODEL_FILE} not found. Run basemodel.py first.")
    global_model = load_model(BASE_MODEL_FILE, compile=False)
    global_model.compile(optimizer="adam", loss="mse")
    print(f" Server: loaded base model from {BASE_MODEL_FILE}")

    for r in range(1, num_rounds + 1):
        print(f"\n=== ROUND {r} ===")
        
        weights_list = []
        samples = []
        missing = [f for f in CLIENT_FILES if not os.path.exists(f)]
        if missing:
            print(" Missing client weight files:", missing)
            print("→ Run all client scripts to generate weight files, then rerun server.")
            return
        for f in CLIENT_FILES:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            weights_list.append(data["weights"])
            samples.append(int(data["samples"]))
            print(f" Server: loaded {f} (samples={data['samples']})")

        
        new_weights = fedavg(weights_list, samples)
        global_model.set_weights(new_weights)
        print(" ✔ Server: aggregated global weights (FedAvg)")

    global_model.save(OUT_GLOBAL)
    print(f"\n Final global model saved as {OUT_GLOBAL}")

if __name__ == "__main__":
    run_server()

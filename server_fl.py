# server_fl.py
import flwr as fl
import tensorflow as tf
import pickle

# ======================================================
# 🚀 FEDERATED SERVER CONFIGURATION
# ======================================================

NUM_ROUNDS = 3
SERVER_ADDRESS = "127.0.0.1:8080"  # for local, also works with 127.0.0.1

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_available_clients=3,
)

print(f"🚀 Starting Federated Learning Server at {SERVER_ADDRESS}")
print("============================================================")

# Start the Flower server and capture the aggregated final weights
history, aggregated_parameters = fl.server.start_server(
    server_address=SERVER_ADDRESS,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    return_final_parameters=True,  # ✅ key flag to capture weights
)

print("✅ Federated training completed successfully.")
print("============================================================")

# ======================================================
# 📊 DISPLAY TRAINING METRICS
# ======================================================

if hasattr(history, "metrics_distributed_fit"):
    print("\n📈 Training metrics (fit):")
    print(history.metrics_distributed_fit)

if hasattr(history, "metrics_distributed"):
    print("\n📊 Evaluation metrics:")
    print(history.metrics_distributed)

# ======================================================
# 💾 SAVE GLOBAL MODEL (REAL FEDERATED MODEL)
# ======================================================

print("\n💾 Saving aggregated global model...")

if aggregated_parameters:
    try:
        # Convert Flower Parameters to NumPy arrays
        ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

        # Build same architecture as clients
        input_dim = 4
        encoding_dim = 3
        autoencoder_global = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(encoding_dim, activation="relu"),
            tf.keras.layers.Dense(input_dim, activation="sigmoid")
        ])
        autoencoder_global.compile(optimizer="adam", loss="mse")

        # Set global weights and save
        autoencoder_global.set_weights(ndarrays)
        with open("global_model.pkl", "wb") as f:
            pickle.dump(autoencoder_global.get_weights(), f)
        print("✅ Global model saved as 'global_model.pkl'")

    except Exception as e:
        print(f"⚠️ Error while saving model: {e}")
else:
    print("⚠️ No aggregated parameters found after training.")

print("🏁 Server shutdown complete.")

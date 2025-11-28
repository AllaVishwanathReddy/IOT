import time
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO
import dht11
import subprocess

# -----------------------------
# Load Global Model
# -----------------------------
print("🔍 Loading global model...")
model = tf.keras.models.load_model("global_model.h5", compile=False)
model.compile(optimizer="adam", loss="mse")
print("✅ Model loaded")

# -----------------------------
# Initialize DHT11 Sensor
# -----------------------------
GPIO.setmode(GPIO.BOARD)
myDHT = dht11.DHT11(pin=11)

# -----------------------------
# Preprocessing (same as training)
# -----------------------------
def preprocess(arr):
    arr = np.array(arr, dtype=np.float32)
    return arr.reshape(1, -1)

# -----------------------------
# Network Latency Function
# -----------------------------
def get_ping():
    try:
        output = subprocess.check_output("ping -c 1 google.com", shell=True).decode()
        ms = float(output.split("time=")[1].split(" ")[0])
        return ms
    except:
        return 999.0   # extreme anomaly value

# -----------------------------
# Real-Time Loop
# -----------------------------
print("\n🚀 Starting REAL-TIME anomaly detection...")
print("Press CTRL+C to stop.\n")

# Threshold tuned according to test results (DHT + Network)
THRESHOLD = 1.8   # You can adjust this based on your model results

while True:
    # Read DHT sensor
    result = myDHT.read()
    humidity = result.humidity
    temp = result.temperature

    if humidity is None or temp is None:
        print("⚠ DHT read failure... retrying")
        time.sleep(2)
        continue

    # Read network latency
    network_ms = get_ping()

    # Feature vector (IMPORTANT: same order used for training)
    features = [temp, humidity, network_ms]

    # Preprocess
    X = preprocess(features)

    # Predict reconstruction
    y = model.predict(X, verbose=0)
    mse = np.mean((X - y) ** 2)

    is_anomaly = mse > THRESHOLD

    print("-" * 50)
    print(f"🌡 Temp       : {temp:.2f} °C")
    print(f"💧 Humidity   : {humidity:.2f} %")
    print(f"🌐 Latency    : {network_ms:.2f} ms")
    print(f"📉 MSE        : {mse:.6f}")

    if is_anomaly:
        print("🚨 ANOMALY DETECTED!")
    else:
        print("✅ Normal")

    time.sleep(2)

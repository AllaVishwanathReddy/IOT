# 🌐 Federated Learning–Based Multi-Sensor Anomaly Detection in IoT

A privacy-preserving IoT anomaly detection system using **Federated Learning (FL)** across multiple Raspberry Pi edge devices and heterogeneous sensors (Environmental, Electrical, Network).

Developed as part of the *Internet of Things course (2025–26), BITS Pilani Hyderabad*.

---

## 📌 Overview

Traditional IoT anomaly detection requires centralized sensor data collection, which introduces privacy risks, bandwidth overhead, and latency.  
This project implements a **distributed federated learning framework** where multiple IoT edge devices collaboratively train an anomaly detection model **without sharing raw sensor data**.

Each Raspberry Pi client trains a local Autoencoder on its own sensor data and shares only model weights with a central server, which aggregates them using **Federated Averaging (FedAvg)** to produce a global anomaly detection model.

---

## 🎯 Objectives

- Detect anomalies in IoT sensor data in real time  
- Preserve data privacy by avoiding raw data transfer  
- Reduce network bandwidth usage  
- Enable distributed edge intelligence on Raspberry Pi nodes  
- Learn unified normal behavior across heterogeneous sensors  

---

## 🧰 System Architecture

**Topology:** Star (Central Server + Multiple Edge Clients)

- Edge Clients: Raspberry Pi 4 with sensors  
- Central Server: Aggregation and global model training  
- Communication: Socket-based / Federated Learning rounds  

Workflow:
Sensors → Raspberry Pi Client → Local Autoencoder Training
→ Model Weights Upload → Central Server (FedAvg)
→ Global Model → Clients

---

## 📡 Sensors & Data Sources

The system integrates three heterogeneous IoT data streams:

- 🌡️ Environmental: DHT22 (Temperature, Humidity)
- ⚡ Electrical: INA219 (Voltage, Current, Power)
- 🌐 Network: Packet traffic logs (IP, Packet Size)

Data preprocessing:
- Timestamping
- Min-Max normalization [0,1]
- Feature alignment across modalities

---

## 🧠 Machine Learning Approach

### Autoencoder-Based Anomaly Detection

Each client trains an unsupervised Autoencoder:

- Encoder → latent representation
- Decoder → reconstruction
- Loss → Mean Squared Error (MSE)

**Anomaly logic:**  
High reconstruction error ⇒ anomalous behavior

---

## 🤝 Federated Learning Strategy

Instead of sharing data:

1. Server initializes global model  
2. Clients train locally (1–5 epochs)  
3. Clients upload weights (.pkl)  
4. Server aggregates using **FedAvg**  
5. Updated global model redistributed  
6. Repeat until convergence  

---

## 💻 Tech Stack

**Hardware**
- Raspberry Pi 4 Model B
- DHT22 sensor
- INA219 sensor

**Software**
- Python 3.9
- TensorFlow / Keras
- Flower (FL framework)
- NumPy, Pandas
- Socket communication

---

## 📊 Experimental Results

The global federated model successfully learned normal behavior across all sensor domains.

| Sensor | Mean MSE | 95th Percentile | Anomalies Detected |
|--------|---------|----------------|-------------------|
| DHT22 | 0.5158 | 1.3801 | 2.17% |
| INA219 | 0.7279 | 1.6512 | 5% |
| Network | 0.6606 | 4.7117 | 5% |

Key observations:

- Environmental data highly predictable → lowest error  
- Network traffic highly variable → highest threshold  
- Successful FL convergence across heterogeneous data  

---

## 🚀 Key Contributions

- Multi-sensor federated IoT anomaly detection system  
- Privacy-preserving distributed ML on edge devices  
- Cross-domain anomaly detection (environmental, electrical, network)  
- Practical Raspberry Pi deployment architecture  
- Reduced bandwidth vs centralized IoT ML  

---

## 📈 Results & Impact

- Detects anomalies across heterogeneous IoT domains  
- Preserves edge data privacy  
- Scales across distributed IoT networks  
- Reduces latency and cloud dependence  

---

## 🔮 Future Work

- Full multi-node real-time deployment  
- Secure aggregation in FL rounds  
- Additional sensors (gas, motion, light)  
- Real-time anomaly dashboard  
- Edge optimization & model compression  

---

## 👨‍💻 Authors

**Vishwanath Reddy Alla**  
BITS Pilani Hyderabad  
GitHub: https://github.com/AllaVishwanathReddy  

**Gadupudi Sri Surya**
BITS Pilani Hyderabad  
GitHub: https://github.com/AllaVishwanathReddy 

Group-16 IoT Project

---

## 📚 References

1. Bonawitz et al., *Towards Federated Learning at Scale*, SysML 2019  
2. Liu et al., *Deep Anomaly Detection for IIoT*, IEEE IoT Journal  
3. Dritsas & Trigka, *Federated Learning for IoT Survey*, 2025  

---



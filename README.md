# 🎙️ Speech Emotion Recognition (SER)

### Hybrid CNN-BiLSTM Architecture with Global Attention Mechanism

🔗 **Live Demo:**
[https://dmxgbcbd7rxxdtrpelajyq.streamlit.app/](https://dmxgbcbd7rxxdtrpelajyq.streamlit.app/)

## 📌 Project Overview

This project implements a **state-of-the-art Speech Emotion Recognition (SER) system**.
It combines:

* **CNN (Convolutional Neural Network)** → for spatial feature extraction
* **Bidirectional LSTM (BiLSTM)** → for temporal sequence learning
* **Attention Mechanism** → for focusing on emotionally relevant parts

The model predicts **8 different emotions** from raw audio and achieves **98.2% accuracy** on the RAVDESS dataset.

---

## ✨ Key Features

* 🔹 **Hybrid Deep Learning Architecture**
  Combines CNN + BiLSTM + Custom Attention Layer

* 🔹 **Rich Feature Extraction (174-Dimensional Vector)**

  * MFCCs
  * Mel-Spectrogram
  * Chroma Features
  * Zero Crossing Rate (ZCR)
  * Root Mean Square Energy (RMS)

* 🔹 **Data Augmentation (5x Expansion)**

  * Noise Injection
  * Pitch Shifting
  * Time Stretching

* 🔹 **Interactive Web App**
  Built with **Streamlit** for real-time emotion prediction

---

## 🧠 Model Architecture

The model processes a **(128 × 174)** feature matrix through three main stages:

### 1️⃣ CNN Block

* Learns **local spectral patterns**
* Captures frequency textures

### 2️⃣ BiLSTM Block

* Models **temporal dependencies**
* Processes sequences in **forward + backward direction**

### 3️⃣ Attention Mechanism

* Assigns importance weights to time steps
* Focuses on **emotionally rich segments**
* Reduces impact of silence/noise

---

## 📊 Dataset: RAVDESS

* 🎭 **Dataset Name:** Ryerson Audio-Visual Database of Emotional Speech and Song

* 👥 **Actors:** 24 (12 male, 12 female)

* 🎯 **Emotions (8 classes):**

  * Neutral
  * Calm
  * Happy
  * Sad
  * Angry
  * Fearful
  * Disgust
  * Surprised

* 📈 **Data Augmentation:**

  * Original Samples: 2,880
  * Augmented Samples: 14,400

---

## 🛠️ Tech Stack

| Category         | Tools                       |
| ---------------- | --------------------------- |
| Language         | Python                      |
| Deep Learning    | TensorFlow, Keras           |
| Audio Processing | Librosa                     |
| Data Science     | NumPy, Pandas, Scikit-learn |
| Visualization    | Matplotlib                  |
| Deployment       | Streamlit Cloud             |

---

## 📂 Repository Structure

```
├── app.py                # Streamlit Web Application
├── scaler.pkl            # StandardScaler for normalization
├── label_map.json        # Label mapping (int ↔ emotion)
├── ser_model.weights.h5  # Trained model weights
├── packages.txt          # System dependencies (Streamlit Cloud)
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python version
└── Notebooks/
    ├── 1_Feature_Extraction.ipynb  # Feature engineering & augmentation
    └── 2_Model_Training.ipynb      # Model training & evaluation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-link>
cd Speech-Emotion-Recognizer-Project
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📈 Results

| Metric                          | Score |
| ------------------------------- | ----- |
| Test Accuracy                   | 98.2% |
| F1-Score (Weighted)             | 0.98  |
| UAR (Unweighted Average Recall) | 0.98  |

---

## 📜 References

* Livingstone, S. R. (2018). *The RAVDESS Dataset*. PLOS ONE
* Roy, C., et al. (2025). *Stacked CNN for SER*. Nature Scientific Reports
* Bhanbhro, J. (2025). *Attention-Enhanced CNN-LSTM Models*. Signals


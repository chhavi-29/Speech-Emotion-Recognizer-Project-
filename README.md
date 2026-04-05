

# 🎙️ Speech Emotion Recognition (SER)

### Hybrid CNN-BiLSTM Architecture with Global Attention Mechanism

https://dmxgbcbd7rxxdtrpelajyq.streamlit.app/

## 📌 Project Overview

This project implements a state-of-the-art **Speech Emotion Recognition (SER)** system. By fusing spatial feature extraction (CNN) with temporal sequence modeling (Bidirectional LSTM) and an Attention mechanism, the model identifies 8 distinct emotions from raw audio with **98.2% accuracy** on the RAVDESS dataset.

### ✨ Key Features

  * **Hybrid Deep Learning:** Combines CNN, BiLSTM, and a Custom Attention Layer.
  * **Rich Feature Fusion:** Extracts a 174-dimensional vector (MFCCs, Mel-Spectrogram, Chroma, ZCR, RMS).
  * **Data Augmentation:** 5x dataset expansion using Noise Injection, Pitch Shifting, and Time Stretching.
  * **Interactive Dashboard:** Live inference via a Streamlit web interface.

-----

## 🧠 Model Architecture

The model processes a **(128, 174)** feature matrix through three specialized stages:

1.  **CNN Block:** Learns local spectral patterns and frequency textures.
2.  **BiLSTM Block:** Captures long-term temporal dependencies in both forward and backward directions.
3.  **Attention Mechanism:** Assigns importance weights to specific timesteps, focusing on emotional peaks while ignoring background silence.

-----

## 📊 Dataset: RAVDESS

We utilize the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**.

  * **Emotions:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.
  * **Actors:** 24 professional actors (12 male, 12 female).
  * **Augmentation:** Original 2,880 samples expanded to **14,400** to ensure high generalization.

-----

## 🛠️ Tech Stack

  * **Language:** Python
  * **Deep Learning:** TensorFlow, Keras
  * **Audio Processing:** Librosa
  * **Deployment:** Streamlit Cloud
  * **Data Science:** NumPy, Pandas, Scikit-learn, Matplotlib

-----

## 📂 Repository Structure

```text
├── app.py                # Streamlit Web Application
├── scaler.pkl            # Trained StandardScaler (for normalization)
├── label_map.json        # Mapping between Integers and Emotion Labels
├── ser_model.weights.h5  # Trained Model Weights
├── packages.txt          # System-level dependencies for Streamlit Cloud
├── requirements.txt      # Python library dependencies
├── runtime.txt           # Python version specification
└── Notebooks/
    ├── 1_Feature_Extraction.ipynb  # Data Engineering & Augmentation
    └── 2_Model_Training.ipynb      # Model Building & Evaluation
```

-----

## 🚀 Getting Started

### 1\. Clone the repository

```bash
git clone <>
cd Speech-Emotion-Recognizer-Project-
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

### 3\. Run the App locally

```bash
streamlit run app.py
```

-----

## 📈 Results

The model achieves near-perfect performance on studio-quality data:

  * **Test Accuracy:** 98.2%
  * **F1-Score (Weighted):** 0.98
  * **UAR (Unweighted Average Recall):** 0.98

-----

## 📜 References

1.  **Livingstone, S. R.** (2018). *The RAVDESS Dataset.* PLOS ONE.
2.  **Roy, C., et al.** (2025). *Stacked CNN for SER.* Nature Scientific Reports.
3.  **Bhanbhro, J.** (2025). *Attention-Enhanced CNN-LSTM Models.* Signals.

-----




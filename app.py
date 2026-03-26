import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import json

# --- 1. YOUR EXACT ATTENTION LAYER FROM NOTEBOOK 2 ---
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name='W', shape=(d, d),
                                  initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='b', shape=(d,),
                                  initializer='zeros',          trainable=True)
        self.u = self.add_weight(name='u', shape=(d, 1),
                                  initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, x):
        # Score each timestep
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        # Weighted sum
        context = tf.reduce_sum(x * attention_weights, axis=1)
        # During prediction, we only return the context
        return context

    def get_config(self):
        return super().get_config()

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_ser_assets():
    # Load model with your exact AttentionLayer
    model = tf.keras.models.load_model('ser_model_final.keras', 
                                      custom_objects={'AttentionLayer': AttentionLayer})
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_map.json', 'r') as f:
        labels = json.load(f)['int_to_label']
    return model, scaler, labels

# --- 3. FEATURE EXTRACTION (Matches Notebook 1) ---
def extract_live_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Standard Features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # Stack to match (128, 174)
    combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2, mel, chroma, zcr, rms], axis=0)
    
    # Pad/Trim to 128
    if combined.shape[1] < 128:
        combined = np.pad(combined, ((0, 0), (0, 128 - combined.shape[1])), mode='constant')
    else:
        combined = combined[:, :128]
        
    return combined.T  # (128, 174)

# --- 4. STREAMLIT UI ---
st.title("🎙️ Speech Emotion Recognition")

try:
    model, scaler, int_to_label = load_ser_assets()
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Predict Emotion"):
            # Process [cite: 1]
            features = extract_live_features(uploaded_file)
            scaled = scaler.transform(features)
            final_input = np.expand_dims(scaled, axis=0)
            
            # Predict
            preds = model.predict(final_input)[0]
            emotion = int_to_label[str(np.argmax(preds))]
            
            st.success(f"Detected Emotion: {emotion.upper()}")
            st.bar_chart({int_to_label[k]: float(preds[int(k)]) for k in int_to_label})

except Exception as e:
    st.error(f"Error: {e}")
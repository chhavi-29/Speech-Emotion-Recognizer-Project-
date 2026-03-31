import streamlit as st
import sys

# 1. THE ULTIMATE BYPASS: If pkg_resources is missing, we create a fake one
# This prevents the app from crashing before it even starts.
try:
    import pkg_resources
except ImportError:
    # We create a dummy object so 'import pkg_resources' doesn't fail
    import types
    pkg_resources = types.ModuleType("pkg_resources")
    sys.modules["pkg_resources"] = pkg_resources

import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
import json
import tempfile
import base64
from streamlit_mic_recorder import mic_recorder





import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import json
import tempfile
import os
import base64
from streamlit_mic_recorder import mic_recorder

# ─────────────────────────────────────────
# 1. ATTENTION LAYER (Matches Colab)
# ─────────────────────────────────────────
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
        score             = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context           = tf.reduce_sum(x * attention_weights, axis=1)
        return context, attention_weights

    def get_config(self):
        return super().get_config()

# ─────────────────────────────────────────
# 2. BACKGROUND & CSS (Fully Restored)
# ─────────────────────────────────────────
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    ext = image_file.split(".")[-1].lower()
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:{mime};base64,{encoded}");
            background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed;
        }}
        .stApp::before {{
            content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 20, 40, 0.72); z-index: 0;
        }}
        .block-container {{ position: relative; z-index: 1; }}
        h1, h2, h3 {{ color: #00f5d4 !important; text-shadow: 0 0 10px rgba(0,245,212,0.4); }}
        p, label {{ color: #e0f7fa !important; }}
        .stButton > button {{
            background: linear-gradient(135deg, #00b4d8, #0077b6) !important;
            color: white !important; border: none !important; border-radius: 8px !important;
            transition: all 0.3s ease;
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, #00f5d4, #00b4d8) !important;
            transform: scale(1.03); box-shadow: 0 0 15px rgba(0,245,212,0.5);
        }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────
# 3. ASSET LOADING (With Functional Fix)
# ─────────────────────────────────────────
@st.cache_resource
def load_ser_assets():
    # Attempt to handle Keras 3 'Functional' class mismatch
    custom_objs = {'AttentionLayer': AttentionLayer}
    
    try:
        from keras.src.models.functional import Functional
        custom_objs['Functional'] = Functional
    except ImportError:
        pass # Fallback if using Keras 2 environment

    model = tf.keras.models.load_model(
        'ser_model_final.keras',
        custom_objects=custom_objs
    )
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_map.json', 'r') as f:
        labels = json.load(f)['int_to_label']
    return model, scaler, labels

# ─────────────────────────────────────────
# 4. FEATURE EXTRACTION (Unchanged)
# ─────────────────────────────────────────
def extract_features(audio_path, scaler):
    y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
    y, _  = librosa.effects.trim(y, top_db=20)
    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mel         = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), ref=np.max)
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    zcr         = librosa.feature.zero_crossing_rate(y)
    rms         = librosa.feature.rms(y=y)
    combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2, mel, chroma, zcr, rms], axis=0)
    if combined.shape[1] < 128:
        combined = np.pad(combined, ((0, 0), (0, 128 - combined.shape[1])), mode='constant')
    else:
        combined = combined[:, :128]
    features = combined.T
    T, F     = features.shape
    scaled   = scaler.transform(features.reshape(-1, F)).reshape(1, T, F)
    return scaled

# ─────────────────────────────────────────
# 5. PREDICTION LOGIC (Multi-output safe)
# ─────────────────────────────────────────
EMOTION_EMOJI = {'neutral': '😐', 'calm': '😌', 'happy': '😄', 'sad': '😢', 'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'}
EMOTION_COLOR = {'neutral': '#95A5A6', 'calm': '#3498DB', 'happy': '#F1C40F', 'sad': '#7F8C8D', 'angry': '#E74C3C', 'fearful': '#9B59B6', 'disgust': '#27AE60', 'surprised': '#E67E22'}

def predict_and_display(audio_bytes, model, scaler, int_to_label):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        final_input = extract_features(tmp_path, scaler)
        raw_preds = model.predict(final_input, verbose=0)
        
        # Determine if output is list [preds, weights] or single array
        if isinstance(raw_preds, list):
            preds = raw_preds[0][0]
        else:
            preds = raw_preds[0]
        
        top_idx = int(np.argmax(preds))
        emotion = int_to_label[str(top_idx)]
        conf    = float(preds[top_idx])
        emoji   = EMOTION_EMOJI.get(emotion, '🎭')
        color   = EMOTION_COLOR.get(emotion, '#00f5d4')

        st.markdown(f"""
            <div style="background: rgba(0,30,60,0.8); border: 2px solid {color}; border-radius: 12px; padding: 15px; text-align: center; margin: 10px 0; display: flex; align-items: center; justify-content: center; gap: 15px;">
                <span style="font-size: 2.2rem">{emoji}</span>
                <span style="font-size: 1.8rem; font-weight: bold; color: {color}">{emotion.upper()}</span>
                <span style="color: #80deea; font-size: 1rem">{conf*100:.1f}% confidence</span>
            </div>
            """, unsafe_allow_html=True)
        st.progress(conf)
        st.bar_chart({int_to_label[k]: float(preds[int(k)]) for k in int_to_label})
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    finally:
        os.unlink(tmp_path)

# ─────────────────────────────────────────
# 6. MAIN APP UI
# ─────────────────────────────────────────
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

if os.path.exists("img.jpg"):
    set_background("img.jpg")

st.markdown("<h1 style='text-align:center;'>🎙️ Speech Emotion Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#80deea;'>CNN + BiLSTM + Attention | Trained on RAVDESS</p>", unsafe_allow_html=True)
st.markdown("---")

try:
    model, scaler, int_to_label = load_ser_assets()
except Exception as e:
    st.error(f"❌ Model Loading Error: {e}")
    st.info("Check requirements.txt to ensure tensorflow and keras versions match your training environment.")
    st.stop()

tab1, tab2 = st.tabs(["📁 File Upload", "🎤 Live Recording"])

with tab1:
    up_file = st.file_uploader("Upload .wav", type=["wav"])
    if up_file:
        st.audio(up_file)
        if st.button("Predict Emotion", key="up_p", type="primary"):
            predict_and_display(up_file.read(), model, scaler, int_to_label)

with tab2:
    if 'mic_bytes' not in st.session_state: 
        st.session_state['mic_bytes'] = None
    
    audio = mic_recorder(start_prompt="⏺ Start Recording", stop_prompt="⏹ Stop Recording", key="mic_rec")
    if audio and audio.get("bytes"):
        st.session_state['mic_bytes'] = audio["bytes"]
    
    if st.session_state['mic_bytes']:
        st.audio(st.session_state['mic_bytes'])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Predict Recording", key="mic_p", type="primary"):
                predict_and_display(st.session_state['mic_bytes'], model, scaler, int_to_label)
        with col2:
            if st.button("🔄 Reset", key="mic_r"):
                st.session_state['mic_bytes'] = None
                st.rerun()

st.markdown("---")
st.markdown("<p style='text-align:center; color:#80deea; font-size:0.8rem;'>Built with TensorFlow + Streamlit | @2026</p>", unsafe_allow_html=True)

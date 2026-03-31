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
# 1. ATTENTION LAYER
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
        return context

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────
# 2. BACKGROUND IMAGE
# ─────────────────────────────────────────
def set_background(image_file):
    """Read image file → base64 → inject as CSS background."""
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    ext = image_file.split(".")[-1].lower()
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

    st.markdown(
        f"""
        <style>
        /* Full page background */
        .stApp {{
            background-image: url("data:{mime};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Dark overlay so text is readable */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 20, 40, 0.72);
            z-index: 0;
        }}

        /* Make all content sit above overlay */
        .block-container {{
            position: relative;
            z-index: 1;
        }}

        /* ── Text colors ── */
        h1, h2, h3, h4, h5, h6 {{
            color: #00f5d4 !important;
            text-shadow: 0 0 10px rgba(0,245,212,0.4);
        }}
        p, label, .stMarkdown, .stCaption {{
            color: #e0f7fa !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            background: rgba(0, 50, 80, 0.6);
            border-radius: 10px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: #80deea !important;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(0, 245, 212, 0.2) !important;
            color: #00f5d4 !important;
            border-radius: 8px;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            background: linear-gradient(135deg, #00b4d8, #0077b6) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease;
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, #00f5d4, #00b4d8) !important;
            transform: scale(1.03);
            box-shadow: 0 0 15px rgba(0,245,212,0.5);
        }}

        /* ── Info / Success / Error boxes ── */
        .stAlert {{
            background: rgba(0, 40, 70, 0.75) !important;
            border-radius: 10px !important;
            border-left: 4px solid #00f5d4 !important;
            color: #e0f7fa !important;
        }}

        /* ── File uploader ── */
        .stFileUploader {{
            background: rgba(0, 40, 70, 0.6) !important;
            border-radius: 10px !important;
            border: 1px dashed #00b4d8 !important;
            padding: 10px;
        }}

        /* ── Audio player ── */
        audio {{
            width: 100%;
            border-radius: 8px;
        }}

        /* ── Progress bar ── */
        .stProgress > div > div {{
            background: linear-gradient(90deg, #00b4d8, #00f5d4) !important;
            border-radius: 10px;
        }}

        /* ── Bar chart ── */
        .stVegaLiteChart {{
            background: rgba(0, 30, 60, 0.6) !important;
            border-radius: 10px !important;
            padding: 10px;
        }}

        /* ── Divider ── */
        hr {{
            border-color: rgba(0, 245, 212, 0.3) !important;
        }}

        /* ── Footer ── */
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────
# 3. LOAD ASSETS
# ─────────────────────────────────────────
@st.cache_resource
def load_ser_assets():
    model = tf.keras.models.load_model(
        'ser_model_final.keras',
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_map.json', 'r') as f:
        labels = json.load(f)['int_to_label']
    return model, scaler, labels


# ─────────────────────────────────────────
# 4. FEATURE EXTRACTION
# ─────────────────────────────────────────
def extract_features(audio_path, scaler):
    y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
    y, _  = librosa.effects.trim(y, top_db=20)

    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mel         = librosa.power_to_db(
                      librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40),
                      ref=np.max)
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    zcr         = librosa.feature.zero_crossing_rate(y)
    rms         = librosa.feature.rms(y=y)

    combined = np.concatenate(
        [mfcc, mfcc_delta, mfcc_delta2, mel, chroma, zcr, rms], axis=0
    )

    if combined.shape[1] < 128:
        combined = np.pad(combined, ((0, 0), (0, 128 - combined.shape[1])), mode='constant')
    else:
        combined = combined[:, :128]

    features = combined.T
    T, F     = features.shape
    scaled   = scaler.transform(features.reshape(-1, F)).reshape(1, T, F)
    return scaled


# ─────────────────────────────────────────
# 5. PREDICT + DISPLAY
# ─────────────────────────────────────────
EMOTION_EMOJI = {
    'neutral': '😐', 'calm': '😌', 'happy': '😄', 'sad': '😢',
    'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
}
EMOTION_COLOR = {
    'neutral': '#95A5A6', 'calm': '#3498DB', 'happy': '#F1C40F',
    'sad': '#7F8C8D',  'angry': '#E74C3C', 'fearful': '#9B59B6',
    'disgust': '#27AE60', 'surprised': '#E67E22'
}

def predict_and_display(audio_bytes, model, scaler, int_to_label):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        final_input = extract_features(tmp_path, scaler)
        preds       = model.predict(final_input, verbose=0)[0]
        top_idx     = int(np.argmax(preds))
        emotion     = int_to_label[str(top_idx)]
        conf        = float(preds[top_idx])
        emoji       = EMOTION_EMOJI.get(emotion, '🎭')
        color       = EMOTION_COLOR.get(emotion, '#00f5d4')

        # Result card — compact inline layout
        st.markdown(
            f"""
            <div style="
                background: rgba(0,30,60,0.8);
                border: 2px solid {color};
                border-radius: 12px;
                padding: 10px 20px;
                text-align: center;
                margin: 8px 0;
                box-shadow: 0 0 12px {color}55;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 14px;
            ">
                <span style="font-size: 2rem">{emoji}</span>
                <span style="font-size: 1.6rem; font-weight: bold; color: {color}">
                    {emotion.upper()}
                </span>
                <span style="color: #80deea; font-size: 0.95rem">
                    {conf*100:.1f}% confidence
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(conf)
        st.bar_chart({int_to_label[k]: float(preds[int(k)]) for k in int_to_label})

    except Exception as e:
        st.error(f"Prediction failed: {e}")
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────
# 6. PAGE CONFIG + BACKGROUND
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "Speech Emotion Recognition",
    page_icon  = "🎙️",
    layout     = "centered"
)

# Apply background — image must be in same folder as app.py
if os.path.exists("img.jpg"):
    set_background("img.jpg")
elif os.path.exists("bg.jpg"):
    set_background("bg.jpg")
# If neither found, app still works with default background


# ─────────────────────────────────────────
# 7. HEADER
# ─────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; font-size:2.5rem'>
        🎙️ Speech Emotion Recognition
    </h1>
    <p style='text-align:center; color:#80deea; font-size:1rem'>
        CNN + BiLSTM + Attention &nbsp;|&nbsp; Trained on RAVDESS Dataset
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")


# ─────────────────────────────────────────
# 8. LOAD MODEL
# ─────────────────────────────────────────
try:
    model, scaler, int_to_label = load_ser_assets()
  
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.info(
        "Make sure these files are in the same folder as app.py:\n"
        "- `ser_model_final.keras`\n- `scaler.pkl`\n- `label_map.json`"
    )
    st.stop()


# ─────────────────────────────────────────
# 9. TWO TABS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎤 Record from Microphone"])


# ── TAB 1: File Upload ──
with tab1:
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Predict Emotion", type="primary", key="upload_predict"):
            with st.spinner("Analyzing..."):
                predict_and_display(
                    audio_bytes  = uploaded_file.read(),
                    model        = model,
                    scaler       = scaler,
                    int_to_label = int_to_label
                )


# ── TAB 2: Live Mic (session_state fix) ──
with tab2:
    st.markdown("#### 🎤 Speak into your microphone")
    st.info("💡 Click **Start**, speak for 2–3 seconds, then click **Stop**.")

    if 'mic_audio_bytes' not in st.session_state:
        st.session_state['mic_audio_bytes'] = None

    audio = mic_recorder(
        start_prompt        = "⏺ Start Recording",
        stop_prompt         = "⏹ Stop Recording",
        just_once           = False,
        use_container_width = True,
        key                 = "mic_recorder"
    )

    if audio and audio.get("bytes"):
        st.session_state['mic_audio_bytes'] = audio["bytes"]

    if st.session_state['mic_audio_bytes']:
        st.success("✅ Recording captured!")
        st.audio(st.session_state['mic_audio_bytes'], format="audio/wav")

        col1, col2 = st.columns([1, 1])
        with col1:
            predict_btn = st.button("Predict Emotion", type="primary", key="mic_predict")
        with col2:
            if st.button("🔄 Record Again", key="mic_clear"):
                st.session_state['mic_audio_bytes'] = None
                st.rerun()

        if predict_btn:
            with st.spinner("Analyzing your voice..."):
                predict_and_display(
                    audio_bytes  = st.session_state['mic_audio_bytes'],
                    model        = model,
                    scaler       = scaler,
                    int_to_label = int_to_label
                )


# ─────────────────────────────────────────
# 10. FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#80deea; font-size:0.8rem'>"
    "Built with TensorFlow + Streamlit &nbsp;|&nbsp; @2026"
    "</p>",
    unsafe_allow_html=True
)
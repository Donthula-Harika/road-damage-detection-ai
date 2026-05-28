import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import json

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================
model = load_model("road_damage_model.keras", compile=False)

with open("label_mapping.json", "r") as f:
    label_mapping = json.load(f)

label_mapping = {
    int(k): v for k, v in label_mapping.items()
}

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(168,85,247,0.15));
    padding: 40px;
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 48px;
    font-weight: 700;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: #cbd5e1;
    margin-top: 8px;
    line-height: 1.7;
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 25px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
}

.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 18px;
    color: white;
}

.metric-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(14,165,233,0.10));
    border-radius: 22px;
    padding: 22px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

.metric-label {
    color: #cbd5e1;
    font-size: 16px;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: white;
}

.high {
    color: #ef4444;
}

.medium {
    color: #f59e0b;
}

.low {
    color: #22c55e;
}

.recommendation {
    background: rgba(255,255,255,0.05);
    padding: 22px;
    border-radius: 20px;
    border-left: 5px solid #3b82f6;
    line-height: 1.8;
    font-size: 16px;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border-radius: 20px;
    padding: 15px;
    border: 2px dashed rgba(255,255,255,0.15);
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HERO SECTION
# =====================================
st.markdown(f"""
<div class="hero">
    <div class="hero-title">
        🛣️ AI-Based Road Damage Detection System
    </div>

    
</div>
""", unsafe_allow_html=True)

# =====================================
# ABOUT SECTION
# =====================================
# st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">📘 About the Project</div>',
    unsafe_allow_html=True
)

st.write(
    """
    Road damage causes accidents, traffic congestion, and infrastructure deterioration.

    This intelligent CNN-based system automatically analyzes road surface images
    and classifies damage types such as potholes, cracks, and manholes.

    ### Industry Applications
    - Smart city surveillance
    - Automated road inspection
    - AI-powered infrastructure monitoring
    - Highway maintenance systems
    - Municipal safety analytics
    """
)

# st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# IMAGE UPLOAD
# =====================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">📤 Upload Road Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# IF IMAGE UPLOADED
# =====================================
if uploaded_file:

    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1.1, 1])

    # =====================================
    # IMAGE PREVIEW
    # =====================================
    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">🖼️ Uploaded Image</div>',
            unsafe_allow_html=True
        )

        st.image(img, width=500)

        st.markdown('</div>', unsafe_allow_html=True)

    # =====================================
    # MODEL PREDICTION
    # =====================================
    with col2:

        img_resized = img.resize((128, 128))

        img_array = image.img_to_array(img_resized)

        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        predicted_label = label_mapping[predicted_class]

        # =====================================
        # SEVERITY
        # =====================================
        if predicted_label == "Pothole":
            severity = "High"
            severity_class = "high"

        elif predicted_label == "Crack":
            severity = "Medium"
            severity_class = "medium"

        else:
            severity = "Low"
            severity_class = "low"

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">🧠 Prediction Result</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prediction</div>
                <div class="metric-value">{predicted_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Severity</div>
                <div class="metric-value {severity_class}">{severity}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =====================================
    # VISUALIZATION
    # =====================================
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📊 Confidence Visualization</div>',
        unsafe_allow_html=True
    )

    class_names = list(label_mapping.values())

    probabilities = prediction[0] * 100

    fig, ax = plt.subplots(figsize=(8,4))

    bars = ax.bar(class_names, probabilities)

    ax.set_ylabel("Confidence (%)")

    ax.set_title("Class Confidence Scores")

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f'{height:.1f}%',
            ha='center'
        )

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================
    # RECOMMENDATIONS
    # =====================================
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🚨 Maintenance Recommendations</div>',
        unsafe_allow_html=True
    )

    if severity == "High":

        recommendation = """
        Immediate maintenance recommended.<br><br>
        High-risk pothole detected. This road condition may cause accidents and vehicle damage.
        """

    elif severity == "Medium":

        recommendation = """
        Maintenance should be scheduled soon.<br><br>
        Moderate crack detected. Continuous deterioration may increase future repair costs.
        """

    else:

        recommendation = """
        Low-risk condition detected.<br><br>
        Regular monitoring recommended. No immediate maintenance required.
        """

    st.markdown(f"""
    <div class="recommendation">
        {recommendation}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# FOOTER
# =====================================
st.markdown("---")

st.markdown(
    """
    <center>
        <h4>🌍 AI for Smarter & Safer Roads</h4>
        <p>Powered by CNN • TensorFlow • Streamlit</p>
    </center>
    """,
    unsafe_allow_html=True
)



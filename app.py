import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# =====================================
# CUSTOM CSS – Navbar + Tampilan Modern
# =====================================
st.markdown("""
<style>

body {
    background-color: #f5f7fa;
}

/* NAVBAR */
.navbar {
    background-color: #2563eb;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 25px;
    display: flex;
    gap: 25px;
}

.nav-item {
    color: white;
    font-size: 17px;
    font-weight: 600;
    text-decoration: none;
    padding: 5px 15px;
    border-radius: 8px;
}

.nav-item:hover {
    background: #1e40af;
    cursor: pointer;
}

.selected {
    background: #1e40af;
}

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.title {
    font-size: 32px;
    font-weight: 800;
    color: #1f2937;
}

.subtitle {
    font-size: 16px;
    color: #4b5563;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# NAVBAR LOGIC
# =====================================
nav = st.session_state.get("nav", "Home")

st.markdown("<div class='navbar'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Home"):
        st.session_state.nav = "Home"

with col2:
    if st.button("Kamera"):
        st.session_state.nav = "Kamera"

with col3:
    if st.button("Tentang"):
        st.session_state.nav = "Tentang"

st.markdown("</div>", unsafe_allow_html=True)

page = st.session_state.get("nav", "Home")

# =====================================
# Load model
# =====================================
MODEL_PATH = 'JAGUNG/Dataset/model_jagung.h5'

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("Model berhasil dimuat ✓")
    except:
        st.error("Model gagal dimuat")
else:
    st.error("Model tidak ditemukan.")

CLASSES = ['Healthy', 'Common Rust', 'Gray Leaf Spot', 'Blight']

TREATMENTS = {
    "Healthy": """
    🌱 **Tanaman Sehat**
    - Tidak perlu tindakan khusus
    - Monitoring rutin
    - Pertahankan nutrisi seimbang
    """,

    "Common Rust": """
    🔧 **Penanganan Common Rust**
    - Gunakan fungisida: *Azoxystrobin, Mancozeb*
    - Rotasi tanaman sangat dianjurkan
    - Hindari penyiraman ke daun
    """,

    "Gray Leaf Spot": """
    🔧 **Penanganan Gray Leaf Spot**
    - Gunakan fungisida: *Trifloxystrobin / Propiconazole*
    - Kurangi kelembaban
    - Buang daun yang terinfeksi berat
    """,

    "Blight": """
    🔧 **Penanganan Blight**
    - Gunakan fungisida: *Copper fungicide / Mancozeb*
    - Tingkatkan sirkulasi udara tanaman
    - Buang daun yang terinfeksi
    """
}

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(img_array):
    preds = model.predict(img_array)
    idx = np.argmax(preds, axis=1)[0]
    return CLASSES[idx], preds[0][idx]

# =====================================
# PAGE CONTENT
# =====================================
def home_page():
    st.markdown("<div class='title'>Deteksi Penyakit Daun Jagung</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Aplikasi AI untuk mendeteksi penyakit daun jagung</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.write("""
    **Cara Penggunaan**
    1. Klik menu **Kamera** di atas  
    2. Ambil gambar daun jagung  
    3. Lihat hasil prediksi + cara penanganan
    """)

    st.write("**Penyakit yang dapat dideteksi:**")
    st.write("- Healthy\n- Common Rust\n- Gray Leaf Spot\n- Blight")

    st.markdown("</div>", unsafe_allow_html=True)


def camera_page():
    st.markdown("<div class='title'>Deteksi Menggunakan Kamera</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Ambil gambar daun jagung untuk dianalisis</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    cam = st.camera_input("Ambil foto daun jagung:")

    if cam:
        st.image(cam, caption="Gambar Diambil", use_container_width=True)

        img = Image.open(cam)
        img_array = preprocess_image(img)

        label, conf = predict_image(img_array)

        st.subheader("🔍 Hasil Prediksi")
        st.success(f"**Kategori:** {label}")
        st.info(f"**Probabilitas:** {conf:.2f}")

        st.markdown("### 🛠 Cara Penanganan")
        st.markdown(f"<div class='card'>{TREATMENTS[label]}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def about_page():
    st.markdown("<div class='title'>Tentang Aplikasi</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.write("""
    Aplikasi ini dibuat untuk mendeteksi penyakit daun jagung menggunakan Machine Learning.
    """)

    st.subheader("👥 Kelompok 2")
    st.write("""
    - **Deni Andriansyah** (Ketua)  
    - Afip Dwi Cahyo  
    - Melinda Purnama D  
    """)

    st.subheader("🎯 Tujuan")
    st.write("""
    Membantu petani mendeteksi penyakit lebih cepat untuk mengurangi gagal panen.
    """)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================
# PAGE ROUTING
# =====================================
if page == "Home":
    home_page()
elif page == "Kamera":
    camera_page()
elif page == "Tentang":
    about_page()

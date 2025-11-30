import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# =====================================
# CUSTOM CSS – Tampilan Modern
# =====================================
st.markdown("""
<style>
.main { background-color: #f5f7fa; }

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

.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.stButton>button {
    background: #2563eb;
    color: white;
    padding: 0.6rem 1.2rem;
    border-radius: 12px;
    border: none;
}

.stButton>button:hover {
    background: #1e40af;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# LOAD MODEL
# =====================================
MODEL_PATH = 'JAGUNG/Dataset/model_jagung.h5'

model = None
if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan di {MODEL_PATH}.")
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.sidebar.success("Model berhasil dimuat ✓")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")

# Kelas penyakit
CLASSES = ['Healthy', 'Common Rust', 'Gray Leaf Spot', 'Blight']


# =====================================
# DATA PENANGANAN
# =====================================
TREATMENTS = {
    "Healthy": """
    🌱 **Tanaman Sehat**
    - Tidak ada tindakan khusus
    - Pertahankan perawatan yang baik
    - Lakukan monitoring rutin
    - Pastikan tanaman mendapat nutrisi seimbang
    """,

    "Common Rust": """
    🔧 **Penanganan Common Rust (Karat Daun)**
    - Gunakan fungisida: *Azoxystrobin, Mancozeb, Pyraclostrobin*
    - Lakukan rotasi tanaman untuk memutus siklus jamur
    - Hindari penyiraman yang mengenai daun
    - Gunakan varietas jagung tahan karat
    """,

    "Gray Leaf Spot": """
    🔧 **Penanganan Gray Leaf Spot**
    - Aplikasikan fungisida: *Trifloxystrobin, Propiconazole, Mancozeb*
    - Kurangi kelembaban dengan memperjarang tanaman
    - Buang daun yang terinfeksi berat
    - Lakukan pergiliran tanaman (crop rotation)
    """,

    "Blight": """
    🔧 **Penanganan Blight (Hawar Daun)**
    - Gunakan fungisida: *Copper fungicide, Chlorothalonil, Mancozeb*
    - Bersihkan sisa tanaman yang terinfeksi setelah panen
    - Tingkatkan sirkulasi udara antar tanaman
    - Gunakan benih unggul yang tahan penyakit
    """
}


# =====================================
# IMAGE PROCESSING
# =====================================
def preprocess_image(img):
    try:
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Error saat memproses gambar: {e}")
        return None


def predict_image(img_array):
    try:
        preds = model.predict(img_array)
        idx = np.argmax(preds, axis=1)[0]
        return CLASSES[idx], preds[0][idx]
    except Exception as e:
        st.error(f"Error saat memprediksi gambar: {e}")
        return None, None


# =====================================
# HALAMAN: HOME
# =====================================
def home_page():
    st.markdown("<div class='title'>Deteksi Penyakit Daun Jagung</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Aplikasi AI untuk mendeteksi penyakit daun jagung secara otomatis</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 Cara Penggunaan")
    st.write("""
    1. Pergi ke menu **Kamera**
    2. Ambil gambar daun jagung
    3. Lihat hasil prediksi dan probabilitas
    4. Ikuti **cara penanganan** yang diberikan
    """)

    st.subheader("📚 Penyakit yang Didukung")
    st.write("""
    - **Healthy (Sehat)**
    - **Common Rust**
    - **Gray Leaf Spot**
    - **Blight**
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================
# HALAMAN: KAMERA
# =====================================
def camera_page():
    st.markdown("<div class='title'>Deteksi Dari Kamera</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Ambil gambar daun jagung untuk dianalisis</div>", unsafe_allow_html=True)

    if model is None:
        st.error("Model belum dimuat!")
        return

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    camera_input = st.camera_input("Ambil foto daun jagung:")

    if camera_input is not None:
        st.image(camera_input, caption="Gambar yang diambil", use_container_width=True)

        img = Image.open(camera_input)
        img_array = preprocess_image(img)

        if img_array is not None:
            label, confidence = predict_image(img_array)

            if label:
                st.subheader("🔍 Hasil Prediksi")
                st.success(f"**Kategori:** {label}")
                st.info(f"**Probabilitas:** {confidence:.2f}")

                # Tampilkan penanganan
                if label in TREATMENTS:
                    st.markdown("### 🛠 Cara Penanganan")
                    st.markdown(f"<div class='card'>{TREATMENTS[label]}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================
# HALAMAN: TENTANG
# =====================================
def about_page():
    st.markdown("<div class='title'>Tentang Aplikasi</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("""
    Aplikasi ini dibuat menggunakan Machine Learning untuk mendeteksi penyakit daun jagung.
    """)

    st.subheader("👥 Kelompok 2")
    st.write("""
    **Ketua:**  
    - Deni Andriansyah  

    **Anggota:**  
    - Afip Dwi Cahyo  
    - Melinda Purnama D  
    """)

    st.subheader("🎯 Tujuan")
    st.write("""
    Mendeteksi penyakit daun jagung secara cepat dan akurat untuk membantu petani mengurangi risiko gagal panen.
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================
# SIDEBAR NAVIGASI
# =====================================
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Kamera", "Tentang Aplikasi"])

if page == "Home":
    home_page()
elif page == "Kamera":
    camera_page()
elif page == "Tentang Aplikasi":
    about_page()

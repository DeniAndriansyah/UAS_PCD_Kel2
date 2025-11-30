import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import os

# =======================
#  CONFIGURASI HALAMAN
# =======================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Jagung",
    page_icon="🌽",
    layout="wide"
)

# =======================
#  LOAD MODEL
# =======================
model_path = "model_jagung.h5"

if not os.path.exists(model_path):
    st.error("❌ model.h5 tidak ditemukan! Pastikan file berada 1 folder dengan app.py")
    model = None
else:
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        model = None

# =======================
#  KELAS & PENANGANAN
# =======================
CLASSES = ["Healthy", "Common Rust", "Gray Leaf Spot", "Blight"]

TREATMENTS = {
    "Healthy": """
    ✔ Tanaman sehat  
    ✔ Pertahankan pola perawatan  
    ✔ Monitoring rutin dan pemupukan seimbang  
    """,

    "Common Rust": """
    **Penanganan Common Rust:**
    - Gunakan fungisida: *Azoxystrobin, Mancozeb, Pyraclostrobin*  
    - Lakukan rotasi tanaman  
    - Hindari daun terlalu lembab  
    - Gunakan varietas jagung tahan penyakit  
    """,

    "Gray Leaf Spot": """
    **Penanganan Gray Leaf Spot:**
    - Aplikasikan fungisida: *Trifloxystrobin, Propiconazole, Mancozeb*  
    - Kurangi kelembaban tanaman  
    - Buang daun yang sudah parah  
    - Lakukan pergiliran tanaman (crop rotation)  
    """,

    "Blight": """
    **Penanganan Blight (Hawar Daun):**
    - Gunakan fungisida: *Copper, Chlorothalonil, Mancozeb*  
    - Tingkatkan sirkulasi udara tanaman  
    - Bersihkan sisa tanaman terinfeksi  
    - Gunakan benih unggul  
    """
}


# =======================
#  STYLE CSS NAVBAR
# =======================
def load_css():
    st.markdown("""
        <style>
        /* NAVBAR */
        .navbar {
            background: #2b8a3e;
            padding: 15px;
            border-radius: 10px;
            display: flex;
            gap: 25px;
            justify-content: center;
            margin-bottom: 20px;
        }

        .navbtn {
            padding: 10px 18px;
            background: #fff;
            color: #2b8a3e;
            font-weight: 600;
            border-radius: 8px;
            text-decoration: none;
            transition: 0.3s;
        }

        .navbtn:hover {
            background: #c4f1c7;
        }

        .selected {
            background: #145c2f !important;
            color: white !important;
        }

        /* CARD */
        .card {
            background: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

load_css()


# =======================
#  FUNGSI PENGOLAHAN
# =======================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict(img_array):
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    return CLASSES[idx], preds[0][idx]


# =======================
#  NAVBAR INTERAKTIF
# =======================
page = st.session_state.get("page", "Home")

navbar_html = f"""
<div class="navbar">
    <a class="navbtn {'selected' if page=='Home' else ''}" href="?page=Home">Home</a>
    <a class="navbtn {'selected' if page=='Camera' else ''}" href="?page=Camera">Kamera</a>
    <a class="navbtn {'selected' if page=='About' else ''}" href="?page=About">Tentang</a>
</div>
"""

st.markdown(navbar_html, unsafe_allow_html=True)

query = st.query_params
if "page" in query:
    page = query["page"]
st.session_state["page"] = page


# =======================
#  HALAMAN HOME
# =======================
if page == "Home":
    st.title("🌽 Deteksi Penyakit Daun Jagung")
    st.write("""
    Aplikasi ini dapat mendeteksi penyakit pada daun jagung menggunakan kamera.
    
    **Fiturnya:**
    - 📸 Deteksi menggunakan kamera
    - 🧠 Model AI (CNN)
    - 🩺 Menampilkan cara penanganan penyakit  
    """)

    st.image(
        "https://cdn.pixabay.com/photo/2016/11/23/14/49/corn-1852369_1280.jpg",
        use_container_width=True,
    )


# =======================
#  HALAMAN KAMERA
# =======================
elif page == "Camera":
    st.title("📸 Deteksi Penyakit Melalui Kamera")

    if model is None:
        st.error("❌ Model belum dimuat.")
    else:
        uploaded = st.camera_input("Ambil gambar daun jagung:")

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Gambar yang diambil", use_container_width=True)

            array = preprocess_image(img)

            label, prob = predict(array)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🔍 Hasil Prediksi")
            st.write(f"**Kategori:** {label}")
            st.write(f"**Probabilitas:** {prob:.2f}")
            st.markdown("### 🛠 Cara Penanganan")
            st.markdown(TREATMENTS[label])
            st.markdown("</div>", unsafe_allow_html=True)


# =======================
#  HALAMAN ABOUT
# =======================
elif page == "About":
    st.title("ℹ Tentang Aplikasi")
    st.write("""
    Aplikasi ini dibuat untuk mendeteksi penyakit daun jagung menggunakan teknologi **Deep Learning**.
    """)

    st.subheader("👥 Kelompok 2")
    st.write("""
    **Ketua:**  
    - Deni Andriansyah  

    **Anggota:**  
    - Afip Dwi Cahyo  
    - Melinda Purnama D
    """)

    st.info("Tujuan aplikasi: Mengidentifikasi penyakit daun jagung dengan cepat & akurat.")


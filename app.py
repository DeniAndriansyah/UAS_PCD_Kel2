import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Deteksi Penyakit Jagung", layout="wide")

# =======================
#  NAVBAR CUSTOM
# =======================
st.markdown("""
<style>
/* Navbar Container */
.navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 65px;
    background: #0D6EFD;
    display: flex;
    align-items: center;
    padding: 0 30px;
    z-index: 100;
    box-shadow: 0px 2px 7px rgba(0,0,0,0.25);
}

/* Navbar Items */
.nav-item {
    margin-right: 25px;
    font-size: 18px;
    color: white;
    cursor: pointer;
    transition: 0.25s;
}

.nav-item:hover {
    color: #FFD700;
}

/* Highlight when active */
.active {
    border-bottom: 3px solid #FFD700;
    padding-bottom: 5px;
    color: #FFD700 !important;
}

/* Add top padding to the body */
body {
    margin-top: 80px !important;
}
</style>

<div class="navbar">
    <span class="nav-item active" onclick="switchPage('Home')">Home</span>
    <span class="nav-item" onclick="switchPage('Kamera')">Kamera Diagnosa</span>
    <span class="nav-item" onclick="switchPage('Tentang')">Tentang</span>
    <span class="nav-item" onclick="switchPage('Bantuan')">Bantuan</span>
</div>

<script>
function switchPage(page) {
    window.location.href = "?page=" + page;
}
</script>
""", unsafe_allow_html=True)


# =======================
# NAVIGATION LOGIC
# =======================
page = st.query_params.get("page", "Home")


# =======================
# FUNGSI PREDIKSI
# =======================
model_path = "model.h5"
model = tf.keras.models.load_model(model_path)

CLASS_NAMES = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]

TREATMENTS = {
    "Healthy": """
    - Tanaman dalam kondisi sehat 🌱  
    - Pertahankan perawatan normal  
    - Lakukan monitoring rutin  
    """,
    "Common Rust": """
    **Penanganan Common Rust:**
    - Gunakan fungisida *Azoxystrobin, Mancozeb*
    - Lakukan rotasi tanaman
    - Buang daun yang terinfeksi berat
    """,
    "Gray Leaf Spot": """
    **Penanganan Gray Leaf Spot:**
    - Aplikasikan fungisida *Propiconazole, Trifloxystrobin*
    - Kurangi kelembapan & lakukan penjarangan
    """,
    "Blight": """
    **Penanganan Hawar Daun (Blight):**
    - Gunakan fungisida *Chlorothalonil, Mancozeb*
    - Tingkatkan sirkulasi udara tanaman
    - Bersihkan daun yang terinfeksi
    """
}


def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    idx = np.argmax(predictions)
    return CLASS_NAMES[idx], float(np.max(predictions))


# =======================
#  PAGE : HOME
# =======================
if page == "Home":
    st.title("🌽 Sistem Deteksi Penyakit Daun Jagung")
    st.write("Gunakan kamera untuk mendiagnosa penyakit daun jagung secara real-time.")


# =======================
# PAGE : KAMERA
# =======================
elif page == "Kamera":
    st.title("📷 Kamera Diagnosa Penyakit Jagung")
    camera_input = st.camera_input("Ambil gambar daun jagung:")

    if camera_input:
        img = Image.open(camera_input)
        st.image(img, caption="Gambar Terdeteksi", width=350)

        label, prob = predict_image(img)

        st.success(f"**Prediksi:** {label}")
        st.info(f"**Probabilitas:** {prob:.2f}")

        st.markdown("### 🛠 Rekomendasi Penanganan")
        st.markdown(TREATMENTS[label])


# =======================
# PAGE : TENTANG
# =======================
elif page == "Tentang":
    st.title("ℹ Tentang Aplikasi")
    st.write("""
    Aplikasi ini dibuat untuk mendeteksi penyakit daun jagung menggunakan model CNN.
    """)


# =======================
# PAGE : BANTUAN
# =======================
elif page == "Bantuan":
    st.title("❓ Bantuan")
    st.write("Jika Anda butuh bantuan instalasi atau upload model, silakan tanyakan di sini.")

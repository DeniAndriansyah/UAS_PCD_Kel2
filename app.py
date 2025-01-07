import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Path ke model
MODEL_PATH = 'JAGUNG/Dataset/model_jagung.h5'

# Cek apakah model tersedia
if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan di {MODEL_PATH}")
else:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Kelas penyakit daun jagung
CLASSES = ['Healthy', 'Common Rust', 'Gray Leaf Spot', 'Blight']

# Fungsi untuk memproses gambar input
def preprocess_image(img):
    img = img.resize((224, 224))  # Ubah ukuran sesuai model
    img_array = np.array(img) / 255.0  # Normalisasi gambar
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    return img_array

# Fungsi untuk prediksi
def predict_image(img_array):
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)
    return CLASSES[class_idx[0]], preds[0][class_idx[0]]

# Fungsi halaman Home
def home_page():
    st.title("Selamat Datang di Aplikasi Deteksi Penyakit Daun Jagung")
    st.write("""
    Aplikasi ini digunakan untuk mendeteksi penyakit pada daun jagung menggunakan gambar yang diambil melalui kamera.
    
    **Cara Penggunaan:**
    1. Navigasikan ke halaman **Kamera** melalui sidebar.
    2. Ambil gambar daun jagung menggunakan kamera.
    3. Lihat hasil prediksi dan probabilitas.
    
    **Kategori Penyakit yang Didukung:**
    - Healthy (Sehat)
    - Common Rust (Penyakit karat)
    - Gray Leaf Spot (Penyakit bercak daun abu-abu)
    - Blight (Hawar daun)
    """)


# Fungsi halaman Kamera
def camera_page():
    st.title("Deteksi Penyakit Daun Jagung Melalui Kamera")

    # Input kamera
    camera_input = st.camera_input("Silakan ambil gambar daun jagung menggunakan kamera di bawah ini:")

    if camera_input is not None:
        # Tampilkan gambar yang diambil
        st.image(camera_input, caption="Gambar yang Diambil", use_container_width=True)

        # Proses dan prediksi gambar
        img = Image.open(camera_input)
        img_array = preprocess_image(img)

        label, confidence = predict_image(img_array)
        st.subheader("Hasil Prediksi:")
        st.write(f"**Kategori:** {label}")
        st.write(f"**Probabilitas:** {confidence:.2f}")

# Fungsi halaman Tentang Aplikasi
def about_page():
    st.title("Tentang Aplikasi Deteksi Penyakit Daun Jagung")
    st.write("""
    **Deteksi Penyakit Daun Jagung** adalah aplikasi yang menggunakan teknologi kecerdasan buatan (AI) untuk mendeteksi penyakit pada daun jagung. 
    Aplikasi ini bekerja dengan cara memproses gambar daun jagung yang diambil melalui kamera dan memprediksi kategori penyakitnya.
    """)

    st.header("Kelompok 2")
    st.write("""
    **Anggota Kelompok:**
    1. Deni Andriansyah 211351040
    2. Afip Dwi Cahyo	201351004
    3. Melinda Purnama	211351082
    """)

    st.subheader("Tujuan Aplikasi")
    st.write("""
    Aplikasi ini bertujuan untuk klasifikasi penyakit daun jagung ini untuk mengembangkan system yang dapat mendeteksi penyakit daun jagung dengan akurat dan efisien 
    sehingga kerusakan dapat di identifikasi lebih cepat, dan meminimalisir tingkat gagal panen.
    """)


# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Kamera", "Tentang Aplikasi"])

# Render halaman sesuai pilihan
if page == "Home":
    home_page()
elif page == "Kamera":
    camera_page()
elif page == "Tentang Aplikasi":
    about_page()

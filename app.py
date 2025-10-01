import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Path ke model
MODEL_PATH = 'JAGUNG/Dataset/model_jagung.h5'

# Cek apakah model tersedia
model = None
if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan di {MODEL_PATH}. Pastikan model tersedia di lokasi yang benar.")
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.sidebar.success("Model berhasil dimuat.")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")

# Kelas penyakit daun jagung
CLASSES = ['Healthy', 'Common Rust', 'Gray Leaf Spot', 'Blight']

# Fungsi untuk memproses gambar input
def preprocess_image(img):
    """Memproses gambar untuk prediksi."""
    try:
        img = img.resize((224, 224))  # Ubah ukuran sesuai model
        img_array = np.array(img) / 255.0  # Normalisasi gambar
        img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error saat memproses gambar: {e}")
        return None

# Fungsi untuk prediksi
def predict_image(img_array):
    """Melakukan prediksi pada gambar input."""
    try:
        preds = model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)
        return CLASSES[class_idx[0]], preds[0][class_idx[0]]
    except Exception as e:
        st.error(f"Error saat memprediksi gambar: {e}")
        return None, None

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

    # Validasi jika model belum dimuat
    if model is None:
        st.error("Model belum dimuat. Silakan pastikan model tersedia di path yang benar.")
        return

    # Input kamera
    camera_input = st.camera_input("Silakan ambil gambar daun jagung menggunakan kamera di bawah ini:")

    if camera_input is not None:
        try:
            # Tampilkan gambar yang diambil
            st.image(camera_input, caption="Gambar yang Diambil", use_container_width=True)

            # Proses dan prediksi gambar
            img = Image.open(camera_input)
            img_array = preprocess_image(img)

            if img_array is not None:
                label, confidence = predict_image(img_array)
                if label and confidence is not None:
                    st.subheader("Hasil Prediksi:")
                    st.write(f"**Kategori:** {label}")
                    st.write(f"**Probabilitas:** {confidence:.2f}")
        except Exception as e:
            st.error(f"Error saat memproses gambar: {e}")

# Fungsi halaman Tentang Aplikasi
def about_page():
    st.title("Tentang Aplikasi Deteksi Penyakit Daun Jagung")
    st.write("""
    **Deteksi Penyakit Daun Jagung** adalah aplikasi yang menggunakan teknologi kecerdasan buatan (AI) untuk mendeteksi penyakit pada daun jagung. 
    Aplikasi ini bekerja dengan cara memproses gambar daun jagung yang diambil melalui kamera dan memprediksi kategori penyakitnya.
    """)

    st.header("Kelompok 2")
    st.write("""
    **Ketua Kelompok:**
    1. Deni Andriansyah
    
    **Anggota Kelompok:**
    2. Afip Dwi Cahyo
    3. Melinda Purnama D  
    """)

    st.subheader("Tujuan Aplikasi")
    st.write("""
    Aplikasi ini bertujuan untuk klasifikasi penyakit daun jagung ini untuk mengembangkan sistem yang dapat mendeteksi penyakit daun jagung dengan akurat dan efisien 
    sehingga kerusakan dapat diidentifikasi lebih cepat, dan meminimalisir tingkat gagal panen.
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

# -*- coding: utf-8 -*-
"""
♻️ DASHBOARD KLASIFIKASI SAMPAH - RANDOM FOREST (SET 3)
Upload gambar satu per satu atau batch → Prediksi → Tampilkan hasil & probabilitas.
Model dan preprocessor diasumsikan berada di folder yang sama dengan app.py (src/).
"""

import streamlit as st
import os
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage import exposure
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
import zipfile
import shutil
from pathlib import Path

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="♻️ Klasifikasi Sampah",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Judul & Deskripsi ---
st.title("♻️ Klasifikasi Sampah Daur Ulang vs Non-Daur Ulang")
st.markdown("""
Aplikasi ini menggunakan model **Random Forest** untuk mengklasifikasikan gambar sampah menjadi dua kelas:
- ✅ **Recycle** (dapat didaur ulang)
- ❌ **Non-Recycle** (tidak dapat didaur ulang)

Unggah gambar satu per satu atau dalam batch untuk melihat hasil prediksi!
""")

# --- Muat Model dan Preprocessor (FIXED: gunakan path absolut) ---
@st.cache_resource
def load_models():
    try:
        # Dapatkan direktori tempat app.py berada
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path lengkap ke file .pkl
        rf_model_path = os.path.join(current_dir, "rf_set3_model.pkl")
        scaler_path = os.path.join(current_dir, "scaler_set3.pkl")
        selector_path = os.path.join(current_dir, "selector_set3.pkl")
        
        # Cek apakah file ada
        if not os.path.exists(rf_model_path):
            st.error(f"❌ File tidak ditemukan: {rf_model_path}")
            st.write("**Path yang dicari:**")
            st.code(current_dir)
            st.write("**File yang tersedia:**")
            st.code(os.listdir(current_dir))
            st.stop()
        
        if not os.path.exists(scaler_path):
            st.error(f"❌ File tidak ditemukan: {scaler_path}")
            st.stop()
            
        if not os.path.exists(selector_path):
            st.error(f"❌ File tidak ditemukan: {selector_path}")
            st.stop()
        
        # Muat model
        rf_model = joblib.load(rf_model_path)
        scaler = joblib.load(scaler_path)
        selector = joblib.load(selector_path)
        
        return rf_model, scaler, selector
        
    except Exception as e:
        st.error(f"⚠️ Error saat memuat model: {str(e)}")
        st.write("**Current Working Directory:**", os.getcwd())
        st.write("**app.py Location:**", os.path.dirname(os.path.abspath(__file__)))
        st.stop()

rf_model, scaler, selector = load_models()

# --- Fungsi Ekstraksi Fitur (SET 3) ---
def extract_set3_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")
    
    # Konversi ke RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Mean Color (R, G, B)
    mean_color = np.mean(img, axis=(0, 1))
    
    # GLCM (grayscale)
    gray = rgb2gray(img)
    gray = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_props = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    
    # Histogram (64 bins)
    hist_vals, _ = exposure.histogram(gray)
    if len(hist_vals) < 64:
        hist_64 = np.pad(hist_vals, (0, 64 - len(hist_vals)), mode='constant')
    else:
        hist_64 = hist_vals[:64]
    hist_64 = hist_64 / hist_64.sum() if hist_64.sum() > 0 else hist_64
    
    features = np.concatenate([mean_color, glcm_props, hist_64])
    return features

# --- Fungsi Prediksi Satu Gambar ---
def predict_single_image(image_path):
    try:
        features = extract_set3_features(image_path)
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_selected = selector.transform(features_scaled)
        pred = rf_model.predict(features_selected)[0]
        proba = rf_model.predict_proba(features_selected)[0]
        
        class_name = "✅ Recycle" if pred == 0 else "❌ Non-Recycle"
        confidence = proba[int(pred)] * 100
        return class_name, confidence, proba
    except Exception as e:
        return f"❌ Error: {str(e)}", 0, [0, 0]

# --- Fungsi Extract File dari ZIP ---
def extract_images_from_zip(zip_file):
    """Extract semua gambar dari file ZIP"""
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_batch")
    
    # Hapus folder temp jika sudah ada
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                # Skip folder dan file hidden
                if file_info.is_dir() or file_info.filename.startswith('__MACOSX'):
                    continue
                
                # Cek ekstensi file
                file_ext = Path(file_info.filename).suffix.lower()
                if file_ext in image_extensions:
                    # Extract file
                    extracted_path = zip_ref.extract(file_info, temp_dir)
                    image_paths.append({
                        'path': extracted_path,
                        'name': os.path.basename(file_info.filename)
                    })
        
        return image_paths, temp_dir
    except Exception as e:
        st.error(f"❌ Error saat extract ZIP: {str(e)}")
        return [], temp_dir

# --- Sidebar Informasi ---
with st.sidebar:
    st.header("📊 Informasi Model")
    st.write("**Model**: Random Forest (SET 3)")
    st.write("**Fitur**: Mean Color + GLCM + Histogram")
    st.write("**Jumlah Fitur**: 50 terpilih")
    st.write("**Akurasi Uji**: 72.5%")
    st.markdown("---")
    st.success("✅ Model berhasil dimuat!")
    st.info("Upload gambar untuk prediksi!")

# --- Tabs: Single vs Batch ---
tab1, tab2, tab3 = st.tabs(["🖼️ Upload Satu Gambar", "📁 Upload Multiple Files", "📦 Upload ZIP Folder"])

# --- Tab 1: Single Upload ---
with tab1:
    uploaded_file = st.file_uploader("Pilih gambar sampah...", type=["jpg", "jpeg", "png"], key="single")
    if uploaded_file:
        # Simpan file temporary dengan path yang lebih aman
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_single.jpg")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="Gambar yang Diupload", use_column_width=True)
        with col2:
            with st.spinner("Menganalisis..."):
                class_name, confidence, proba = predict_single_image(temp_path)
            if "✅" in class_name or "❌" in class_name:
                st.subheader("Hasil Prediksi:")
                if "Recycle" in class_name:
                    st.success(class_name)
                else:
                    st.error(class_name)
                st.write(f"**Confidence**: {confidence:.2f}%")
                
                # Bar chart probabilitas
                prob_df = pd.DataFrame({
                    'Kelas': ['Recycle', 'Non-Recycle'],
                    'Probabilitas (%)': [proba[0]*100, proba[1]*100]
                }).set_index('Kelas')
                st.bar_chart(prob_df)
            else:
                st.error(class_name)
        
        # Cleanup temporary file
        try:
            os.remove(temp_path)
        except:
            pass

# --- Tab 2: Multiple Files Upload ---
with tab2:
    st.info("💡 **Tip**: Pilih beberapa file sekaligus dengan menekan Ctrl (Windows) atau Cmd (Mac)")
    uploaded_files = st.file_uploader(
        "Pilih beberapa gambar...", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="multiple"
    )
    if uploaded_files:
        # Buat folder temporary
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        results = []
        temp_files = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Memproses {i+1}/{len(uploaded_files)}: {file.name}")
            
            temp_path = os.path.join(temp_dir, f"temp_batch_{i}.jpg")
            temp_files.append(temp_path)
            
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            class_name, confidence, proba = predict_single_image(temp_path)
            
            results.append({
                "Nama File": file.name,
                "Path": temp_path,
                "Prediksi": class_name,
                "Confidence (%)": confidence,
                "Probabilitas Recycle": proba[0],
                "Probabilitas Non-Recycle": proba[1]
            })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ Selesai!")
        
        # Preview Gambar dalam Grid
        st.subheader("🖼️ Preview Gambar")
        
        # Tampilkan dalam grid 4 kolom
        cols_per_row = 4
        for idx in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, col in enumerate(cols):
                result_idx = idx + col_idx
                if result_idx < len(results):
                    result = results[result_idx]
                    with col:
                        # Tampilkan gambar saja tanpa prediksi
                        img = Image.open(result["Path"])
                        st.image(img, caption=result['Nama File'], use_column_width=True)
        
        st.markdown("---")
        
        # Tabel Detail
        st.subheader("📋 Tabel Detail Hasil Prediksi")
        df = pd.DataFrame(results).drop(columns=['Path'])  # Hapus kolom Path dari tabel
        st.dataframe(df.style.format({
            "Confidence (%)": "{:.2f}",
            "Probabilitas Recycle": "{:.4f}",
            "Probabilitas Non-Recycle": "{:.4f}"
        }), use_container_width=True)
        
        # Statistik
        recycle_count = df['Prediksi'].str.contains('Recycle', case=False, na=False).sum()
        non_recycle_count = len(df) - recycle_count
        
        st.subheader("📊 Ringkasan")
        col1, col2, col3 = st.columns(3)
        col1.metric("📁 Total Gambar", len(df))
        col2.metric("♻️ Recycle", recycle_count)
        col3.metric("🗑️ Non-Recycle", non_recycle_count)
        
        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Unduh Hasil (CSV)",
            csv,
            "hasil_klasifikasi.csv",
            "text/csv"
        )
        
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

# --- Tab 3: ZIP Folder Upload ---
with tab3:
    st.info("💡 **Tip**: Compress folder gambar Anda menjadi file ZIP terlebih dahulu")
    st.markdown("""
    **Cara membuat ZIP:**
    - **Windows**: Klik kanan folder → Send to → Compressed (zipped) folder
    - **Mac**: Klik kanan folder → Compress
    - **Linux**: `zip -r folder.zip folder/`
    """)
    
    uploaded_zip = st.file_uploader("Upload file ZIP berisi gambar...", type=["zip"], key="zip")
    
    if uploaded_zip:
        with st.spinner("📦 Extracting ZIP file..."):
            # Simpan ZIP temporary
            temp_zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp.zip")
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            
            # Extract gambar
            image_files, temp_dir = extract_images_from_zip(temp_zip_path)
            
            # Hapus file ZIP temporary
            os.remove(temp_zip_path)
        
        if not image_files:
            st.error("❌ Tidak ada gambar ditemukan dalam ZIP file!")
            st.info("Pastikan ZIP berisi file dengan ekstensi: .jpg, .jpeg, .png")
        else:
            st.success(f"✅ Ditemukan {len(image_files)} gambar!")
            
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, img_info in enumerate(image_files):
                status_text.text(f"Memproses {i+1}/{len(image_files)}: {img_info['name']}")
                
                class_name, confidence, proba = predict_single_image(img_info['path'])
                
                results.append({
                    "Nama File": img_info['name'],
                    "Path": img_info['path'],
                    "Prediksi": class_name,
                    "Confidence (%)": confidence,
                    "Probabilitas Recycle": proba[0],
                    "Probabilitas Non-Recycle": proba[1]
                })
                
                progress_bar.progress((i + 1) / len(image_files))
            
            status_text.text("✅ Selesai!")
            
            # Preview Gambar dalam Grid
            st.subheader("🖼️ Preview Gambar")
            
            # Tampilkan dalam grid 4 kolom
            cols_per_row = 4
            for idx in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, col in enumerate(cols):
                    result_idx = idx + col_idx
                    if result_idx < len(results):
                        result = results[result_idx]
                        with col:
                            # Tampilkan gambar saja tanpa prediksi
                            try:
                                img = Image.open(result["Path"])
                                # Potong nama file jika terlalu panjang
                                display_name = result['Nama File']
                                if len(display_name) > 30:
                                    display_name = display_name[:27] + "..."
                                st.image(img, caption=display_name, use_column_width=True)
                            except Exception as e:
                                st.error(f"Error loading: {result['Nama File']}")
            
            st.markdown("---")
            
            # Tabel Detail
            st.subheader("📋 Tabel Detail Hasil Prediksi")
            df = pd.DataFrame(results).drop(columns=['Path'])
            st.dataframe(df.style.format({
                "Confidence (%)": "{:.2f}",
                "Probabilitas Recycle": "{:.4f}",
                "Probabilitas Non-Recycle": "{:.4f}"
            }), use_container_width=True)
            
            # Statistik
            recycle_count = df['Prediksi'].str.contains('Recycle', case=False, na=False).sum()
            non_recycle_count = len(df) - recycle_count
            
            st.subheader("📊 Ringkasan")
            col1, col2, col3 = st.columns(3)
            col1.metric("📁 Total Gambar", len(df))
            col2.metric("♻️ Recycle", recycle_count)
            col3.metric("🗑️ Non-Recycle", non_recycle_count)
            
            # Visualisasi distribusi
            st.subheader("📈 Distribusi Prediksi")
            fig, ax = plt.subplots(figsize=(8, 4))
            pred_counts = df['Prediksi'].value_counts()
            colors = ['#10b981' if 'Recycle' in label else '#ef4444' for label in pred_counts.index]
            pred_counts.plot(kind='bar', ax=ax, color=colors)
            ax.set_ylabel('Jumlah')
            ax.set_xlabel('Prediksi')
            ax.set_title('Distribusi Hasil Prediksi')
            plt.xticks(rotation=0)
            st.pyplot(fig)
            
            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Unduh Hasil (CSV)",
                csv,
                "hasil_klasifikasi_zip.csv",
                "text/csv"
            )
            
            # Cleanup
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

# --- Footer ---
st.markdown("---")
st.caption("© 2025 Projek Klasifikasi Sampah — Random Forest + SET 3 Features")
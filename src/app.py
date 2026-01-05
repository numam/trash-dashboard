# -*- coding: utf-8 -*-
"""
♻️ DASHBOARD KLASIFIKASI SAMPAH - Versi Modern & User Friendly
Aplikasi untuk memilah sampah daur ulang dan non-daur ulang
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
from PIL import Image
import zipfile
import shutil
from pathlib import Path

# --- KONFIGURASI TAMPILAN MODERN ---
st.set_page_config(
    page_title="Klasifikasi Sampah Pintar",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom untuk tampilan lebih menarik
st.markdown("""
<style>
    /* Background dan warna utama */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        color: white;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px 10px 0 0;
    }
    
    /* Image containers */
    .image-container {
        border: 3px solid #e0e0e0;
        border-radius: 12px;
        padding: 0.5rem;
        background: white;
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        border-color: #667eea;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER MODERN ---
st.markdown("""
<div class="header-container">
    <h1 style='text-align: center; font-size: 3rem; margin-bottom: 1rem;'>
        ♻️ Sistem Klasifikasi Sampah Pintar
    </h1>
    <p style='text-align: center; font-size: 1.3rem; opacity: 0.95;'>
        Teknologi AI untuk Membantu Memilah Sampah dengan Mudah dan Cepat
    </p>
</div>
""", unsafe_allow_html=True)

# --- Muat Model dan Preprocessor ---
@st.cache_resource
def load_models():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        rf_model_path = os.path.join(current_dir, "rf_set3_model.pkl")
        scaler_path = os.path.join(current_dir, "scaler_set3.pkl")
        selector_path = os.path.join(current_dir, "selector_set3.pkl")
        
        if not os.path.exists(rf_model_path):
            st.error(f"❌ File model tidak ditemukan: {rf_model_path}")
            st.info("📁 Pastikan file model berada di folder yang sama dengan aplikasi")
            st.stop()
        
        if not os.path.exists(scaler_path):
            st.error(f"❌ File scaler tidak ditemukan: {scaler_path}")
            st.stop()
            
        if not os.path.exists(selector_path):
            st.error(f"❌ File selector tidak ditemukan: {selector_path}")
            st.stop()
        
        rf_model = joblib.load(rf_model_path)
        scaler = joblib.load(scaler_path)
        selector = joblib.load(selector_path)
        
        return rf_model, scaler, selector
        
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memuat model: {str(e)}")
        st.stop()

rf_model, scaler, selector = load_models()

# --- Fungsi Ekstraksi Fitur ---
def extract_set3_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar tidak dapat dibaca: {image_path}")
    
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mean_color = np.mean(img, axis=(0, 1))
    
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
    
    hist_vals, _ = exposure.histogram(gray)
    if len(hist_vals) < 64:
        hist_64 = np.pad(hist_vals, (0, 64 - len(hist_vals)), mode='constant')
    else:
        hist_64 = hist_vals[:64]
    hist_64 = hist_64 / hist_64.sum() if hist_64.sum() > 0 else hist_64
    
    features = np.concatenate([mean_color, glcm_props, hist_64])
    return features

def predict_single_image(image_path):
    try:
        features = extract_set3_features(image_path)
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_selected = selector.transform(features_scaled)
        pred = rf_model.predict(features_selected)[0]
        proba = rf_model.predict_proba(features_selected)[0]
        
        class_name = "Recycle" if pred == 0 else "Non-Recycle"
        confidence = proba[int(pred)] * 100
        return class_name, confidence, proba
    except Exception as e:
        return f"Error: {str(e)}", 0, [0, 0]

def extract_images_from_zip(zip_file):
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_batch")
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.is_dir() or file_info.filename.startswith('__MACOSX'):
                    continue
                
                file_ext = Path(file_info.filename).suffix.lower()
                if file_ext in image_extensions:
                    extracted_path = zip_ref.extract(file_info, temp_dir)
                    image_paths.append({
                        'path': extracted_path,
                        'name': os.path.basename(file_info.filename)
                    })
        
        return image_paths, temp_dir
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membuka file ZIP: {str(e)}")
        return [], temp_dir

# --- SIDEBAR INFORMASI ---
with st.sidebar:
    st.markdown("### 📊 Informasi Sistem")
    
    st.markdown("""
    <div class="card">
        <h4>🤖 Tentang AI</h4>
        <p><strong>Model:</strong> Random Forest</p>
        <p><strong>Akurasi:</strong> 72.5%</p>
        <p><strong>Fitur:</strong> 50 fitur terpilih</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("✅ Sistem siap digunakan!")
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h4>💡 Tips Penggunaan</h4>
        <ul>
            <li>Pastikan foto jelas dan tidak buram</li>
            <li>Ambil foto dari jarak yang cukup dekat</li>
            <li>Gunakan pencahayaan yang baik</li>
            <li>Untuk hasil terbaik, foto fokus pada satu jenis sampah</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📚 Panduan Cepat
    
    **🖼️ Satu Gambar:**
    Untuk memilah satu sampah saja
    
    **📁 Banyak Gambar:**
    Pilih beberapa foto sekaligus
    
    **📦 Folder ZIP:**
    Upload seluruh folder dalam bentuk ZIP
    """)

# --- TABS DENGAN ICON BESAR ---
tab1, tab2, tab3 = st.tabs([
    "🖼️  SATU GAMBAR",
    "📁  BANYAK GAMBAR", 
    "📦  FOLDER ZIP"
])

# --- TAB 1: UPLOAD SATU GAMBAR ---
with tab1:
    st.markdown("""
    <div class="info-box">
        <h3>📸 Upload Foto Sampah Anda</h3>
        <p style='font-size: 1.1rem;'>Ambil atau pilih foto sampah untuk mengetahui apakah bisa didaur ulang atau tidak</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Klik di sini untuk memilih gambar", 
        type=["jpg", "jpeg", "png"],
        key="single",
        help="Format yang didukung: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_single.jpg")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### 📷 Gambar Anda")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔍 Hasil Analisis")
            
            with st.spinner("🤖 AI sedang menganalisis gambar..."):
                class_name, confidence, proba = predict_single_image(temp_path)
            
            if "Error" not in class_name:
                # Tampilkan hasil dengan animasi
                if class_name == "Recycle":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; 
                                box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);'>
                        <h1 style='color: white; font-size: 3rem; margin: 0;'>♻️</h1>
                        <h2 style='color: white; margin: 0.5rem 0;'>BISA DIDAUR ULANG!</h2>
                        <p style='color: white; font-size: 1.2rem; opacity: 0.95;'>
                            Tingkat Keyakinan: {:.1f}%
                        </p>
                    </div>
                    """.format(confidence), unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="info-box" style='margin-top: 1.5rem; background: #10b98115;'>
                        <h4>✅ Apa yang harus dilakukan?</h4>
                        <ul style='font-size: 1.1rem;'>
                            <li>Bersihkan sampah dari sisa makanan/kotoran</li>
                            <li>Keringkan jika basah</li>
                            <li>Masukkan ke tempat sampah <strong>DAUR ULANG</strong></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center;
                                box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);'>
                        <h1 style='color: white; font-size: 3rem; margin: 0;'>🗑️</h1>
                        <h2 style='color: white; margin: 0.5rem 0;'>TIDAK BISA DIDAUR ULANG</h2>
                        <p style='color: white; font-size: 1.2rem; opacity: 0.95;'>
                            Tingkat Keyakinan: {:.1f}%
                        </p>
                    </div>
                    """.format(confidence), unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="info-box" style='margin-top: 1.5rem; background: #ef444415;'>
                        <h4>❌ Apa yang harus dilakukan?</h4>
                        <ul style='font-size: 1.1rem;'>
                            <li>Masukkan ke tempat sampah <strong>BIASA</strong></li>
                            <li>Jangan campur dengan sampah daur ulang</li>
                            <li>Pastikan tertutup rapat</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Grafik probabilitas
                st.markdown("### 📊 Detail Probabilitas")
                prob_df = pd.DataFrame({
                    'Jenis': ['♻️ Daur Ulang', '🗑️ Tidak Daur Ulang'],
                    'Probabilitas': [proba[0]*100, proba[1]*100]
                })
                
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['#10b981', '#ef4444']
                bars = ax.barh(prob_df['Jenis'], prob_df['Probabilitas'], color=colors, height=0.6)
                
                # Tambah nilai di ujung bar
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                           f'{width:.1f}%', ha='left', va='center', fontsize=12, fontweight='bold')
                
                ax.set_xlabel('Probabilitas (%)', fontsize=12, fontweight='bold')
                ax.set_xlim(0, 110)
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.error(f"❌ {class_name}")
        
        try:
            os.remove(temp_path)
        except:
            pass

# --- TAB 2: UPLOAD BANYAK GAMBAR ---
with tab2:
    st.markdown("""
    <div class="info-box">
        <h3>📁 Upload Banyak Gambar Sekaligus</h3>
        <p style='font-size: 1.1rem;'>Pilih beberapa foto sampah untuk dianalisis bersamaan</p>
        <p><strong>💡 Cara:</strong> Tekan tombol Ctrl (Windows) atau Cmd (Mac) sambil memilih file</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Pilih beberapa gambar sekaligus", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="multiple",
        help="Pilih lebih dari satu file dengan Ctrl+Click atau Cmd+Click"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} gambar berhasil dipilih!")
        
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        results = []
        temp_files = []
        
        # Progress bar yang lebih menarik
        st.markdown("### 🔄 Proses Analisis")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                        padding: 1rem; border-radius: 10px; text-align: center;'>
                <p style='font-size: 1.2rem; margin: 0;'>
                    🔍 Memproses: <strong>{file.name}</strong> ({i+1}/{len(uploaded_files)})
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            temp_path = os.path.join(temp_dir, f"temp_batch_{i}.jpg")
            temp_files.append(temp_path)
            
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            class_name, confidence, proba = predict_single_image(temp_path)
            
            # Tambahkan emoji untuk hasil
            display_pred = f"♻️ {class_name}" if class_name == "Recycle" else f"🗑️ {class_name}"
            
            results.append({
                "Nama File": file.name,
                "Path": temp_path,
                "Prediksi": display_pred,
                "Tingkat Keyakinan": f"{confidence:.1f}%",
                "Prob. Recycle": proba[0],
                "Prob. Non-Recycle": proba[1]
            })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.success("✅ Semua gambar berhasil dianalisis!")
        
        # Preview gambar dalam grid
        st.markdown("### 🖼️ Galeri Gambar")
        
        cols_per_row = 4
        for idx in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, col in enumerate(cols):
                result_idx = idx + col_idx
                if result_idx < len(results):
                    result = results[result_idx]
                    with col:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        img = Image.open(result["Path"])
                        st.image(img, use_column_width=True)
                        
                        # Badge hasil
                        if "♻️" in result["Prediksi"]:
                            badge_color = "#10b981"
                            badge_text = "BISA DIDAUR ULANG"
                        else:
                            badge_color = "#ef4444"
                            badge_text = "TIDAK BISA DIDAUR ULANG"
                        
                        st.markdown(f"""
                        <div style='background: {badge_color}; color: white; padding: 0.5rem; 
                                    text-align: center; border-radius: 8px; font-weight: bold; 
                                    font-size: 0.85rem; margin-top: 0.5rem;'>
                            {badge_text}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Nama file (dipendekkan jika panjang)
                        display_name = result['Nama File']
                        if len(display_name) > 25:
                            display_name = display_name[:22] + "..."
                        st.caption(display_name)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabel hasil
        st.markdown("### 📋 Tabel Detail Hasil")
        df = pd.DataFrame(results).drop(columns=['Path', 'Prob. Recycle', 'Prob. Non-Recycle'])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Statistik dengan desain menarik
        recycle_count = sum(1 for r in results if "♻️" in r["Prediksi"])
        non_recycle_count = len(results) - recycle_count
        
        st.markdown("### 📊 Ringkasan Hasil")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="card" style='text-align: center; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);'>
                <h1 style='font-size: 3rem; margin: 0; color: #667eea;'>{len(results)}</h1>
                <p style='font-size: 1.2rem; margin: 0.5rem 0; font-weight: 600;'>Total Gambar</p>
                <p style='font-size: 2rem; margin: 0;'>📁</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card" style='text-align: center; background: linear-gradient(135deg, #10b98115 0%, #05966915 100%);'>
                <h1 style='font-size: 3rem; margin: 0; color: #10b981;'>{recycle_count}</h1>
                <p style='font-size: 1.2rem; margin: 0.5rem 0; font-weight: 600;'>Bisa Didaur Ulang</p>
                <p style='font-size: 2rem; margin: 0;'>♻️</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card" style='text-align: center; background: linear-gradient(135deg, #ef444415 0%, #dc262615 100%);'>
                <h1 style='font-size: 3rem; margin: 0; color: #ef4444;'>{non_recycle_count}</h1>
                <p style='font-size: 1.2rem; margin: 0.5rem 0; font-weight: 600;'>Tidak Bisa Didaur Ulang</p>
                <p style='font-size: 2rem; margin: 0;'>🗑️</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Grafik pie chart
        st.markdown("### 📈 Distribusi Hasil")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sizes = [recycle_count, non_recycle_count]
        labels = ['♻️ Bisa Didaur Ulang', '🗑️ Tidak Bisa Didaur Ulang']
        colors = ['#10b981', '#ef4444']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                           startangle=90, explode=explode, textprops={'fontsize': 12, 'weight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_weight('bold')
        
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download hasil
        csv_df = pd.DataFrame(results).drop(columns=['Path'])
        csv = csv_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "📥 Unduh Hasil (CSV)",
            csv,
            "hasil_klasifikasi.csv",
            "text/csv",
            use_container_width=True
        )
        
        # Cleanup
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

# --- TAB 3: UPLOAD ZIP ---
with tab3:
    st.markdown("""
    <div class="info-box">
        <h3>📦 Upload Folder dalam Bentuk ZIP</h3>
        <p style='font-size: 1.1rem;'>Compress folder yang berisi banyak foto sampah menjadi file ZIP</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📚 Cara Membuat File ZIP", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🪟 Windows:**
            1. Klik kanan pada folder
            2. Pilih "Send to"
            3. Klik "Compressed (zipped) folder"
            """)
        
        with col2:
            st.markdown("""
            **🍎 Mac:**
            1. Klik kanan pada folder
            2. Pilih "Compress"
            3. File ZIP akan otomatis dibuat
            """)
        
        with col3:
            st.markdown("""
            **🐧 Linux:**
            ```
            zip -r folder.zip folder/
            ```
            """)
    
    uploaded_zip = st.file_uploader(
        "Pilih file ZIP yang berisi gambar", 
        type=["zip"], 
        key="zip",
        help="File ZIP harus berisi gambar dengan format JPG, JPEG, atau PNG"
    )
    
    if uploaded_zip:
        with st.spinner("📦 Membuka file ZIP..."):
            temp_zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp.zip")
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            
            image_files, temp_dir = extract_images_from_zip(temp_zip_path)
            os.remove(temp_zip_path)
        
        if not image_files:
            st.error("❌ Tidak ada gambar ditemukan dalam file ZIP!")
            st.info("📝 Pastikan ZIP berisi file dengan format: JPG, JPEG, atau PNG")
        else:
            st.success(f"✅ Berhasil menemukan {len(image_files)} gambar!")
            
            results = []
            
            st.markdown("### 🔄 Proses Analisis")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, img_info in enumerate(image_files):
                status_text.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center;'>
                    <p style='font-size: 1.2rem; margin: 0;'>
                        🔍 Memproses: <strong>{img_info['name']}</strong> ({i+1}/{len(image_files)})
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                class_name, confidence, proba = predict_single_image(img_info['path'])
                
                display_pred = f"♻️ {class_name}" if class_name == "Recycle" else f"🗑️ {class_name}"
                
                results.append({
                    "Nama File": img_info['name'],
                    "Path": img_info['path'],
                    "Prediksi": display_pred,
                    "Tingkat Keyakinan": f"{confidence:.1f}%",
                    "Prob. Recycle": proba[0],
                    "Prob. Non-Recycle": proba[1]
                })
                
                progress_bar.progress((i + 1) / len(image_files))
            
            status_text.success("✅ Semua gambar berhasil dianalisis!")
            
            # Galeri gambar
            st.markdown("### 🖼️ Galeri Gambar")
            
            cols_per_row = 4
            for idx in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_idx, col in enumerate(cols):
                    result_idx = idx + col_idx
                    if result_idx < len(results):
                        result = results[result_idx]
                        with col:
                            try:
                                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                                img = Image.open(result["Path"])
                                st.image(img, use_column_width=True)
                                
                                if "♻️" in result["Prediksi"]:
                                    badge_color = "#10b981"
                                    badge_text = "BISA DIDAUR ULANG"
                                else:
                                    badge_color = "#ef4444"
                                    badge_text = "TIDAK BISA DIDAUR ULANG"
                                
                                st.markdown(f"""
                                <div style='background: {badge_color}; color: white; padding: 0.5rem; 
                                            text-align: center; border-radius: 8px; font-weight: bold; 
                                            font-size: 0.85rem; margin-top: 0.5rem;'>
                                    {badge_text}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                display_name = result['Nama File']
                                if len(display_name) > 25:
                                    display_name = display_name[:22] + "..."
                                st.caption(display_name)
                                st.markdown('</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error: {result['Nama File']}")
            
            st.markdown("---")
            
            # Tabel hasil
            st.markdown("### 📋 Tabel Detail Hasil")
            df = pd.DataFrame(results).drop(columns=['Path', 'Prob. Recycle', 'Prob. Non-Recycle'])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Statistik
            recycle_count = sum(1 for r in results if "♻️" in r["Prediksi"])
            non_recycle_count = len(results) - recycle_count
            
            st.markdown("### 📊 Ringkasan Hasil")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="card" style='text-align: center; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);'>
                    <h1 style='font-size: 3rem; margin: 0; color: #667eea;'>{len(results)}</h1>
                    <p style='font-size: 1.2rem; margin: 0.5rem 0; font-weight: 600;'>Total Gambar</p>
                    <p style='font-size: 2rem; margin: 0;'>📁</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="card" style='text-align: center; background: linear-gradient(135deg, #10b98115 0%, #05966915 100%);'>
                    <h1 style='font-size: 3rem; margin: 0; color: #10b981;'>{recycle_count}</h1>
                    <p style='font-size: 1.2rem; margin: 0.5rem 0; font-weight: 600;'>Bisa Didaur Ulang</p>
                    <p style='font-size: 2rem; margin: 0;'>♻️</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="card" style='text-align: center; background: linear-gradient(135deg, #ef444415 0%, #dc262615 100%);'>
                    <h1 style='font-size: 3rem; margin: 0; color: #ef4444;'>{non_recycle_count}</h1>
                    <p style='font-size: 1.2rem; margin: 0.5rem 0; font-weight: 600;'>Tidak Bisa Didaur Ulang</p>
                    <p style='font-size: 2rem; margin: 0;'>🗑️</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Grafik distribusi
            st.markdown("### 📈 Distribusi Hasil")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sizes = [recycle_count, non_recycle_count]
            labels = ['♻️ Bisa Didaur Ulang', '🗑️ Tidak Bisa Didaur Ulang']
            colors = ['#10b981', '#ef4444']
            explode = (0.05, 0.05)
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                               startangle=90, explode=explode, textprops={'fontsize': 12, 'weight': 'bold'})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(14)
                autotext.set_weight('bold')
            
            ax.axis('equal')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download hasil
            csv_df = pd.DataFrame(results).drop(columns=['Path'])
            csv = csv_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                "📥 Unduh Hasil (CSV)",
                csv,
                "hasil_klasifikasi_zip.csv",
                "text/csv",
                use_container_width=True
            )
            
            # Cleanup
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
            border-radius: 12px; margin-top: 2rem;'>
    <h3>♻️ Sistem Klasifikasi Sampah Pintar</h3>
    <p style='font-size: 1.1rem; margin: 0.5rem 0;'>
        Membantu Anda memilah sampah dengan mudah menggunakan teknologi AI
    </p>
    <p style='margin: 0.5rem 0; opacity: 0.8;'>
        © 2025 Projek Klasifikasi Sampah | Powered by Random Forest AI
    </p>
</div>
""", unsafe_allow_html=True)
# -*- coding: utf-8 -*-
"""
TRAIN MODEL - Random Forest dengan SET 3 (Mean Color + GLCM + Histogram)
Versi stabil: menangani gambar grayscale, error baca, dan bentuk fitur tidak konsisten.
"""

import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage import exposure

# === Konfigurasi ===
DATASET_DIR = 'src'
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
K_FEATURES = 50

print("📦 Memulai proses pelatihan model...")

recycle_dir = os.path.join(DATASET_DIR, 'recycle')
non_recycle_dir = os.path.join(DATASET_DIR, 'non-recycle')

if not os.path.exists(recycle_dir):
    raise FileNotFoundError(f"Folder 'recycle' tidak ditemukan di {recycle_dir}")
if not os.path.exists(non_recycle_dir):
    raise FileNotFoundError(f"Folder 'non-recycle' tidak ditemukan di {non_recycle_dir}")

def extract_set3_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Tidak dapat membaca gambar: {image_path}")

    # Pastikan gambar dalam format BGR → RGB, dan 3 channel
    if img.ndim == 2:
        # Gambar grayscale → konversi ke RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # RGBA → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mean Color (R, G, B)
    mean_color = np.mean(img, axis=(0, 1))  # Selalu 3 nilai

    # GLCM dari grayscale
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

    # Histogram: pastikan panjang 64
    hist_vals, _ = exposure.histogram(gray)
    if len(hist_vals) < 64:
        hist_64 = np.pad(hist_vals, (0, 64 - len(hist_vals)), mode='constant')
    else:
        hist_64 = hist_vals[:64]
    hist_64 = hist_64 / hist_64.sum() if hist_64.sum() > 0 else hist_64

    features = np.concatenate([mean_color, glcm_props, hist_64])
    return features

# === Muat dataset ===
print("🔍 Memuat dan mengekstrak fitur dari dataset...")

X, y = [], []
skipped = []

# Recycle (label 0)
for filename in os.listdir(recycle_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(recycle_dir, filename)
        try:
            feat = extract_set3_features(path)
            X.append(feat)
            y.append(0)
        except Exception as e:
            print(f"⚠️ Skip {path}: {e}")
            skipped.append(path)

# Non-recycle (label 1)
for filename in os.listdir(non_recycle_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(non_recycle_dir, filename)
        try:
            feat = extract_set3_features(path)
            X.append(feat)
            y.append(1)
        except Exception as e:
            print(f"⚠️ Skip {path}: {e}")
            skipped.append(path)

if len(X) == 0:
    raise RuntimeError("Tidak ada gambar yang berhasil dimuat! Periksa folder dataset.")

X = np.array(X, dtype=np.float32)  # Force tipe data konsisten
y = np.array(y)

print(f"✅ Total gambar berhasil dimuat: {len(X)}")
print(f"   - Recycle: {np.sum(y == 0)}")
print(f"   - Non-Recycle: {np.sum(y == 1)}")
print(f"   - Dimensi fitur: {X.shape[1]}")
if skipped:
    print(f"   - Gambar di-skip: {len(skipped)}")

# === Preprocessing ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

k = min(K_FEATURES, X_scaled.shape[1])
selector = SelectKBest(chi2, k=k)
X_selected = selector.fit_transform(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# === Training ===
print("🧠 Melatih model Random Forest...")
rf_model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_train, y_train)

# === Evaluasi ===
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print("\n📊 Hasil Evaluasi:")
print(f"   - Akurasi Training: {train_acc:.4f}")
print(f"   - Akurasi Testing:  {test_acc:.4f}")
print(f"   - Overfitting:      {train_acc - test_acc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=['Recycle', 'Non-Recycle']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Recycle', 'Non-Recycle'],
            yticklabels=['Recycle', 'Non-Recycle'])
plt.title('Confusion Matrix - Random Forest (SET 3)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_set3.png')
print("✅ Confusion matrix disimpan sebagai: confusion_matrix_set3.png")

# === Simpan ===
joblib.dump(rf_model, 'rf_set3_model.pkl')
joblib.dump(scaler, 'scaler_set3.pkl')
joblib.dump(selector, 'selector_set3.pkl')

print("\n🎉 Pelatihan selesai!")
print("File yang dihasilkan:")
print("  - rf_set3_model.pkl")
print("  - scaler_set3.pkl")
print("  - selector_set3.pkl")
print("  - confusion_matrix_set3.png")
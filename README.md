# 🌾 PadiScan — Klasifikasi Penyakit Daun Padi (Vision Transformer) Berbasis Web

Sistem berbasis web untuk mendeteksi dan mengklasifikasikan 10 jenis kondisi/penyakit daun tanaman padi menggunakan arsitektur **Vision Transformer (ViT-B/16)**. Proyek ini dikembangkan sebagai bagian dari penelitian skripsi.

> 🌐 **Live Demo (Frontend Vercel):** [https://skripsipaddy-frontend.vercel.app](https://skripsipaddy-frontend.vercel.app)

---

## 📌 Fitur Utama
- **Klasifikasi Citra ViT:** Model Vision Transformer (`vit_base_patch16_224`) dengan tingkat akurasi uji mencapai **97.57%** dan validasi **97.76%**.
- **10 Kelas Penyakit:** Hawar Daun Bakteri (*Bacterial Leaf Blight*), Garis Daun Bakteri (*Bacterial Leaf Streak*), Hawar Malai (*Bacterial Panicle Blight*), Blas (*Blast*), Bercak Cokelat (*Brown Spot*), Sundep/Beluk (*Dead Heart*), Embun Bulu (*Downy Mildew*), Hama Hispa (*Hispa*), Daun Sehat (*Normal*), dan Tungro.
- **Informasi Agrikultur Lengkap:** Menampilkan tingkat keparahan, penyebab, gejala visual, dan rekomendasi penanganan sesuai standar pertanian.
- **Arsitektur Web Modern:**
  - **Frontend:** HTML5, CSS3 kustom, Vanilla JavaScript (di-hosting di Vercel).
  - **Backend:** FastAPI (Python) dengan inferensi PyTorch berkecepatan tinggi.

---

## 🏗️ Struktur Proyek

```text
Skripsi_paddy/
├── train_vit_padi.py          # Script pelatihan model ViT (Google Colab / GPU T4)
├── Deteksi_Penyakit_Tanaman_Padi.ipynb # Notebook eksperimen pelatihan
├── hasil_training/            # Laporan evaluasi (Confusion Matrix, Kurva Pelatihan, Classification Report)
├── jalankan_server_online.bat # Script 1-klik untuk menjalankan backend + Cloudflare Tunnel
└── web/
    ├── backend/               # Server FastAPI + Inferensi PyTorch
    │   ├── main.py            # Endpoint API: /predict, /health, /diseases, /classes
    │   ├── diseases.py        # Knowledge base 10 penyakit (Bahasa Indonesia)
    │   ├── requirements.txt   # Dependensi backend
    │   └── model/             # Bobot model terlatih (vit_paddy_best.pth) & metadata
    └── frontend/              # Antarmuka pengguna (HTML/CSS/JS)
```

---

## 📊 Hasil Evaluasi Model

| Metrik | Nilai |
|---|---|
| **Akurasi Data Uji (Test Accuracy)** | **97.57%** |
| **Akurasi Validasi Terbaik (Best Val Acc)** | **97.76%** |
| **Macro Precision** | **97.25%** |
| **Macro Recall** | **97.55%** |
| **Macro F1-Score** | **97.39%** |

Grafik evaluasi model (Kurva Loss/Akurasi dan Confusion Matrix) dapat dilihat pada folder `hasil_training/`.

---

## 🚀 Cara Menjalankan Secara Lokal

### 1. Prasyarat
- Python 3.10+
- Bobot model `vit_paddy_best.pth` di dalam `web/backend/model/`

### 2. Jalankan Backend (FastAPI)
```powershell
cd web\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Buka peramban di `http://localhost:8000`.

### 3. Menjalankan Mode Online (Cloudflare Tunnel)
Cukup jalankan file `jalankan_server_online.bat` untuk menghubungkan backend lokal ke frontend Vercel secara publik.

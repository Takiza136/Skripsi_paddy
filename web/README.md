# PadiScan — Klasifikasi Penyakit Daun Padi (Vision Transformer) Berbasis Web

Sistem web untuk mengklasifikasikan penyakit daun padi dari satu foto. Fungsi web dibatasi pada
**unggah citra → inferensi ViT → tampil label penyakit + confidence score** (sesuai batasan masalah proposal).

Struktur:

```
Skripsi/
├── train_vit_padi.py          # Pipeline training (jalankan di Google Colab)
└── web/
    ├── backend/               # FastAPI + inferensi ViT
    │   ├── main.py            # endpoint /predict, /health, /diseases, /classes
    │   ├── diseases.py        # info 10 kelas penyakit (Bahasa Indonesia)
    │   ├── requirements.txt
    │   └── model/             # <-- taruh hasil training di sini
    └── frontend/              # Halaman deteksi (HTML/CSS/JS), disajikan oleh FastAPI
```

## Alur kerja

### 1. Latih model (Google Colab, GPU T4)
1. Upload `paddy-disease-classification.zip` ke Google Drive (`Colab Notebooks/`).
2. Buka `train_vit_padi.py` di Colab. Salin tiap blok `# %%` menjadi satu cell, jalankan berurutan.
3. Hasil di Drive `Colab Notebooks/vit_padi_output/`:
   `vit_paddy_best.pth`, `class_names.json`, `model_info.json`,
   plus `confusion_matrix_test.png`, `classification_report_test.txt`, `kurva_training.png` untuk laporan Bab 4.

### 2. Pasang model di web
Salin 3 file (`vit_paddy_best.pth`, `class_names.json`, `model_info.json`) ke `web/backend/model/`.

### 3. Jalankan web (butuh Python 3.10+)
```powershell
cd web\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Buka http://localhost:8000 — unggah foto daun padi (klik area unggah, drag & drop, atau paste), lalu tekan **Scan Sekarang**.

Cek status model: http://localhost:8000/health

## API ringkas
| Method | Path | Keterangan |
|---|---|---|
| POST | `/predict` | multipart field `file` (JPG/PNG/WebP ≤ 10 MB) → label, confidence, top-3, info penyakit |
| GET | `/health` | status server & model |
| GET | `/classes` | daftar kelas |
| GET | `/diseases` | info semua penyakit |

Contoh respons `/predict`:
```json
{
  "label": "tungro",
  "confidence": 0.9681,
  "low_confidence": false,
  "top": [{"label": "tungro", "name": "Tungro", "confidence": 0.9681}],
  "info": {"name": "Tungro", "severity": "high", "symptoms": ["..."], "treatment": ["..."]}
}
```

## Konfigurasi (environment variable, opsional)
`MODEL_PATH`, `CLASS_NAMES_PATH`, `MODEL_INFO_PATH`, `MAX_UPLOAD_MB` (10), `LOW_CONF_THRESHOLD` (0.60), `ALLOWED_ORIGINS`.

## Pengujian Black Box (bahan Bab 4)
| No | Skenario | Input | Hasil yang diharapkan |
|---|---|---|---|
| 1 | Unggah gambar valid | JPG daun padi, tekan Scan | Muncul label, confidence, top-3, deskripsi/gejala/penanganan |
| 2 | Format salah | PDF/TXT | Ditolak: "Format tidak didukung" |
| 3 | File terlalu besar | > 10 MB | Ditolak: "Ukuran file melebihi 10 MB" |
| 4 | File rusak | Ekstensi .jpg tetapi isi bukan gambar | Ditolak: "File bukan gambar yang valid" |
| 5 | Tanpa gambar | Buka halaman tanpa memilih file | Tombol Scan nonaktif |
| 6 | Gambar bukan padi / buram | Foto objek lain | Muncul peringatan keyakinan rendah bila < 60% |
| 7 | Server mati | Backend dihentikan | Pesan "Tidak dapat terhubung ke server" |
| 8 | Drag & drop / paste | Seret atau tempel gambar | Pratinjau muncul dan bisa di-scan |
| 9 | Ganti foto | Klik "Ganti foto" setelah ada hasil | Hasil lama dibersihkan, foto baru siap di-scan |

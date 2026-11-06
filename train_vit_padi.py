# -*- coding: utf-8 -*-
"""
Pipeline Training ViT - Klasifikasi Penyakit Daun Padi (10 kelas Paddy Doctor)

Versi perbaikan dari notebook lama. Dijalankan di Google Colab (GPU T4).
Setiap blok "# %%" adalah satu cell: copy ke cell notebook Colab, atau
upload file ini ke Colab dan jalankan per-cell.

Perbaikan dibanding versi sebelumnya:
  1. Transform train & val dipisah (dulu tertimpa: augmentasi train tidak aktif).
  2. Split Train/Val/Test = 70/15/15, STRATIFIED, seed tetap (reproducible).
  3. Normalisasi mengikuti konfigurasi bawaan model (timm), bukan ditebak.
  4. Class weight di loss karena kelas tidak seimbang.
  5. Scheduler cosine + mixed precision + simpan model TERBAIK (berdasar val).
  6. Evaluasi akhir HANYA di test set (sesuai proposal).
  7. Output siap pakai web: vit_paddy_best.pth, class_names.json, model_info.json.
"""

# %% Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm', 'scikit-learn', 'matplotlib', 'seaborn'])

import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

# %% Cell 2: Konfigurasi & Seed
import os, json, time, random, zipfile
import numpy as np

CONFIG = {
    "seed": 42,
    "model_name": "vit_base_patch16_224",
    "img_size": 224,
    "batch_size": 32,
    "epochs": 15,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "zip_path": "/content/drive/MyDrive/Colab Notebooks/paddy-disease-classification.zip",
    "extract_dir": "/content/paddy_data",
    "output_dir": "/content/drive/MyDrive/Colab Notebooks/vit_padi_output",
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Cell 3: Unzip dataset
if not os.path.exists(CONFIG["extract_dir"]):
    assert os.path.exists(CONFIG["zip_path"]), "File zip tidak ditemukan, cek zip_path!"
    with zipfile.ZipFile(CONFIG["zip_path"], 'r') as z:
        z.extractall(CONFIG["extract_dir"])
    print("Unzip selesai!")
else:
    print("Dataset sudah di-extract.")

# Cari folder train_images (struktur zip bisa berbeda-beda)
data_dir = None
for root, dirs, _ in os.walk(CONFIG["extract_dir"]):
    if os.path.basename(root) == 'train_images':
        data_dir = root
        break
assert data_dir is not None, "Folder train_images tidak ditemukan."
print("data_dir:", data_dir)

# %% Cell 4: Model (dibuat dulu supaya normalisasi mengikuti konfigurasi model)
import timm
from timm.data import resolve_data_config

# Dataset dimuat dulu untuk tahu jumlah kelas
from torchvision import datasets
probe_ds = datasets.ImageFolder(root=data_dir)
class_names = probe_ds.classes
num_classes = len(class_names)
targets = np.array(probe_ds.targets)
print(f"Total gambar: {len(probe_ds)} | Jumlah kelas: {num_classes}")
for i, c in enumerate(class_names):
    print(f"  {i}: {c:28s} {int((targets == i).sum())}")

model = timm.create_model(CONFIG["model_name"], pretrained=True, num_classes=num_classes)
data_cfg = resolve_data_config({}, model=model)
MEAN, STD = list(data_cfg["mean"]), list(data_cfg["std"])
print("Normalisasi model -> mean:", MEAN, "std:", STD)
model = model.to(device)

# %% Cell 5: Split stratified + DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

idx_all = np.arange(len(targets))
idx_trainval, idx_test = train_test_split(
    idx_all, test_size=CONFIG["test_ratio"], stratify=targets, random_state=CONFIG["seed"])
val_rel = CONFIG["val_ratio"] / (1 - CONFIG["test_ratio"])
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=val_rel, stratify=targets[idx_trainval], random_state=CONFIG["seed"])
print(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

S = CONFIG["img_size"]
train_transform = transforms.Compose([
    transforms.Resize((S, S)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(contrast=0.2),   # penyesuaian kontras (sesuai proposal)
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_transform = transforms.Compose([
    transforms.Resize((S, S)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Dua instance ImageFolder berbeda -> transform train tidak menimpa transform val/test
ds_train = Subset(datasets.ImageFolder(data_dir, transform=train_transform), idx_train)
ds_val   = Subset(datasets.ImageFolder(data_dir, transform=eval_transform), idx_val)
ds_test  = Subset(datasets.ImageFolder(data_dir, transform=eval_transform), idx_test)

train_loader = DataLoader(ds_train, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(ds_val,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(ds_test,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

# %% Cell 6: Loss (class weight), Optimizer, Scheduler
import torch.nn as nn

train_targets = targets[idx_train]
counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
class_weights = counts.sum() / (num_classes * counts)          # kelas langka -> bobot besar
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

# %% Cell 7: Training loop (simpan model terbaik)
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

best_val_acc, history = 0.0, []
best_path = os.path.join(CONFIG["output_dir"], "vit_paddy_best.pth")

print("Mulai training ViT...\n" + "=" * 50)
for epoch in range(CONFIG["epochs"]):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)
    scheduler.step()
    history.append({"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": va_loss, "val_acc": va_acc})

    saved = ""
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), best_path)
        saved = "  <- model terbaik disimpan"
    print(f"Epoch [{epoch+1:02d}/{CONFIG['epochs']}] {time.time()-t0:5.0f}s | "
          f"Train {tr_loss:.4f}/{tr_acc*100:.2f}% | Val {va_loss:.4f}/{va_acc*100:.2f}%{saved}")

print(f"\nVal acc terbaik: {best_val_acc*100:.2f}%")

# %% Cell 8: Kurva training
import matplotlib.pyplot as plt

ep = [h["epoch"] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ep, [h["train_loss"] for h in history], label="Train")
ax[0].plot(ep, [h["val_loss"] for h in history], label="Val")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(ep, [h["train_acc"] for h in history], label="Train")
ax[1].plot(ep, [h["val_acc"] for h in history], label="Val")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "kurva_training.png"), dpi=150)
plt.show()

# %% Cell 9: Evaluasi FINAL di TEST SET (pakai model terbaik)
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support)

model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            out = model(images.to(device))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

print("=== CLASSIFICATION REPORT (TEST SET) ===")
report_txt = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report_txt)
test_acc = accuracy_score(all_labels, all_preds)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='macro', zero_division=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Macro Precision: {prec_m*100:.2f}% | Macro Recall: {rec_m*100:.2f}% | Macro F1: {f1_m*100:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prediksi Model'); plt.ylabel('Label Asli')
plt.title('Confusion Matrix - Test Set (ViT)')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix_test.png"), dpi=150)
plt.show()

with open(os.path.join(CONFIG["output_dir"], "classification_report_test.txt"), "w") as f:
    f.write(report_txt + f"\nTest Accuracy: {test_acc*100:.2f}%\n")

# %% Cell 10: Ekspor artefak untuk web
with open(os.path.join(CONFIG["output_dir"], "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=2)

model_info = {
    "model_name": CONFIG["model_name"],
    "img_size": CONFIG["img_size"],
    "mean": MEAN,
    "std": STD,
    "num_classes": num_classes,
    "test_accuracy": round(float(test_acc), 4),
    "test_precision_macro": round(float(prec_m), 4),
    "test_recall_macro": round(float(rec_m), 4),
    "test_f1_macro": round(float(f1_m), 4),
    "best_val_accuracy": round(float(best_val_acc), 4),
}
with open(os.path.join(CONFIG["output_dir"], "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2)

print("Artefak tersimpan di:", CONFIG["output_dir"])
print("Salin 3 file ini ke  web/backend/model/ :")
print("  - vit_paddy_best.pth\n  - class_names.json\n  - model_info.json")

# %% Cell 11 (opsional): Tes prediksi 1 gambar dari file
from PIL import Image
from google.colab import files

uploaded = files.upload()
for fn in uploaded.keys():
    img = Image.open(fn).convert('RGB')
    x = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x)[0], dim=0)
    top_p, top_i = torch.topk(prob, 3)
    plt.figure(figsize=(4, 4)); plt.imshow(img); plt.axis('off')
    plt.title(f"{class_names[top_i[0]]} ({top_p[0]*100:.1f}%)"); plt.show()
    for p, i in zip(top_p, top_i):
        print(f"  {class_names[i]:28s} {p*100:.2f}%")

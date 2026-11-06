# -*- coding: utf-8 -*-
"""
Training ViT lokal (Windows + GPU) - Klasifikasi Penyakit Daun Padi, 10 kelas Paddy Doctor.

Versi lokal dari train_vit_padi.py (Colab). Logika sama:
  - split 70/15/15 stratified, seed tetap
  - transform train/val/test dipisah
  - normalisasi mengikuti konfigurasi model timm
  - class weight, cosine scheduler, mixed precision
  - simpan model TERBAIK (val acc), early stopping
  - evaluasi final hanya di TEST SET
  - ekspor vit_paddy_best.pth, class_names.json, model_info.json ke web/backend/model

Jalankan:
    python train_vit_local.py                 # training penuh
    python train_vit_local.py --smoke         # uji cepat (subset kecil, 1 epoch)
    python train_vit_local.py --resume        # lanjut dari checkpoint terakhir
"""
import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "paddy-disease-classification" / "train_images"
OUT_DIR = ROOT / "training_output"
WEB_MODEL_DIR = ROOT / "web" / "backend" / "model"

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--patience", type=int, default=5, help="early stopping (epoch tanpa peningkatan val acc)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="uji cepat: subset kecil, 1 epoch, tidak menimpa web/backend/model")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = OUT_DIR / ("smoke" if args.smoke else "full")
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "train_log.txt", "a", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log(f"Device: {device} | {torch.cuda.get_device_name(0) if use_amp else 'CPU'} | torch {torch.__version__}", fh)
    if args.smoke:
        args.epochs = 1

    # ---------- data ----------
    probe = datasets.ImageFolder(str(DATA_DIR))
    class_names = probe.classes
    num_classes = len(class_names)
    targets = np.array(probe.targets)
    log(f"Total gambar: {len(probe)} | kelas: {num_classes}", fh)
    for i, c in enumerate(class_names):
        log(f"  {i}: {c:28s} {int((targets == i).sum())}", fh)

    idx_all = np.arange(len(targets))
    if args.smoke:  # subset stratified kecil
        idx_all, _ = train_test_split(idx_all, train_size=400, stratify=targets, random_state=args.seed)

    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=0.15, stratify=targets[idx_all], random_state=args.seed)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.15 / 0.85, stratify=targets[idx_trainval], random_state=args.seed)
    log(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}", fh)

    # ---------- model ----------
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    cfg = resolve_data_config({}, model=model)
    mean, std = list(cfg["mean"]), list(cfg["std"])
    log(f"Normalisasi model: mean={mean} std={std}", fh)
    model = model.to(device)

    S = 224
    train_tf = transforms.Compose([
        transforms.Resize((S, S)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([
        transforms.Resize((S, S)), transforms.ToTensor(), transforms.Normalize(mean, std)])

    ds_train = Subset(datasets.ImageFolder(str(DATA_DIR), transform=train_tf), idx_train)
    ds_val = Subset(datasets.ImageFolder(str(DATA_DIR), transform=eval_tf), idx_val)
    ds_test = Subset(datasets.ImageFolder(str(DATA_DIR), transform=eval_tf), idx_test)

    kw = dict(num_workers=args.workers, pin_memory=use_amp, persistent_workers=args.workers > 0)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size * 2, shuffle=False, **kw)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size * 2, shuffle=False, **kw)

    counts = np.bincount(targets[idx_train], minlength=num_classes).astype(np.float32)
    weights = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_path = out_dir / "vit_paddy_best.pth"
    ckpt_path = out_dir / "last_checkpoint.pt"
    history, best_val, start_epoch, bad = [], 0.0, 0, 0

    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        history, best_val, start_epoch, bad = ck["history"], ck["best_val"], ck["epoch"], ck["bad"]
        log(f"Resume dari epoch {start_epoch} (best val {best_val*100:.2f}%)", fh)

    def run_epoch(loader, train):
        model.train(train)
        tot_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model(images)
                    loss = criterion(out, labels)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                tot_loss += loss.item() * images.size(0)
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return tot_loss / total, correct / total

    log("Mulai training...", fh)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, True)
        va_loss, va_acc = run_epoch(val_loader, False)
        scheduler.step()
        history.append(dict(epoch=epoch + 1, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc))

        mark = ""
        if va_acc > best_val:
            best_val, bad = va_acc, 0
            torch.save(model.state_dict(), best_path)
            mark = "  <- terbaik"
        else:
            bad += 1
        log(f"Epoch {epoch+1:02d}/{args.epochs} {time.time()-t0:5.0f}s | train {tr_loss:.4f}/{tr_acc*100:.2f}% | "
            f"val {va_loss:.4f}/{va_acc*100:.2f}%{mark}", fh)

        torch.save(dict(model=model.state_dict(), optim=optimizer.state_dict(), sched=scheduler.state_dict(),
                        scaler=scaler.state_dict(), history=history, best_val=best_val, epoch=epoch + 1, bad=bad),
                   ckpt_path)
        if bad >= args.patience:
            log(f"Early stopping: tidak membaik {args.patience} epoch.", fh)
            break

    log(f"Val acc terbaik: {best_val*100:.2f}%", fh)

    # ---------- kurva ----------
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="Train"); ax[0].plot(ep, [h["val_loss"] for h in history], label="Val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], label="Train"); ax[1].plot(ep, [h["val_acc"] for h in history], label="Val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out_dir / "kurva_training.png", dpi=150); plt.close()
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ---------- evaluasi TEST ----------
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(images.to(device))
            preds.extend(out.argmax(1).cpu().numpy()); labels_all.extend(labels.numpy())

    used = sorted(set(labels_all) | set(preds))
    report = classification_report(labels_all, preds, labels=list(range(num_classes)),
                                   target_names=class_names, digits=4, zero_division=0)
    test_acc = accuracy_score(labels_all, preds)
    log("=== CLASSIFICATION REPORT (TEST SET) ===\n" + report + f"\nTest Accuracy: {test_acc*100:.2f}%", fh)
    (out_dir / "classification_report_test.txt").write_text(report + f"\nTest Accuracy: {test_acc*100:.2f}%\n")

    cm = confusion_matrix(labels_all, preds, labels=list(range(num_classes)))
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prediksi Model"); plt.ylabel("Label Asli"); plt.title("Confusion Matrix - Test Set (ViT)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_test.png", dpi=150); plt.close()

    # ---------- ekspor ----------
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    info = dict(model_name="vit_base_patch16_224", img_size=S, mean=mean, std=std, num_classes=num_classes,
                test_accuracy=round(float(test_acc), 4), best_val_accuracy=round(float(best_val), 4),
                epochs_trained=len(history), seed=args.seed,
                split=dict(train=len(idx_train), val=len(idx_val), test=len(idx_test)))
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2))

    if not args.smoke:
        WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for f in ("vit_paddy_best.pth", "class_names.json", "model_info.json"):
            shutil.copy2(out_dir / f, WEB_MODEL_DIR / f)
        log(f"Artefak disalin ke {WEB_MODEL_DIR}", fh)
    log("SELESAI.", fh)


if __name__ == "__main__":
    main()

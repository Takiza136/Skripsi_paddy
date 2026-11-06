"""Backend FastAPI - Klasifikasi Penyakit Daun Padi (Vision Transformer).

Menjalankan:
    uvicorn main:app --reload --port 8000
lalu buka http://localhost:8000

Endpoint:
    GET  /health    status server & model
    GET  /classes   daftar kelas
    GET  /diseases  info semua penyakit
    POST /predict   upload gambar (multipart field "file") -> hasil prediksi
Frontend (../frontend) disajikan langsung oleh server ini pada "/".
"""
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from diseases import DISEASES, get_disease_info

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "vit_paddy_best.pth"))
CLASS_NAMES_PATH = Path(os.getenv("CLASS_NAMES_PATH", BASE_DIR / "model" / "class_names.json"))
MODEL_INFO_PATH = Path(os.getenv("MODEL_INFO_PATH", BASE_DIR / "model" / "model_info.json"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
# Di bawah ambang ini, hasil ditandai "tidak yakin" (model ViT cenderung overconfident).
LOW_CONF_THRESHOLD = float(os.getenv("LOW_CONF_THRESHOLD", "0.60"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

state = {"model": None, "classes": [], "transform": None, "info": {}, "error": None}


def load_model() -> None:
    """Muat model ViT + metadata. Error disimpan di state agar /health informatif."""
    import timm

    try:
        for p in (MODEL_PATH, CLASS_NAMES_PATH):
            if not p.exists():
                raise FileNotFoundError(f"File tidak ditemukan: {p}")

        classes = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        info = {}
        if MODEL_INFO_PATH.exists():
            info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))

        model_name = info.get("model_name", "vit_base_patch16_224")
        img_size = int(info.get("img_size", 224))
        # Normalisasi HARUS sama dengan saat training (disimpan di model_info.json)
        mean = info.get("mean", [0.5, 0.5, 0.5])
        std = info.get("std", [0.5, 0.5, 0.5])

        model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        state.update(
            model=model,
            classes=classes,
            info=info,
            error=None,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
        )
        print(f"[OK] Model dimuat: {model_name} | {len(classes)} kelas")
    except Exception as exc:  # noqa: BLE001
        state["error"] = str(exc)
        print(f"[WARN] Model belum siap: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="PadiScan AI - ViT Rice Disease API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok" if state["model"] is not None else "model_not_loaded",
        "model_loaded": state["model"] is not None,
        "num_classes": len(state["classes"]),
        "model_info": state["info"],
        "detail": state["error"],
    }


@app.get("/classes")
def classes():
    return {"classes": state["classes"]}


@app.get("/diseases")
def diseases():
    return {"diseases": DISEASES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(503, f"Model belum dimuat. {state['error'] or ''}".strip())

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Format tidak didukung. Gunakan JPG, PNG, atau WebP.")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "File kosong.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Ukuran file melebihi {MAX_UPLOAD_MB} MB.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File bukan gambar yang valid.")

    x = state["transform"](image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(state["model"](x)[0], dim=0)

    k = min(3, len(state["classes"]))
    top_p, top_i = torch.topk(probs, k)
    top = [
        {
            "label": state["classes"][i],
            "name": get_disease_info(state["classes"][i])["name"],
            "confidence": round(float(p), 4),
        }
        for p, i in zip(top_p.tolist(), top_i.tolist())
    ]
    best = top[0]
    return {
        "label": best["label"],
        "confidence": best["confidence"],
        "low_confidence": best["confidence"] < LOW_CONF_THRESHOLD,
        "threshold": LOW_CONF_THRESHOLD,
        "top": top,
        "info": get_disease_info(best["label"]),
    }


# ---- Frontend statis (didaftarkan terakhir agar tidak menimpa route API) ----
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

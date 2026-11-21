"""Layanan AI klasifikasi sampah (YOLO). Cara menjalankan: lihat README.md."""
import io
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

WEIGHTS = "models/best.pt"
CONFIDENCE_THRESHOLD = 0.60
MAX_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}

# Key TIPS dan GROUPS harus sama dengan nama kelas pada model.
TIPS = {
    "organik": [
        "Pisahkan dari sampah anorganik dan jangan dicampur plastik.",
        "Bisa diolah jadi kompos atau pakan maggot.",
        "Kalau tidak diolah, buang di tempat sampah basah.",
    ],
    "plastik": [
        "Kosongkan dan bilas botol atau kemasan.",
        "Lepas tutup dan label.",
        "Masukkan ke tempat sampah daur ulang.",
    ],
    "kertas_kardus": [
        "Jaga tetap kering dan tidak berminyak.",
        "Ratakan atau lipat kardus agar hemat tempat.",
        "Setor ke bank sampah atau pengepul kertas.",
    ],
    "logam": [
        "Bilas sisa isi kaleng.",
        "Pipihkan kaleng bila bisa.",
        "Jual ke pengepul logam atau setor ke bank sampah.",
    ],
    "kaca": [
        "Hati-hati dengan pecahan, bungkus dengan kertas tebal sebelum dibuang.",
        "Bilas botol dan pisahkan dari sampah lain.",
        "Setor ke bank sampah atau pengepul kaca.",
    ],
    "b3": [
        "Jangan dibuang bersama sampah biasa.",
        "Simpan di wadah tertutup dan jauhkan dari anak-anak.",
        "Serahkan ke titik pengumpulan sampah B3 atau fasilitas khusus.",
    ],
    "residu": [
        "Sampah ini sulit atau tidak bisa didaur ulang.",
        "Kurangi pemakaiannya bila memungkinkan.",
        "Buang ke tempat sampah residu atau sampah umum.",
    ],
}
GROUPS = {
    "organik": "organik",
    "plastik": "anorganik",
    "kertas_kardus": "anorganik",
    "logam": "anorganik",
    "kaca": "anorganik",
    "b3": "berbahaya",
    "residu": "lainnya",
}
UNCERTAIN_TIPS = ["Foto kurang jelas, coba lagi. Pastikan cahaya cukup dan objek ada di tengah."]


class TrashClassifier:
    def __init__(self, weights: str = WEIGHTS, threshold: float = CONFIDENCE_THRESHOLD):
        self.model = YOLO(weights)
        self.threshold = threshold
        missing = [n for n in (getattr(self.model, "names", None) or {}).values() if n not in TIPS]
        if missing:
            print(f"PERINGATAN: kelas model {missing} tidak ada di TIPS/GROUPS")

    @staticmethod
    def _load(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        return Image.open(Path(image)).convert("RGB")

    def predict(self, image) -> dict:
        res = self.model.predict(self._load(image), imgsz=224, verbose=False)[0]
        top3 = [
            {"category": res.names[int(i)], "confidence": round(float(c), 4)}
            for i, c in zip(res.probs.top5[:3], res.probs.top5conf[:3])
        ]
        category, conf = top3[0]["category"], top3[0]["confidence"]
        uncertain = conf < self.threshold
        return {
            "category": "tidak_yakin" if uncertain else category,
            "group": None if uncertain else GROUPS.get(category),
            "confidence": conf,
            "is_uncertain": uncertain,
            "tips": UNCERTAIN_TIPS if uncertain else TIPS.get(category, []),
            "top3": top3,
        }


@lru_cache(maxsize=1)
def get_classifier() -> TrashClassifier:
    return TrashClassifier()


app = FastAPI(title="AI Trash Sorting Assistant")

# Sesuaikan dengan port web dev server tim.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:  # chatbot.py opsional
    from chatbot import router as chat_router
    app.include_router(chat_router)
except ImportError:
    pass


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"success": False, "error": {"code": code, "message": message}},
    )


@app.post("/api/classify")
def classify(image: UploadFile = File(...)):
    """Klasifikasi foto sampah.

    Form-data `image` (JPG/PNG, maks 5 MB).
    Berhasil: `{"success": true, "data": {category, group, confidence, tips}}`.
    - category: organik, plastik, kertas_kardus, logam, kaca, b3, residu, atau tidak_yakin
    - group: organik, anorganik, berbahaya, lainnya (null jika tidak_yakin)

    Gagal: `{"success": false, "error": {code, message}}`
    dengan code INVALID_IMAGE (400), FILE_TOO_LARGE (413), atau AI_ERROR (500).
    """
    if image.content_type not in ALLOWED_TYPES:
        return _error(400, "INVALID_IMAGE", "Format gambar tidak didukung")
    data = image.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return _error(413, "FILE_TOO_LARGE", "Ukuran gambar maksimal 5 MB")
    try:
        r = get_classifier().predict(data)
    except UnidentifiedImageError:
        return _error(400, "INVALID_IMAGE", "Format gambar tidak didukung")
    except Exception:
        return _error(500, "AI_ERROR", "Gagal memproses gambar, coba lagi")
    return {
        "success": True,
        "data": {k: r[k] for k in ("category", "group", "confidence", "tips")},
    }

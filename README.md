# AI Trash Sorting Assistant: Layanan AI

## Menjalankan

Prasyarat: Python 3.10 dan file model di `models/best.pt`.

```bash
python -m venv .venv
pip install -r requirements.txt
uvicorn classifier:app --port 8001
```

Cek: buka http://localhost:8001/docs, lalu coba `POST /api/classify` dengan satu foto.
Detail endpoint dan format respons ada di halaman itu.

Agar bisa diakses dari perangkat lain di jaringan yang sama, tambahkan `--host 0.0.0.0`.

# LTX 2.5 Serverless Worker (Native ltx-pipelines)

Worker RunPod Serverless untuk **LTX 2.5 (Lightricks)** menggunakan **native `ltx-pipelines` Python API** (tanpa ComfyUI).

## Fitur
- **T2V + Audio**: Text-to-Video dengan audio native tersinkronisasi
- **I2V + Audio**: Image-to-Video dengan audio native
- **Retake**: Regenerasi segmen video eksisting via `RetakePipeline`

## Arsitektur
```
GitHub Actions → Docker Hub (z0ci/ltx25-serverless:latest) → RunPod Serverless
                                                              ↓ mount
                                          Network Volume /workspace/models/ltx-2.5 (66 GB)
```

- **Engine**: `DistilledPipeline` / `RetakePipeline` (native Python)
- **GPU**: 48 GB (L40S / RTX 6000 Ada, pool `ADA_48_PRO`)
- **Kuantisasi**: `FP8_CAST` + `OffloadMode.CPU` (muat di 48 GB)
- **Region**: EU-RO-1 (selaras volume & S3 existing)

## Deploy

### 1. Build Image (GitHub Actions)
Push ke branch `main`/`master` → workflow `.github/workflows/deploy.yml` otomatis build & push
ke `z0ci/ltx25-serverless:latest`. **JANGAN build Docker di laptop.**

### 2. Siapkan Network Volume + Preload Bobot
Buat Network Volume 100 GB di `EU-RO-1`, mount ke Pod sementara di `/workspace`, lalu:
```bash
export HF_TOKEN="hf_xxx"
python preload_weights.py
```
(LTX-2.5 gated di HuggingFace → wajib HF Read Token.)

### 3. Buat Serverless Endpoint
- Image: `z0ci/ltx25-serverless:latest`
- GPU: `ADA_48_PRO` (L40S / RTX 6000 Ada 48 GB)
- Volume: mount Network Volume ke `/workspace`
- Env: `HF_TOKEN=hf_xxx` (opsional, hanya bila worker perlu unduh tambahan)
- Workers: min 0, max 1, idleTimeout 30s

## Skema Input

```json
{
  "input": {
    "mode": "t2v | i2v | retake",
    "prompt": "deskripsi video",
    "image": "<base64 RAW / URL>",
    "video": "<base64 RAW / URL>",
    "duration": 3.4,
    "fps": 24,
    "width": 1024,
    "height": 576,
    "seed": 42,
    "start_time": 0.0,
    "end_time": 2.0,
    "regenerate_video": true,
    "regenerate_audio": true
  }
}
```

## Output
```json
{
  "video": "<base64 RAW mp4, sudah termasuk audio AAC>",
  "mode": "t2v",
  "num_frames": 81,
  "fps": 24,
  "execution_time": 42.5
}
```

> Output `video` = **base64 mentah tanpa prefix `data:`** — UI wajib normalisasi:
> `out.video.startsWith('data:') ? out.video : 'data:video/mp4;base64,' + out.video`

# LTX 2.5 Serverless Worker (Native ltx-pipelines)

Worker RunPod Serverless untuk **LTX 2.5 (Lightricks)** menggunakan **native `ltx-pipelines` Python API** (tanpa ComfyUI).

## Fitur
- **T2V + Audio**: Text-to-Video dengan audio native tersinkronisasi
- **I2V + Audio**: Image-to-Video dari 1 gambar awal
- **Interpolasi (Start + End Image)**: Morfing/transisi antara gambar awal (frame 0) dan gambar akhir (frame N-1)
- **Multi-Keyframe**: Beberapa gambar di posisi frame bebas (`frame_idx`, `strength`) sebagai anchor
- **Video Panjang (Long Video)**: Hingga 20 detik via overlapping temporal windows (`stream_chunks`)
- **LoRA Kustom**: Injeksi LoRA ComfyUI-compatible (anatomi tubuh, gaya, karakter) dengan strength per-LoRA
- **Retake**: Regenerasi segmen video eksisting (visual dan/atau audio) via `RetakePipeline`
- **Enhance Prompt**: Peningkatan prompt otomatis via Gemma

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

## Live Deployment
- **Endpoint ID**: `5ky9a1hacoybl1`
- **Template ID**: `ha1fp46pv8`
- **Run URL**: `https://api.runpod.ai/v2/5ky9a1hacoybl1/run`
- **Docker Image**: `z0ci/ltx25-serverless:latest` (15.39 GB)
- **GPU Pools**: `AMPERE_48` (A40 @ $1.22/h) + `ADA_48_PRO` (L40S @ $1.75/h)
- **Scaling**: min 0, max 1, idleTimeout 30s (Scale-to-Zero)

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
    "mode": "t2v | i2v | interpolation | keyframes | retake",
    "prompt": "deskripsi video dan audionya",
    "image": "<base64 RAW / URL>",             // Start image (frame 0)
    "image_end": "<base64 RAW / URL>",         // End image (frame N-1, mode interpolation)
    "keyframes": [                             // Optional keyframes timeline
      { "image": "<b64/url>", "frame_idx": 60, "strength": 0.8 }
    ],
    "video": "<base64 RAW / URL>",             // Mode retake saja
    "duration": 3.4,                           // 2.0, 3.4, 5.0, 6.7, 10.0, 15.0, 20.0
    "fps": 24,
    "width": 1024,
    "height": 576,
    "seed": 42,
    "enhance_prompt": false,                   // Prompt enhancement via Gemma
    "loras": [                                 // Injeksi LoRA kustom (anatomi/gaya)
      { "path": "my_lora.safetensors", "strength": 0.8 },
      { "url": "https://.../model.safetensors", "strength": 0.7 }
    ],
    "start_time": 0.0,                         // Mode retake
    "end_time": 2.0,
    "regenerate_video": true,
    "regenerate_audio": true
  }
}
```

## Video Sangat Panjang (Long Video Streaming)
- Durasi `<= 6.7s` (<= 161 frame): Single-pass direct denoise (cepat & hemat).
- Durasi `> 6.7s` (hingga 20s / 481 frame): Otomatis beralih ke **Overlapping Temporal Windows** (`stream_chunks` via `ChunkConfig` jendela 97 frame + carry 25 frame) dengan crossfade blending temporal dan jahitan audio mulus tanpa loncatan.

## Output
```json
{
  "video": "<base64 RAW mp4, sudah termasuk audio AAC>",
  "mode": "i2v",
  "num_frames": 81,
  "fps": 24,
  "width": 1024,
  "height": 576,
  "has_audio": true,
  "chunked": false,
  "loras": [],
  "execution_time": 42.5
}
```

> Output `video` = **base64 mentah tanpa prefix `data:`** — UI wajib normalisasi:
> `out.video.startsWith('data:') ? out.video : 'data:video/mp4;base64,' + out.video`

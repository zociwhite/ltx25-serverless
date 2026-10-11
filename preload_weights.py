"""
Skrip Preload Bobot LTX 2.5 ke Network Volume RunPod (/workspace/models/ltx-2.5)

Jalankan sekali via Pod sementara dengan volume terpasang:
    HF_TOKEN="hf_xxx" python preload_weights.py

File yang diunduh (total ~66.2 GB):
1. diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors (39.1 GB)
2. text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors (24.5 GB)
3. vae/ltx-2.5-video-vae-bf16.safetensors (1.37 GB)
4. vae/ltx-2.5-audio-vae-bf16.safetensors (0.34 GB)
5. latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors (0.93 GB)
6. model_patches/ltx-2.5-duration-head-bf16.safetensors (3.8 MB)
"""
import os
import sys
from huggingface_hub import hf_hub_download

REPO_ID = "Lightricks/LTX-2.5"
TARGET_DIR = os.environ.get("TARGET_DIR", "/workspace/models/ltx-2.5")
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

FILES = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "model_patches/ltx-2.5-duration-head-bf16.safetensors",
]

def main():
    if not TOKEN:
        print("[ERROR] HF_TOKEN wajib diset! (LTX-2.5 repo berstatus gated di HuggingFace)", file=sys.stderr)
        sys.exit(1)

    print(f"=== PRELOAD LTX 2.5 WEIGHTS KE: {TARGET_DIR} ===")
    os.makedirs(TARGET_DIR, exist_ok=True)

    for rel_path in FILES:
        out_path = os.path.join(TARGET_DIR, rel_path)
        if os.path.exists(out_path):
            print(f"[SKIP] Sudah ada: {rel_path} ({os.path.getsize(out_path):,} bytes)")
            continue

        print(f"[DOWNLOAD] Mengunduh {rel_path} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=rel_path,
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False,
            token=TOKEN,
        )
        print(f"[OK] Selesai: {rel_path}")

    print("\n=== SEMUA BOBOT LTX 2.5 BERHASIL DIUNDUH ===")

if __name__ == "__main__":
    main()

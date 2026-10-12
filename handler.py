import base64
import gc
import logging
import os
import re
import tempfile
import time
import requests
import runpod
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from ltx_core.model.video_vae import get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.retake import RetakePipeline
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ltx25-handler")

MODELS_DIR = os.environ.get("LTX25_MODELS_DIR", "/workspace/models/ltx-2.5")
TRANSFORMER_PATH = os.environ.get(
    "TRANSFORMER_PATH",
    os.path.join(MODELS_DIR, "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
)
TEXT_ENCODER_PATH = os.environ.get(
    "TEXT_ENCODER_PATH",
    os.path.join(MODELS_DIR, "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
)
VIDEO_VAE_PATH = os.environ.get(
    "VIDEO_VAE_PATH",
    os.path.join(MODELS_DIR, "vae/ltx-2.5-video-vae-bf16.safetensors"),
)
AUDIO_VAE_PATH = os.environ.get(
    "AUDIO_VAE_PATH",
    os.path.join(MODELS_DIR, "vae/ltx-2.5-audio-vae-bf16.safetensors"),
)
SPATIAL_UPSAMPLER_PATH = os.environ.get(
    "SPATIAL_UPSAMPLER_PATH",
    os.path.join(
        MODELS_DIR,
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    ),
)
DURATION_HEAD_PATH = os.environ.get(
    "DURATION_HEAD_PATH",
    os.path.join(MODELS_DIR, "model_patches/ltx-2.5-duration-head-bf16.safetensors"),
)
if not os.path.exists(DURATION_HEAD_PATH):
    DURATION_HEAD_PATH = None

_distilled_pipeline = None
_retake_pipeline = None


def _get_model_paths() -> ModelPaths:
    return ModelPaths.from_split(
        transformer_path=TRANSFORMER_PATH,
        text_encoder_path=TEXT_ENCODER_PATH,
        video_vae_path=VIDEO_VAE_PATH,
        audio_vae_path=AUDIO_VAE_PATH,
        duration_head_path=DURATION_HEAD_PATH,
    )


def _load_distilled_pipeline() -> DistilledPipeline:
    global _distilled_pipeline
    if _distilled_pipeline is None:
        logger.info("Loading LTX 2.5 DistilledPipeline (FP8_CAST, CPU offload)...")
        model_paths = _get_model_paths()
        quant_policy = QuantizationKind.FP8_CAST.to_policy(TRANSFORMER_PATH)

        _distilled_pipeline = DistilledPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
            loras=(),
            quantization=quant_policy,
            offload_mode=OffloadMode.CPU,
        )
        logger.info("DistilledPipeline ready.")
    return _distilled_pipeline


def _load_retake_pipeline() -> RetakePipeline:
    global _retake_pipeline
    if _retake_pipeline is None:
        logger.info("Loading LTX 2.5 RetakePipeline (FP8_CAST, CPU offload)...")
        model_paths = _get_model_paths()
        quant_policy = QuantizationKind.FP8_CAST.to_policy(TRANSFORMER_PATH)

        _retake_pipeline = RetakePipeline(
            model_paths=model_paths,
            loras=(),
            quantization=quant_policy,
            distilled=True,
            offload_mode=OffloadMode.CPU,
        )
        logger.info("RetakePipeline ready.")
    return _retake_pipeline


def _download_or_decode(src: str, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if re.match(r"^https?://", src):
        resp = requests.get(src, timeout=180)
        resp.raise_for_status()
        tmp.write(resp.content)
    else:
        raw_b64 = src.split(",", 1)[1] if src.startswith("data:") else src
        tmp.write(base64.b64decode(raw_b64))
    tmp.close()
    return tmp.name


def _file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _snap_frames(target: int) -> int:
    # LTX-2.5 causal grid: 8k + 1
    # Choices: 49 (≈2s @24fps), 81 (≈3.4s), 121 (≈5s), 161 (≈6.7s)
    k = max(1, round((target - 1) / 8))
    return int(8 * k + 1)


def _snap_resolution(value: int, divisor: int) -> int:
    # Two-stage pipelines (Distilled) require multiples of 64; one-stage (Retake) 32.
    snapped = max(divisor, round(value / divisor) * divisor)
    if snapped != value:
        logger.warning("Resolution %d snapped to %d (must be a multiple of %d).", value, snapped, divisor)
    return int(snapped)


@torch.inference_mode()
def handler(job: dict) -> dict:
    job_input = job.get("input") or {}
    mode = (job_input.get("mode") or "t2v").lower().strip()
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt and mode != "retake":
        raise ValueError("'prompt' is required")

    seed = int(job_input.get("seed", 42))
    raw_w = int(job_input.get("width", 1024))
    raw_h = int(job_input.get("height", 576))
    divisor = 64 if mode in ("t2v", "i2v") else 32
    width = _snap_resolution(raw_w, divisor)
    height = _snap_resolution(raw_h, divisor)
    fps = int(job_input.get("fps", 24))

    duration_sec = float(job_input.get("duration", 3.4))
    raw_frames = int(duration_sec * fps)
    num_frames = _snap_frames(raw_frames)

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    tmp_files = [out_file]
    t0 = time.time()

    try:
        if mode in ("t2v", "i2v"):
            pipeline = _load_distilled_pipeline()
            images_input = []

            img_src = job_input.get("image") or job_input.get("image_base64") or job_input.get("image_url")
            if img_src:
                img_path = _download_or_decode(img_src, ".png")
                tmp_files.append(img_path)
                images_input.append(ImageConditioningInput(path=img_path, frame_idx=0, strength=1.0))

            output = pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                frame_rate=float(fps),
                images=images_input,
                num_frames=num_frames,
            )

            chunks_num = get_video_chunks_number(output.num_frames, output.tiling_config)
            encode_video(
                video=output.video,
                fps=fps,
                audio=output.audio,
                output_path=out_file,
                video_chunks_number=chunks_num,
            )

        elif mode == "retake":
            vid_src = job_input.get("video") or job_input.get("video_base64") or job_input.get("video_url")
            if not vid_src:
                raise ValueError("Mode 'retake' requires 'video', 'video_base64', or 'video_url'")

            vid_path = _download_or_decode(vid_src, ".mp4")
            tmp_files.append(vid_path)

            src_meta = get_videostream_metadata(vid_path)
            actual_fps = int(round(src_meta.fps))

            start_time = float(job_input.get("start_time", 0.0))
            end_time = float(job_input.get("end_time", duration_sec))
            regen_video = bool(job_input.get("regenerate_video", True))
            regen_audio = bool(job_input.get("regenerate_audio", True))

            pipeline = _load_retake_pipeline()
            output = pipeline(
                video_path=vid_path,
                prompt=prompt,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
                regenerate_video=regen_video,
                regenerate_audio=regen_audio,
            )

            chunks_num = get_video_chunks_number(output.num_frames, output.tiling_config)
            encode_video(
                video=output.video,
                fps=actual_fps,
                audio=output.audio,
                output_path=out_file,
                video_chunks_number=chunks_num,
            )
            fps = actual_fps
            num_frames = output.num_frames
        else:
            raise ValueError(f"Unknown mode '{mode}'. Supported: 't2v', 'i2v', 'retake'")

        exec_time = time.time() - t0
        video_b64 = _file_to_base64(out_file)

        return {
            "video": video_b64,
            "mode": mode,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "has_audio": True,
            "execution_time": round(exec_time, 2),
        }

    finally:
        for p in tmp_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

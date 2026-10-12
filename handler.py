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

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import get_video_chunks_number
from ltx_pipelines.chunks.export import pipeline_output_from_chunks
from ltx_pipelines.chunks.layout import ChunkConfig
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.retake import RetakePipeline
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ltx25-handler")

MODELS_DIR = os.environ.get("LTX25_MODELS_DIR", "/workspace/models/ltx-2.5")
LORAS_DIR = os.path.join(MODELS_DIR, "loras")
os.makedirs(LORAS_DIR, exist_ok=True)
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
_distilled_pipeline_loras_key = None
_retake_pipeline_loras_key = None
LORA_WEIGHT_MAX_BYTES = 800 * 1024 * 1024  # 800 MB guard per LoRA file


def _get_model_paths() -> ModelPaths:
    return ModelPaths.from_split(
        transformer_path=TRANSFORMER_PATH,
        text_encoder_path=TEXT_ENCODER_PATH,
        video_vae_path=VIDEO_VAE_PATH,
        audio_vae_path=AUDIO_VAE_PATH,
        duration_head_path=DURATION_HEAD_PATH,
    )


def _resolve_lora_paths(loras_spec: list) -> list[str]:
    """Resolve each LoRA entry ({path|url, strength}) into a local safetensors file.

    Local entries are matched against LORAS_DIR by basename. Remote entries are
    downloaded once into LORAS_DIR and reused afterwards.
    """
    resolved = []
    for entry in loras_spec or []:
        if isinstance(entry, str):
            entry = {"path": entry, "strength": 1.0}
        src = entry.get("path") or entry.get("url") or entry.get("name")
        if not src:
            continue
        if re.match(r"^https?://", src):
            name = os.path.basename(src.split("?", 1)[0]) or "lora.safetensors"
            dst = os.path.join(LORAS_DIR, name)
            if not os.path.exists(dst):
                logger.info("Downloading LoRA: %s", src)
                resp = requests.get(src, timeout=600, stream=True)
                resp.raise_for_status()
                written = 0
                with open(dst, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        written += len(chunk)
                        if written > LORA_WEIGHT_MAX_BYTES:
                            fh.close()
                            os.remove(dst)
                            raise ValueError(f"LoRA file exceeds {LORA_WEIGHT_MAX_BYTES} bytes: {name}")
                        fh.write(chunk)
            src = dst
        else:
            candidate = src if os.path.isabs(src) else os.path.join(LORAS_DIR, os.path.basename(src))
            if not os.path.exists(candidate):
                raise ValueError(f"LoRA not found: {src} (looked in {LORAS_DIR})")
            src = candidate
        resolved.append((src, float(entry.get("strength", 1.0))))
    return resolved


def _build_lora_ops(resolved: list[tuple[str, float]]) -> list[LoraPathStrengthAndSDOps]:
    return [
        LoraPathStrengthAndSDOps(path=p, strength=s, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
        for p, s in resolved
    ]


def _load_distilled_pipeline(loras_key: str = "", loras_spec: list | None = None) -> DistilledPipeline:
    global _distilled_pipeline, _distilled_pipeline_loras_key
    if _distilled_pipeline is None or _distilled_pipeline_loras_key != loras_key:
        logger.info("Loading LTX 2.5 DistilledPipeline (FP8_CAST, CPU offload) loras=%s...", loras_key or "none")
        model_paths = _get_model_paths()
        quant_policy = QuantizationKind.FP8_CAST.to_policy(TRANSFORMER_PATH)
        loras = _build_lora_ops(_resolve_lora_paths(loras_spec or [])) if loras_spec else []

        _distilled_pipeline = DistilledPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
            loras=loras,
            quantization=quant_policy,
            offload_mode=OffloadMode.CPU,
        )
        _distilled_pipeline_loras_key = loras_key
        logger.info("DistilledPipeline ready.")
    return _distilled_pipeline


def _load_retake_pipeline(loras_key: str = "", loras_spec: list | None = None) -> RetakePipeline:
    global _retake_pipeline, _retake_pipeline_loras_key
    if _retake_pipeline is None or _retake_pipeline_loras_key != loras_key:
        logger.info("Loading LTX 2.5 RetakePipeline (FP8_CAST, CPU offload) loras=%s...", loras_key or "none")
        model_paths = _get_model_paths()
        quant_policy = QuantizationKind.FP8_CAST.to_policy(TRANSFORMER_PATH)
        loras = _build_lora_ops(_resolve_lora_paths(loras_spec or [])) if loras_spec else []

        _retake_pipeline = RetakePipeline(
            model_paths=model_paths,
            loras=loras,
            quantization=quant_policy,
            distilled=True,
            offload_mode=OffloadMode.CPU,
        )
        _retake_pipeline_loras_key = loras_key
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
    # Choices: 49 (≈2s @24fps), 81 (≈3.4s), 121 (≈5s), 161 (≈6.7s), 241 (≈10s), 481 (≈20s)
    k = max(1, round((target - 1) / 8))
    return int(8 * k + 1)


def _snap_resolution(value: int, divisor: int) -> int:
    # Two-stage pipelines (Distilled) require multiples of 64; one-stage (Retake) 32.
    snapped = max(divisor, round(value / divisor) * divisor)
    if snapped != value:
        logger.warning("Resolution %d snapped to %d (must be a multiple of %d).", value, snapped, divisor)
    return int(snapped)


def _collect_images(job_input: dict, num_frames: int, tmp_files: list) -> list[ImageConditioningInput]:
    """Build the image conditioning list from start/end images and arbitrary keyframes.

    Sources accepted:
      - image / image_base64 / image_url           -> start image at frame 0
      - image_end / image_end_base64 / image_end_url -> end image at frame num_frames - 1
      - keyframes: [ {image|url|path, frame_idx, strength} ]
    frame_idx must satisfy 0 <= frame_idx < num_frames (LTX rejects negatives).
    """
    images_input: list[ImageConditioningInput] = []

    def _add(src, frame_idx, strength):
        idx = int(frame_idx)
        if idx < 0 or idx >= num_frames:
            raise ValueError(f"frame_idx={idx} is outside [0, {num_frames})")
        p = _download_or_decode(src, ".png")
        tmp_files.append(p)
        images_input.append(ImageConditioningInput(path=p, frame_idx=idx, strength=float(strength)))

    start_src = (
        job_input.get("start_image")
        or job_input.get("image")
        or job_input.get("image_base64")
        or job_input.get("image_url")
    )
    if start_src:
        _add(start_src, 0, float(job_input.get("start_strength", 1.0)))

    end_src = (
        job_input.get("end_image")
        or job_input.get("image_end")
        or job_input.get("image_end_base64")
        or job_input.get("image_end_url")
    )
    if end_src:
        _add(end_src, num_frames - 1, float(job_input.get("end_strength", job_input.get("image_end_strength", 1.0))))

    for kf in job_input.get("keyframes") or []:
        src = kf.get("image") or kf.get("image_base64") or kf.get("image_url") or kf.get("url") or kf.get("path")
        if not src:
            continue
        _add(src, kf.get("frame_idx", 0), kf.get("strength", 1.0))

    # De-duplicate identical (frame_idx) start/end collisions by keeping the first.
    seen = set()
    deduped = []
    for im in images_input:
        if im.frame_idx in seen:
            continue
        seen.add(im.frame_idx)
        deduped.append(im)
    return deduped


@torch.inference_mode()
def handler(job: dict) -> dict:
    job_input = job.get("input") or {}
    mode = (job_input.get("mode") or "t2v").lower().strip()
    # Normalize aliases for the distilled-pipeline family.
    if mode in ("i2v_transition", "interpolation", "morph", "transition"):
        mode = "interpolation"
    if mode in ("keyframe", "multi_keyframe"):
        mode = "keyframes"
    distilled_modes = ("t2v", "i2v", "interpolation", "keyframes")
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt and mode != "retake":
        raise ValueError("'prompt' is required")

    seed = int(job_input.get("seed", 42))
    raw_w = int(job_input.get("width", 1024))
    raw_h = int(job_input.get("height", 576))
    divisor = 64 if mode in distilled_modes else 32
    width = _snap_resolution(raw_w, divisor)
    height = _snap_resolution(raw_h, divisor)
    fps = int(job_input.get("fps", 24))

    duration_sec = float(job_input.get("duration", 3.4))
    raw_frames = int(duration_sec * fps)
    num_frames = _snap_frames(raw_frames)

    loras_spec = job_input.get("loras") or []
    loras_key = "|".join(
        sorted(f"{e.get('path') or e.get('url') or e.get('name')}:{e.get('strength', 1.0)}"
               for e in loras_spec if isinstance(e, dict))
    ) if loras_spec else ""

    # Long clips (> ~6.7s / 161 frames) use overlapping temporal windows.
    use_chunked = num_frames > 161 and mode in distilled_modes

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    tmp_files = [out_file]
    t0 = time.time()

    try:
        if mode in distilled_modes:
            pipeline = _load_distilled_pipeline(loras_key, loras_spec)
            images_input = _collect_images(job_input, num_frames, tmp_files)

            call_kwargs = dict(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                frame_rate=float(fps),
                images=images_input,
                num_frames=num_frames,
                enhance_prompt=bool(job_input.get("enhance_prompt", False)),
            )

            if use_chunked:
                logger.info("Chunked long-video mode: %d frames (>161) via stream_chunks.", num_frames)
                chunk_cfg = ChunkConfig(
                    chunk_pixel_frames=int(job_input.get("chunk_pixel_frames", 97)),
                    next_video_carry_frames=int(job_input.get("chunk_carry_frames", 25)),
                )
                chunks_iter, stitched_frames, tiling_cfg = pipeline.stream_chunks(
                    chunk_config=chunk_cfg, **call_kwargs
                )
                output = pipeline_output_from_chunks(
                    chunks_iter, num_frames=stitched_frames, tiling_config=tiling_cfg
                )
                num_frames = stitched_frames
            else:
                output = pipeline(**call_kwargs)

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

            pipeline = _load_retake_pipeline(loras_key, loras_spec)
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
            raise ValueError(
                f"Unknown mode '{mode}'. Supported: 't2v', 'i2v', 'interpolation', 'keyframes', 'retake'"
            )

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
            "chunked": use_chunked,
            "loras": [e.get("path") or e.get("url") or e.get("name") for e in loras_spec] if loras_spec else [],
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

"""
Runtime compatibility shims for LTX 2.5 UNCENSORED checkpoints (ChrisColeTech split format).

The ChrisColeTech / ibyteohdear uncensored checkpoints differ from the official
Lightricks release in three ways that break native `ltx-pipelines` loading:

  1. Connector linear weights are quantized to float8_e4m3fn, while the connector
     module runs in bfloat16 -> "self and mat2 must have the same dtype".
  2. Safetensors keys lack the `model.diffusion_model.` prefix that
     LTXV_MODEL_COMFY_RENAMING_MAP and EMBEDDINGS_PROCESSOR_KEY_OPS expect,
     so every transformer block / connector key is dropped during load.
  3. The block-streaming key scan then reports
     "Block weights under prefix 'transformer_blocks.' do not match the 48 model blocks".

Applying this module makes the uncensored checkpoints load exactly like the
official ones. Import it BEFORE constructing any pipeline.
"""
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("ltx25-compat")


def apply() -> None:
    """Install all compatibility patches. Idempotent."""
    _patch_linear_dtype()
    _patch_transformer_prefix()
    _patch_embeddings_prefix()
    _patch_block_key_scan()
    _patch_prompt_enhancer()


def _patch_linear_dtype() -> None:
    """Upcast linear weights on the fly when their dtype differs from the input.

    ChrisColeTech quantizes connector weights to float8_e4m3fn but the connector
    runs in bfloat16, which raises a dtype-mismatch error deep inside F.linear.
    """
    if getattr(F, "_ltx25_compat_linear", False):
        return
    original = F.linear

    def _safe_linear(input, weight, bias=None):
        if input.dtype != weight.dtype:
            weight = weight.to(input.dtype)
        if bias is not None and bias.dtype != input.dtype:
            bias = bias.to(input.dtype)
        return original(input, weight, bias)

    _safe_linear._ltx25_compat_linear = True
    F.linear = _safe_linear
    torch.nn.functional.linear = _safe_linear
    logger.info("compat: F.linear auto-upcast enabled.")


def _patch_transformer_prefix() -> None:
    """Accept transformer keys with OR without the `model.diffusion_model.` prefix.

    The stock map only matches `model.diffusion_model.` and silently drops every
    other key. An empty `with_matching()` matches any key, so the replacement is
    applied when present and the key passes through untouched when absent.
    """
    try:
        from ltx_core.loader.sd_ops import SDOps
        from ltx_core.model.transformer import model_configurator as tc_mc
        import ltx_core.model.transformer as tc_tr

        extended = (
            SDOps("LTXV_MODEL_COMFY_PREFIX_MAP_EXTENDED")
            .with_matching()
            .with_replacement("model.diffusion_model.", "")
        )
        tc_mc.LTXV_MODEL_COMFY_RENAMING_MAP = extended
        tc_tr.LTXV_MODEL_COMFY_RENAMING_MAP = extended
        logger.info("compat: transformer prefix map extended.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("compat: transformer prefix patch skipped (%s).", exc)


def _patch_embeddings_prefix() -> None:
    """Accept connector keys without the `model.diffusion_model.` prefix."""
    try:
        from ltx_core.text_encoders.gemma.encoders import encoder_configurator as ec
        import ltx_core.text_encoders.gemma as gemma_pkg

        extended = (
            ec.EMBEDDINGS_PROCESSOR_KEY_OPS
            .with_matching(prefix="video_embeddings_connector.")
            .with_replacement("video_embeddings_connector.", "video_connector.")
            .with_matching(prefix="audio_embeddings_connector.")
            .with_replacement("audio_embeddings_connector.", "audio_connector.")
        )
        ec.EMBEDDINGS_PROCESSOR_KEY_OPS = extended
        if hasattr(gemma_pkg, "EMBEDDINGS_PROCESSOR_KEY_OPS"):
            gemma_pkg.EMBEDDINGS_PROCESSOR_KEY_OPS = extended
        logger.info("compat: embeddings prefix map extended.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("compat: embeddings prefix patch skipped (%s).", exc)


def _patch_block_key_scan() -> None:
    """Fallback for the block-streaming key scan.

    If the sd_ops-mapped scan yields no blocks, rebuild the block map straight
    from the raw checkpoint keys. Guards against the
    "Block weights under prefix 'transformer_blocks.' do not match" failure.
    """
    try:
        import safetensors
        from ltx_core.block_streaming import builder as bsb

        if getattr(bsb._scan_checkpoint_keys, "_ltx25_compat_scan", False):
            return
        original = bsb._scan_checkpoint_keys

        def _safe_scan(checkpoint_paths, sd_ops, blocks_prefix):
            block_key_map, non_block_keys = original(checkpoint_paths, sd_ops, blocks_prefix)
            if not block_key_map:
                prefix_dot = blocks_prefix + "."
                rebuilt: dict = {}
                for path in checkpoint_paths:
                    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
                        for sft_key in handle.keys():
                            if not sft_key.startswith(prefix_dot):
                                continue
                            rest = sft_key[len(prefix_dot):]
                            idx_str, _, param_name = rest.partition(".")
                            try:
                                idx = int(idx_str)
                            except ValueError:
                                continue
                            rebuilt.setdefault(idx, []).append((sft_key, param_name))
                if rebuilt:
                    logger.info("compat: rebuilt %d blocks from raw keys.", len(rebuilt))
                    return rebuilt, non_block_keys
            return block_key_map, non_block_keys

        _safe_scan._ltx25_compat_scan = True
        bsb._scan_checkpoint_keys = _safe_scan
        logger.info("compat: block key scan fallback enabled.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("compat: block scan patch skipped (%s).", exc)


def _patch_prompt_enhancer() -> None:
    """Silently ignore `enhance_first_prompt=True` when the encoder model_type
    is `gemma4_unified` (which lacks generative instruct weights).

    Prevents: ValueError: Prompt enhancement with encode root model_type='gemma4_unified'
    requires --prompt-enhancer-gemma-root pointing at a generative instruct checkpoint.
    """
    try:
        from ltx_pipelines.utils import blocks

        if getattr(blocks.PromptEncoder, "_ltx25_compat_enhancer", False):
            return
        orig_call = blocks.PromptEncoder.__call__

        def _safe_call(self, prompts, *, enhance_first_prompt=False, **kwargs):
            if enhance_first_prompt:
                model_type = getattr(self, "_encode_model_type", "")
                enhancer_root = getattr(self, "_prompt_enhancer_gemma_root", None)
                if model_type != "gemma3" and not enhancer_root:
                    logger.info(
                        "compat: disabling enhance_first_prompt for %s (no separate enhancer root).",
                        model_type,
                    )
                    enhance_first_prompt = False
            return orig_call(self, prompts, enhance_first_prompt=enhance_first_prompt, **kwargs)

        _safe_call._ltx25_compat_enhancer = True
        blocks.PromptEncoder.__call__ = _safe_call
        logger.info("compat: prompt enhancer guard enabled.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("compat: prompt enhancer patch skipped (%s).", exc)


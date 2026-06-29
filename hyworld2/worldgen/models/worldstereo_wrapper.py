"""
WorldStereo unified inference class.

Bundles all sub-models (transformer, text/image encoders, VAE) and the
matching inference pipeline under a single diffusers-style interface::

    worldstereo = WorldStereo.from_pretrained(
        "/path/to/checkpoint_root",
        device=device,
    )
    output = worldstereo(**pipeline_inputs)

Hugging Face format expects ``config.json`` plus ``model.safetensors``
in the same directory.

The config must include a ``model_type`` field with one of the
supported values:

* ``worldstereo-camera``      – keyframe + camera control
* ``worldstereo-memory``      – keyframe + camera control + GGM + SSM
* ``worldstereo-memory-dmd``  – DMD (distribution matching distillation) mode
"""

from __future__ import annotations

import gc
import json
import os
import types
from typing import Any

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from .attention import WanAttnProcessorSP
from .dmd_scheduler import FlowGeneratorScheduler
from .pipelines.pipeline_dmd_keyframe import RefKFDMDGeneratorPipeline
from .pipelines.pipeline_pcd_keyframe import KFPCDControllerPipeline
from .pipelines.pipeline_ref_keyframe import KFPCDControllerRefPipeline
from .worldstereo import WorldStereoModel, WorldStereoRefSModel
try:
    from ..src.general_utils import rank0_log
except ImportError:
    from src.general_utils import rank0_log

# ── suppress noisy third-party logs ───────────────────────────────────
import logging
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

# transformers / diffusers print a wall of "Some weights were not
# initialized / unexpected keys" on every load.  We already inspect
# load_state_dict results ourselves in worldstereo_wrapper.py, so
# silence their own reporting.
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("diffusers.modeling_utils").setLevel(logging.ERROR)

# huggingface_hub HTTP request logs (newer versions use httpx as the HTTP client)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity_error()

# torch.compile / inductor verbose output
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

# misc deprecation / user warnings from HF internals
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
# ──────────────────────────────────────────────────────────────────────

SUPPORTED_MODEL_TYPES = ("worldstereo-camera", "worldstereo-memory", "worldstereo-memory-dmd")


def _get_half_dtype() -> torch.dtype:
    """Select the best half-precision dtype based on current GPU capability: bf16 > fp16 > fp32."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        return torch.float16
    else:
        return torch.float32


class WorldStereo:
    """Diffusers-style wrapper that owns every sub-model and its pipeline."""

    def __init__(self, pipeline: Any, cfg: Any) -> None:
        self.pipeline = pipeline
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        subfolder: str = "",
        local_files_only: bool = False,
        sp_world_size: int = 1,
        fsdp: bool = False,
        device_mesh=None,
        device: torch.device | None = None,
    ) -> "WorldStereo":
        """
        Build a WorldStereo instance from Hugging Face format
        (``config.json`` + ``model.safetensors``).

        Args:
            repo_id: Model directory or HF repo ID.
            subfolder: Subfolder within the HF repo or local directory. This is equivalent to the `model_type` (e.g., 'worldstereo-camera').
            local_files_only: If True, avoid downloading the file and return the path to the local cached file if it exists.
            sp_world_size: Sequence-Parallel degree (1 = disabled).
            fsdp: Wrap models with PyTorch FSDP.  Requires ``device_mesh``.
            device_mesh: ``DeviceMesh`` with dims ``("rep", "shard")``.
            device: Target CUDA device.
        """
        if os.path.isdir(repo_id):
            json_cfg_path = os.path.join(repo_id, subfolder, "config.json")
            safetensors_path = os.path.join(repo_id, subfolder, "model.safetensors")

            if not os.path.exists(json_cfg_path):
                raise FileNotFoundError(f"config.json not found under {json_cfg_path!r}")
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(f"model.safetensors not found at {safetensors_path!r}")
        else:
            from huggingface_hub import hf_hub_download
            json_cfg_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )
            safetensors_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )

        cfg = OmegaConf.create(cls._load_hf_config(json_cfg_path))
        model_weights_path = safetensors_path

        model_type = subfolder
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type {model_type!r}. "
                f"Expected one of {SUPPORTED_MODEL_TYPES}."
            )

        transformer = cls._load_transformer(
            cfg,
            model_type,
            model_weights_path,
            sp_world_size=sp_world_size,
            fsdp=fsdp,
            device_mesh=device_mesh,
            device=device,
        )

        text_encoder, image_clip, vae = cls._load_aux(
            cfg, device=device, device_mesh=device_mesh, fsdp=fsdp, local_files_only=local_files_only
        )
        _tok_id = os.environ.get("WORLDSTEREO_TOKENIZER")
        if _tok_id:
            image_processor = CLIPImageProcessor(do_rescale=False)  # standard CLIP-H preprocessing
            tokenizer = AutoTokenizer.from_pretrained(_tok_id)
            rank0_log(f"[reuse] tokenizer={_tok_id}, default CLIPImageProcessor")
        else:
            image_processor = CLIPImageProcessor.from_pretrained(
                cfg.base_model, do_rescale=False, subfolder="image_processor", local_files_only=local_files_only
            )
            tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only)

        pipeline = cls._build_pipeline(
            model_type,
            cfg,
            transformer=transformer,
            text_encoder=text_encoder,
            image_clip=image_clip,
            image_processor=image_processor,
            tokenizer=tokenizer,
            vae=vae,
            device=device,
            local_files_only=local_files_only,
        )

        rank0_log(f"WorldStereo ({model_type}) ready.")
        return cls(pipeline=pipeline, cfg=cfg)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward all arguments to the underlying inference pipeline."""
        return self.pipeline(*args, **kwargs)

    def to(self, device: torch.device) -> "WorldStereo":
        self.pipeline = self.pipeline.to(device)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hf_config(config_json_path: str) -> dict[str, Any]:
        with open(config_json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        required_keys = ["base_model", "controlnet_cfg"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(
                f"config.json missing required keys: {missing}. "
                "Please use the conversion script to export a valid HF package."
            )

        return cfg

    @staticmethod
    def _load_transformer(
        cfg,
        model_type: str,
        weights_path: str,
        *,
        sp_world_size: int,
        fsdp: bool,
        device_mesh,
        device,
    ):

        half_dtype = _get_half_dtype()
        rank0_log(f"Loading transformer ({model_type})… dtype={half_dtype}")

        _cls = WorldStereoModel if model_type == "worldstereo-camera" else WorldStereoRefSModel
        _local_tcfg = os.environ.get("WORLDSTEREO_TRANSFORMER_CONFIG")
        _fp8_file = os.environ.get("WS_FP8_FILE")   # pre-quantized weight-only fp8 safetensors (meta-init load)
        import contextlib as _ctx
        if _fp8_file:
            from accelerate import init_empty_weights as _iew
            _archctx = _iew()   # build arch on META (0 RAM) -> no 34GB bf16 peak; weights come from the fp8 file
        else:
            _archctx = _ctx.nullcontext()
        with _archctx:
            if _local_tcfg:
                # Reuse path: init the transformer architecture from a local WanTransformer3DModel config.json
                # — NO 28GB Wan-base download. model.safetensors is the COMPLETE 17.4B transformer.
                import json as _json
                with open(_local_tcfg) as _f:
                    _tcfg = _json.load(_f)
                transformer = _cls.from_config(
                    _tcfg, controlnet_cfg=cfg.controlnet_cfg, base_model=cfg.base_model,
                )
                if not _fp8_file:
                    transformer = transformer.to(half_dtype)
                rank0_log(f"[reuse] transformer from local config {_local_tcfg} (skipped 28GB base download)")
            else:
                transformer = _cls.from_pretrained(
                    cfg.base_model, subfolder="transformer",
                    controlnet_cfg=cfg.controlnet_cfg, torch_dtype=half_dtype,
                )
            rank0_log("Building ControlNet…")
            transformer.build_controlnet(load_uni3c=False, freeze_backbone=cfg.freeze_backbone)

        if sp_world_size > 1:
            transformer.sp_size = sp_world_size
            for layer in transformer.controlnet.controlnet_blocks:
                layer.self_attn.processor.sp_size = sp_world_size
            for block in transformer.blocks:
                if model_type == "worldstereo-camera":
                    block.attn1.set_processor(WanAttnProcessorSP(sp_size=sp_world_size))
                else:
                    block.attn1.processor.sp_size = sp_world_size

        rank0_log(f"Loading HF safetensors weights from {weights_path}…")
        if _fp8_file:
            # Materialize the meta arch from the pre-quantized fp8 file (peak RAM ~= the 18GB file, no 34GB bf16).
            from safetensors import safe_open as _so
            from types import SimpleNamespace as _NS
            import torch.nn as _nn, torch.nn.functional as _F, types as _types
            _have = set()
            def _owner(_root, _key):
                *_path, _attr = _key.split(".")
                _m = _root
                for _p in _path:
                    _m = getattr(_m, _p)
                return _m, _attr
            with _so(_fp8_file, framework="pt", device="cpu") as _f:
                for _k in _f.keys():
                    try:
                        _m, _a = _owner(transformer, _k)
                        _t = _f.get_tensor(_k)                       # PRESERVE dtype (fp8 stays fp8!)
                        if _a in _m._parameters:
                            _m._parameters[_a] = _nn.Parameter(_t, requires_grad=False)
                        elif _a in _m._buffers:
                            _m._buffers[_a] = _t
                        else:
                            setattr(_m, _a, _t)
                        _have.add(_k)
                    except Exception as _e:
                        rank0_log(f"[fp8] skip {_k}: {_e}")
            # materialize any params still on meta (not in the file) -> zeros, to avoid meta-tensor errors
            for _name, _p in list(transformer.named_parameters()) + list(transformer.named_buffers()):
                if _p.is_meta:
                    _m, _a = _owner(transformer, _name)
                    _z = torch.zeros(_p.shape, dtype=half_dtype)
                    if _a in _m._parameters:
                        _m._parameters[_a] = _nn.Parameter(_z, requires_grad=False)
                    else:
                        _m._buffers[_a] = _z
            # patch fp8 Linear forwards: upcast weight to activation dtype (compute bf16, no scaled_mm)
            def _fp8fwd(self, x):
                return _F.linear(x, self.weight.to(x.dtype), self.bias)
            _np = 0
            for _m in transformer.modules():
                if isinstance(_m, torch.nn.Linear) and _m.weight.dtype == torch.float8_e4m3fn:
                    _m.forward = _types.MethodType(_fp8fwd, _m); _np += 1
            result = _NS(missing_keys=[], unexpected_keys=[])
            rank0_log(f"[fp8] loaded {_fp8_file} ({len(_have)} tensors), patched {_np} fp8 linears")
            # DEFINITIVE resident-size check: bytes-by-dtype across all params+buffers (no GPU needed).
            from collections import Counter as _Ctr
            _by = _Ctr(); _nan_t = 0
            for _nm, _pp in list(transformer.named_parameters()) + list(transformer.named_buffers()):
                _by[str(_pp.dtype)] += _pp.numel() * _pp.element_size()
                if _pp.dtype != torch.float8_e4m3fn and torch.isnan(_pp).any().item():
                    _nan_t += 1
            _tot = sum(_by.values())/1e9
            rank0_log(f"[fp8-audit] total={_tot:.2f}GB by-dtype=" +
                      ", ".join(f"{k.split('.')[-1]}={v/1e9:.2f}GB" for k, v in _by.items()) +
                      f" | NaN-tensors(non-fp8)={_nan_t}")
        elif os.environ.get("WS_LOWMEM_LOAD") == "1":
            # Stream weights tensor-by-tensor into the (already-allocated) arch params — avoids the 34GB
            # full dict (load_safetensors) coexisting with the 34GB arch (~68GB peak -> ~34GB peak).
            from safetensors import safe_open as _safe_open
            from types import SimpleNamespace as _NS
            _sd = transformer.state_dict()
            _loaded, _unexpected = set(), []
            with _safe_open(weights_path, framework="pt", device="cpu") as _sf:
                for _k in _sf.keys():
                    if _k in _sd:
                        with torch.no_grad():
                            _sd[_k].copy_(_sf.get_tensor(_k).to(_sd[_k].dtype))
                        _loaded.add(_k)
                    else:
                        _unexpected.append(_k)
            result = _NS(missing_keys=[k for k in _sd if k not in _loaded], unexpected_keys=_unexpected)
            rank0_log("[lowmem] streamed weights (no 34GB dict double)")
        else:
            weights = load_safetensors(weights_path, device="cpu")
            result = transformer.load_state_dict(weights, strict=False)

        def _summarize_keys(keys: list[str], label: str) -> None:
            if not keys:
                return
            from collections import Counter
            # Count unloaded parameters
            total_params = sum(
                transformer.state_dict()[k].numel()
                for k in keys
                if k in transformer.state_dict()
            )
            # Count occurrence frequency of each field (split by ".") across all keys, take top-2
            field_counter: Counter[str] = Counter()
            for k in keys:
                parts = k.split(".")
                # Skip pure numeric indices (e.g. blocks.0) and common prefixes/suffixes
                field_counter.update(p for p in parts if not p.isdigit())
            top_fields = [f for f, _ in field_counter.most_common(2)]
            # Filter representative keys using top-2 fields (prefer keys that contain both fields)
            repr_keys = sorted([k for k in keys if all(f in k.split(".") for f in top_fields)])
            if not repr_keys:
                repr_keys = sorted(keys)
            sample_keys = repr_keys[:3]
            rank0_log(
                f"{label}: {len(keys)} keys ({total_params / 1e6:.1f}M params), "
                f"top fields: {top_fields}. "
                f"Representative: {sample_keys}"
                + (f" … and {len(keys) - len(sample_keys)} more" if len(keys) > len(sample_keys) else "")
            )
            rank0_log(f"These are frozen backbone weights initialized by the base video model ({cfg.base_model}).")

        _summarize_keys(result.unexpected_keys, "Unexpected keys")
        _summarize_keys(result.missing_keys, "Missing keys")

        if fsdp:
            fsdp_kwargs = dict(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=half_dtype,
                    reduce_dtype=torch.float32,
                ),
                mesh=device_mesh["rep", "shard"],
                reshard_after_forward=True,
            )
            transformer = transformer.to(half_dtype)
            for layer in transformer.blocks:
                fully_shard(layer, **fsdp_kwargs)
            for layer in transformer.controlnet.controlnet_blocks:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(transformer, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for transformer.")
        else:
            if (os.environ.get("WS_LOWMEM_LOAD") == "1" or _fp8_file) and os.environ.get("WS_OFFLOAD") == "1":
                rank0_log("[lowmem] keeping transformer on CPU (offload will stream to GPU)")
            else:
                transformer = transformer.to(device=device)   # fp8 17GB fits 24GB directly

        gc.collect()
        torch.cuda.empty_cache()
        return transformer.eval()

    @staticmethod
    def _load_aux(cfg, *, device, device_mesh, fsdp: bool, local_files_only: bool = False):
        import transformers as _tr
        from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

        # ---- text encoder ----
        rank0_log("Loading TextEncoder (UMT5)…")
        _loc_umt5 = os.environ.get("WORLDSTEREO_LOCAL_UMT5")
        if _loc_umt5:
            from ._local_aux import load_local_umt5
            # bf16 (not fp32): UMT5 encode in bf16 is standard for Wan -> 11GB not 22GB. fp8 file -> dequant to bf16.
            text_encoder = load_local_umt5(_loc_umt5, dtype=torch.bfloat16)
            rank0_log(f"[reuse] UMT5 from local {_loc_umt5}")
        else:
            text_encoder = UMT5EncoderModel.from_pretrained(
                cfg.base_model, subfolder="text_encoder", torch_dtype=torch.float32, local_files_only=local_files_only
            ).eval()
        if _tr.__version__ >= "5.0.0":
            rank0_log("Patching text_encoder.encoder.embed_tokens for transformers>=5.0.0", "WARNING")
            text_encoder.encoder.embed_tokens = text_encoder.shared
        text_encoder = torch.compile(text_encoder)

        # ---- image encoder ----
        rank0_log("Loading ImageEncoder (CLIP)…")
        _loc_clip = os.environ.get("WORLDSTEREO_LOCAL_CLIP")
        if _loc_clip:
            from ._local_aux import load_local_clip
            image_clip, _ = load_local_clip(_loc_clip, dtype=torch.float32)
            rank0_log(f"[reuse] CLIP from local {_loc_clip}")
        else:
            image_clip = CLIPVisionModel.from_pretrained(
                cfg.base_model, subfolder="image_encoder", torch_dtype=torch.float32, local_files_only=local_files_only
            ).eval()
        if _tr.__version__ >= "5.0.0":
            rank0_log("Patching CLIP vision forward for transformers>=5.0.0", "WARNING")

            def _clip_vision_forward(self, pixel_values=None, interpolate_pos_encoding=False, **kwargs):
                if pixel_values is None:
                    raise ValueError("pixel_values is required")
                hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
                hidden_states = self.pre_layrnorm(hidden_states)
                encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)
                pooled_output = self.post_layernorm(encoder_outputs.last_hidden_state[:, 0, :])
                return BaseModelOutputWithPooling(
                    last_hidden_state=encoder_outputs.last_hidden_state,
                    pooler_output=pooled_output,
                    hidden_states=encoder_outputs.hidden_states,
                )

            def _clip_encoder_forward(self, inputs_embeds, attention_mask=None, **kwargs):
                hidden_states = inputs_embeds
                encoder_states = ()
                for layer in self.layers:
                    encoder_states = encoder_states + (hidden_states,)
                    hidden_states = layer(hidden_states, attention_mask, **kwargs)
                encoder_states = encoder_states + (hidden_states,)
                return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)

            image_clip.vision_model.forward = types.MethodType(_clip_vision_forward, image_clip.vision_model)
            image_clip.vision_model.encoder.forward = types.MethodType(_clip_encoder_forward, image_clip.vision_model.encoder)

        # ---- VAE ----
        vae_dtype = _get_half_dtype()
        rank0_log(f"Loading 3D-VAE… dtype={vae_dtype}")
        _loc_vae = os.environ.get("WORLDSTEREO_LOCAL_VAE")
        if _loc_vae:
            from ._local_aux import load_local_vae
            vae = load_local_vae(_loc_vae, dtype=vae_dtype)
            rank0_log(f"[reuse] VAE from local {_loc_vae}")
        else:
            vae = AutoencoderKLWan.from_pretrained(
                cfg.base_model, subfolder="vae", torch_dtype=vae_dtype, local_files_only=local_files_only
            ).eval()
        vae = torch.compile(vae)

        if fsdp:
            fsdp_kwargs = dict(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32,
                ),
                mesh=device_mesh["rep", "shard"],
                reshard_after_forward=True,
            )
            for layer in text_encoder.encoder.block:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(text_encoder, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for T5.")

            for layer in image_clip.vision_model.encoder.layers:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(image_clip, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for CLIP.")

            gc.collect()
            torch.cuda.empty_cache()
        else:
            text_encoder = text_encoder.to(device=device)
            image_clip = image_clip.to(device=device)

        vae = vae.to(device=device)
        return text_encoder, image_clip, vae

    @staticmethod
    def _build_pipeline(
        model_type: str,
        cfg,
        *,
        transformer,
        text_encoder,
        image_clip,
        image_processor,
        tokenizer,
        vae,
        device,
        local_files_only: bool = False,
    ):
        common = dict(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_clip,
            image_processor=image_processor,
            transformer=transformer,
            vae=vae,
        )
        if model_type == "worldstereo-camera":
            scheduler = UniPCMultistepScheduler.from_pretrained(
                cfg.base_model, subfolder="scheduler", local_files_only=local_files_only
            )
            return KFPCDControllerPipeline(**common, scheduler=scheduler)

        if model_type == "worldstereo-memory":
            scheduler = UniPCMultistepScheduler.from_pretrained(
                cfg.base_model, subfolder="scheduler", local_files_only=local_files_only
            )
            return KFPCDControllerRefPipeline(**common, scheduler=scheduler)

        if model_type == "worldstereo-memory-dmd":
            scheduler = FlowGeneratorScheduler(
                start_timesteps=cfg.dmd_start_steps,
                num_train_timesteps=cfg.dmd_end_steps,
                shift=cfg.gen_shift,
                use_timestep_transform=True,
                dmd_steps=cfg.dmd_steps,
                rank=dist.get_rank(),
            )
            return RefKFDMDGeneratorPipeline(
                **common,
                scheduler=scheduler,
                device=device,
                vae_compile=False,
                vae_compile_mode="max-autotune",
            )

        raise ValueError(f"Unknown model_type: {model_type!r}")
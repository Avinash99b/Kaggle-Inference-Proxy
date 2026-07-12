"""
run_model_with_proxy.py
================================================================================
Private Kaggle GGUF inference worker.

Run this in a Kaggle notebook (ideally with the "GPU T4 x2" accelerator
enabled). It:

  1. Installs missing Python/system dependencies automatically.
  2. Installs/builds a CUDA-enabled llama.cpp Python backend (llama-cpp-python).
  3. Connects OUTBOUND to the proxy (WebSocket first, long-poll fallback).
     Kaggle never needs an inbound port -- this script only makes outgoing
     connections.
  4. Waits for jobs, downloads/caches the requested GGUF model from Hugging
     Face on demand, loads it (splitting across both T4s when it helps),
     runs generation, and reports the result back to the proxy.
  5. Manages its on-disk model cache: before every download it checks how
     much free space is available and, if there isn't enough room, evicts
     the least-recently-used cached model(s) first. A background monitor
     also periodically checks free space so the cache stays healthy even
     between downloads. The cache lives outside /kaggle/working (which has
     a small, quota-limited size on Kaggle) so there's much more headroom.
  6. Is VRAM-aware, purely for INFERENCE (this script never trains or
     fine-tunes anything -- it is solely an inference client node that
     loads pre-trained GGUF weights and runs forward-pass generation, so
     the only VRAM consumers it ever accounts for are the model weights
     themselves and the KV-cache; there's no optimizer state, no gradients,
     and no activation memory for backprop to budget for the way you would
     for training). Only one model is ever kept loaded at a time --
     switching models automatically unloads whatever's currently resident
     first, so a new model can claim that freed VRAM. By default (config
     "gpu.n_ctx": "max"), every model is loaded with the LARGEST context
     window it was itself trained with (read straight from its GGUF
     header), and that is only reduced as far as necessary to fit
     currently-free VRAM (with a configurable safety headroom, default 5%),
     using the model's own GGUF header to estimate KV-cache cost per token.
     If a load still hits a CUDA OOM (estimates are inherently approximate),
     it automatically retries with a smaller context window a few times
     before giving up. An explicit integer context size can still be set
     per-model (or globally) in config to override the "always maximize"
     default.
  7. Cleanly tells the proxy it's going away on Ctrl+C / shutdown, so the
     proxy stops waiting on it immediately instead of leaving in-flight
     HTTP requests hanging until they individually time out.

Just run:
    python run_model_with_proxy.py
or paste the contents into a Kaggle notebook cell and run it.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import hmac
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Set

# --------------------------------------------------------------------------- #
# Step 0: self-install Python dependencies before importing them
# --------------------------------------------------------------------------- #


def _pip_install(*packages: str, extra_index: Optional[str] = None, env: Optional[dict] = None) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages]
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


def ensure_base_dependencies() -> None:
    needed = {
        "websockets": "websockets",
        "requests": "requests",
        "huggingface_hub": "huggingface_hub",
    }
    for module_name, pip_name in needed.items():
        try:
            __import__(module_name)
        except ImportError:
            print(f"[worker] Installing missing dependency: {pip_name}")
            _pip_install(pip_name)


ensure_base_dependencies()

import requests  # noqa: E402
import websockets  # noqa: E402
from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata  # noqa: E402

# --------------------------------------------------------------------------- #
# Step 1: config load / generate
# --------------------------------------------------------------------------- #

CONFIG_PATH = os.environ.get("WORKER_CONFIG_PATH", "inference_server_config.json")

DEFAULT_CONFIG = {
    "proxy_url": "http://localhost:8000",
    "worker_id": "kaggle-worker-1",
    "worker_shared_secret": "change-me-worker-secret",
    "huggingface_token": "",
    # NOTE: deliberately NOT under /kaggle/working -- that path is backed by
    # a small, quota-limited "output" volume on Kaggle (often ~20GB) meant
    # for notebook artifacts you commit, not a multi-GB model cache. /tmp
    # sits on the much larger underlying instance disk.
    "model_cache_dir": "/tmp/gguf_models",
    "models": {
        "example-model": {
            "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
            "file": "llama-2-7b-chat.Q4_K_M.gguf"
            # Optional per-model overrides (fall back to the global "gpu"
            # section below if omitted): "n_ctx", "n_gpu_layers", "n_batch",
            # "n_threads". e.g. a long-context model could set a bigger
            # "n_ctx" here without raising it globally for every model.
        }
    },
    "default_generation": {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 512
    },
    "transport": {
        "prefer_websocket": True,
        "ws_path": "/worker/ws",
        "heartbeat_interval_seconds": 15,
        "reconnect_initial_backoff_seconds": 2,
        "reconnect_max_backoff_seconds": 60,
        "ws_failure_threshold_before_poll_fallback": 3,
        "poll_interval_seconds": 2,
        "long_poll_wait_seconds": 40
    },
    "download": {
        "max_retries": 4,
        "retry_backoff_seconds": 5
    },
    "disk_management": {
        "enabled": True,
        # Extra headroom (beyond the file we're about to download) that we
        # always try to keep free. Eviction targets required_bytes + this.
        "safety_margin_gb": 2,
        # Floor used by the background monitor loop: if free space drops
        # below this at any time (not just during a download), evict LRU
        # cached models until we're back above it (or nothing is left to
        # evict).
        "min_free_space_gb": 5,
        "monitor_interval_seconds": 30
    },
    "vram_management": {
        # Fraction of TOTAL VRAM (across the GPU(s) this worker uses) that
        # is always kept free as headroom, e.g. 0.05 = never plan to use
        # the last 5%.
        "headroom_fraction": 0.05,
        # Never size a context window below this many tokens, even under
        # heavy VRAM pressure.
        "min_n_ctx": 256,
        # On a CUDA OOM while loading, retry with n_ctx multiplied by this
        # factor (repeatedly, down to min_n_ctx) before giving up.
        "oom_shrink_factor": 0.75,
        "max_oom_retries": 4
    },
    "backend_build": {
        "prefer_prebuilt_wheel": True,
        "cmake_cuda_args": "-DGGML_CUDA=on"
    },
    "gpu": {
        "attempt_dual_gpu_split": True,
        # "max" (the default) means: always try to load every model at the
        # largest context window IT supports -- read from its own GGUF
        # header's trained context length -- and only shrink from there as
        # far as needed to fit available VRAM. This is an inference-only
        # setting (KV-cache sizing), not a training batch/sequence length.
        # Set an explicit integer here (or per-model under "models") if you
        # want a fixed cap instead of always-maximize.
        "n_ctx": "max",
        "n_batch": 512,
        "n_threads": 0,
        "n_gpu_layers": -1
    },
    "logging": {
        "level": "INFO",
        "file": "worker.log"
    }
}


def load_or_create_config(path: str, defaults: dict) -> dict:
    abs_path = os.path.abspath(path)
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(defaults, f, indent=2)
        print(f"[worker] No config found at '{abs_path}'. A default config "
              f"(with only the built-in 'example-model' and gpu.n_ctx='max') has "
              f"been created there. If you meant to use a custom config -- your "
              f"own models/n_ctx/etc -- make sure it's actually saved at that "
              f"exact path (or set WORKER_CONFIG_PATH to point at it) BEFORE "
              f"the next run, otherwise this freshly-created default is what "
              f"will keep loading.")
        return json.loads(json.dumps(defaults))
    with open(path, "r") as f:
        cfg = json.load(f)
    merged = json.loads(json.dumps(defaults))
    merged.update(cfg)
    for k, v in defaults.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merged[k] = {**v, **cfg[k]}
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
    print(f"[worker] Loaded EXISTING config from '{abs_path}' (last modified {mtime}). "
          f"Note: an existing config file is never auto-overwritten with new script "
          f"defaults -- if this file predates a script update (e.g. it still has an "
          f"explicit integer 'gpu.n_ctx' instead of \"max\"), delete/replace it and "
          f"rerun to pick up new defaults, or edit it directly.")
    return merged


CONFIG = load_or_create_config(CONFIG_PATH, DEFAULT_CONFIG)

logging.basicConfig(
    level=getattr(logging, CONFIG["logging"].get("level", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(CONFIG["logging"].get("file", "worker.log"))],
)
logger = logging.getLogger("worker")

os.makedirs(CONFIG["model_cache_dir"], exist_ok=True)

# --------------------------------------------------------------------------- #
# Step 2: GPU detection
# --------------------------------------------------------------------------- #


def detect_gpus() -> int:
    """Best-effort GPU count detection without hard-depending on torch."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            lines = [l for l in out.stdout.splitlines() if l.strip().startswith("GPU")]
            if lines:
                return len(lines)
    except Exception:
        pass
    try:
        import torch  # noqa
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


GPU_COUNT = detect_gpus()
logger.info("Detected %d GPU(s).", GPU_COUNT)


def query_gpu_memory() -> List[dict]:
    """Per-GPU memory stats via nvidia-smi, in MB:
    [{'index': 0, 'total_mb': ..., 'used_mb': ..., 'free_mb': ...}, ...]
    Returns [] if nvidia-smi isn't available (e.g. CPU-only worker)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
        stats = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            idx, total, used, free = parts
            stats.append({
                "index": int(idx),
                "total_mb": int(total),
                "used_mb": int(used),
                "free_mb": int(free),
            })
        return stats
    except Exception as e:
        logger.debug("nvidia-smi memory query failed: %s", e)
        return []


class VramManager:
    """Best-effort VRAM accounting used to decide how large a context window
    a model can safely be loaded with, and to keep a safety headroom free so
    llama.cpp's own transient allocations (or anything else briefly sharing
    the GPU) don't tip things into an OOM."""

    def __init__(self, config: dict):
        self.config = config

    def relevant_gpu_indices(self) -> List[int]:
        gpu_cfg = self.config.get("gpu", {})
        if GPU_COUNT >= 2 and gpu_cfg.get("attempt_dual_gpu_split", True):
            return list(range(GPU_COUNT))
        elif GPU_COUNT >= 1:
            return [0]
        return []

    def totals(self) -> Dict[str, int]:
        """{'total_bytes', 'free_bytes', 'used_bytes'} summed across the
        GPU(s) this worker is configured to use. All zero if there's no
        usable GPU / nvidia-smi is unavailable."""
        indices = set(self.relevant_gpu_indices())
        if not indices:
            return {"total_bytes": 0, "free_bytes": 0, "used_bytes": 0}
        stats = query_gpu_memory()
        total = used = free = 0
        for s in stats:
            if s["index"] in indices:
                total += s["total_mb"] * (1024 ** 2)
                used += s["used_mb"] * (1024 ** 2)
                free += s["free_mb"] * (1024 ** 2)
        return {"total_bytes": total, "free_bytes": free, "used_bytes": used}

    def headroom_bytes(self) -> int:
        frac = self.config.get("vram_management", {}).get("headroom_fraction", 0.05)
        return int(self.totals()["total_bytes"] * frac)

    def usable_free_bytes(self) -> int:
        """Currently-free VRAM minus the configured safety headroom."""
        totals = self.totals()
        return max(totals["free_bytes"] - self.headroom_bytes(), 0)


# --------------------------------------------------------------------------- #
# Step 3: install/build the CUDA-enabled llama.cpp backend (llama-cpp-python)
# --------------------------------------------------------------------------- #


def ensure_llama_cpp_backend():
    """Install llama-cpp-python. Try a prebuilt CUDA wheel first (fast,
    no compiler needed); if that fails, build from source with CUDA via
    CMAKE_ARGS; if THAT fails, fall back to a CPU-only build so the worker
    can still serve requests (slower, but functional)."""
    try:
        import llama_cpp  # noqa
        logger.info("llama-cpp-python already importable.")
        return
    except ImportError:
        pass

    cfg = CONFIG["backend_build"]

    if GPU_COUNT > 0 and cfg.get("prefer_prebuilt_wheel", True):
        try:
            logger.info("Attempting prebuilt CUDA wheel for llama-cpp-python...")
            _pip_install(
                "llama-cpp-python",
                extra_index="https://abetlen.github.io/llama-cpp-python/whl/cu121",
            )
            import llama_cpp  # noqa
            logger.info("Installed prebuilt CUDA llama-cpp-python wheel.")
            return
        except Exception as e:
            logger.warning("Prebuilt CUDA wheel install failed (%s); will try source build.", e)

    if GPU_COUNT > 0:
        try:
            logger.info("Building llama-cpp-python from source with CUDA (%s)...",
                        cfg.get("cmake_cuda_args"))
            _pip_install(
                "llama-cpp-python",
                env={"CMAKE_ARGS": cfg.get("cmake_cuda_args", "-DGGML_CUDA=on"),
                     "FORCE_CMAKE": "1"},
            )
            import llama_cpp  # noqa
            logger.info("Built CUDA-enabled llama-cpp-python from source.")
            return
        except Exception as e:
            logger.warning("CUDA source build failed (%s); falling back to CPU build.", e)

    logger.info("Installing CPU-only llama-cpp-python.")
    _pip_install("llama-cpp-python")
    import llama_cpp  # noqa


ensure_llama_cpp_backend()
from llama_cpp import Llama  # noqa: E402

# --------------------------------------------------------------------------- #
# Step 4: disk cache manager (LRU eviction to keep the model cache within
# available disk space)
# --------------------------------------------------------------------------- #


class DiskCacheManager:
    """Tracks which downloaded GGUF files live in the cache directory and
    when each was last used, so we can evict the least-recently-used
    model(s) whenever we're about to run low on disk space -- instead of
    downloads failing outright with 'not enough free disk space'.

    State is persisted to a small JSON file inside the cache directory so
    LRU ordering survives a notebook restart.
    """

    def __init__(self, cache_dir: str, config: dict):
        self.cache_dir = cache_dir
        self.config = config
        self.metadata_path = os.path.join(cache_dir, "cache_metadata.json")
        self._lock = threading.Lock()
        self.entries: Dict[str, dict] = self._load_metadata()
        self._reconcile_with_disk()

    # -- persistence ----------------------------------------------------- #

    def _load_metadata(self) -> Dict[str, dict]:
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception:
                logger.warning("Could not read cache metadata at %s, starting fresh.",
                                self.metadata_path)
        return {}

    def _save_metadata(self) -> None:
        try:
            tmp_path = self.metadata_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(self.entries, f, indent=2)
            os.replace(tmp_path, self.metadata_path)
        except Exception as e:
            logger.warning("Could not persist cache metadata: %s", e)

    def _reconcile_with_disk(self) -> None:
        """Drop metadata entries whose backing file no longer exists (e.g.
        someone manually cleared the cache dir), so eviction decisions are
        never made against stale/missing entries."""
        with self._lock:
            missing = [k for k, v in self.entries.items() if not os.path.exists(v.get("path", ""))]
            for k in missing:
                del self.entries[k]
            if missing:
                self._save_metadata()

    # -- disk stats -------------------------------------------------------- #

    @staticmethod
    def free_space_bytes(path: str) -> int:
        usage = shutil.disk_usage(path)
        return usage.free

    # -- bookkeeping --------------------------------------------------------- #

    def touch(self, cache_key: str, local_path: str, size_bytes: int) -> None:
        """Record/refresh that `cache_key` was just used, so it's the least
        likely candidate for eviction right now."""
        with self._lock:
            self.entries[cache_key] = {
                "path": local_path,
                "size_bytes": size_bytes,
                "last_used": time.time(),
            }
            self._save_metadata()

    def remove(self, cache_key: str) -> None:
        with self._lock:
            entry = self.entries.pop(cache_key, None)
            self._save_metadata()
        if entry is None:
            return
        path = entry["path"]
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info("Evicted cached model '%s' (%s, %.2f GB) to free disk space.",
                            cache_key, path, entry.get("size_bytes", 0) / (1024 ** 3))
            # Clean up the now-possibly-empty per-repo directory.
            parent = os.path.dirname(path)
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except Exception as e:
            logger.warning("Failed to remove cached file for '%s' (%s): %s", cache_key, path, e)

    def ensure_space_for(self, required_bytes: int, protect_paths: Optional[Set[str]] = None) -> bool:
        """Evict least-recently-used cached models (oldest 'last_used'
        first) until at least `required_bytes` plus the configured safety
        margin is free, skipping anything in `protect_paths` (e.g. whatever
        model is currently loaded in memory -- deleting that out from under
        an active model would be bad).

        Returns True if the target was met (or exceeded); False if there
        genuinely isn't enough disk space even after evicting everything
        evictable.
        """
        protect_paths = protect_paths or set()
        disk_cfg = self.config.get("disk_management", {})
        margin_bytes = int(disk_cfg.get("safety_margin_gb", 2) * (1024 ** 3))
        target = required_bytes + margin_bytes

        free = self.free_space_bytes(self.cache_dir)
        if free >= target:
            return True

        logger.info("Low disk space: %.2f GB free, need %.2f GB (including safety margin). "
                    "Evicting least-recently-used cached models...",
                    free / (1024 ** 3), target / (1024 ** 3))

        with self._lock:
            candidates = sorted(
                (k for k, v in self.entries.items() if v.get("path") not in protect_paths),
                key=lambda k: self.entries[k]["last_used"],
            )

        for cache_key in candidates:
            if free >= target:
                break
            self.remove(cache_key)
            free = self.free_space_bytes(self.cache_dir)

        if free < target:
            logger.warning(
                "Only %.2f GB free after evicting every evictable cached model "
                "(%.2f GB was requested including margin). Proceeding only if the "
                "raw file itself still fits.", free / (1024 ** 3), target / (1024 ** 3))
            return free >= required_bytes
        return True


# --------------------------------------------------------------------------- #
# Step 5: GGUF header introspection (for VRAM/KV-cache estimation)
# --------------------------------------------------------------------------- #


def probe_gguf_hyperparams(model_path: str) -> Optional[dict]:
    """Loads just the GGUF header + vocab (n_gpu_layers=0, vocab_only=True,
    minimal n_ctx) to read architecture hyperparameters needed for a
    KV-cache VRAM estimate, without putting any weights on the GPU. Cheap
    and CPU-only. Returns None if the probe or metadata parsing fails --
    callers must be able to fall back to a coarser estimate."""
    try:
        probe = Llama(model_path=model_path, vocab_only=True, n_ctx=8,
                       n_gpu_layers=0, verbose=False)
    except Exception as e:
        logger.warning("Could not probe GGUF header for %s: %s", model_path, e)
        return None
    try:
        meta = getattr(probe, "metadata", {}) or {}

        def meta_int(*keys):
            for k in keys:
                v = meta.get(k)
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        continue
            return None

        arch = meta.get("general.architecture", "")
        n_layer = meta_int(f"{arch}.block_count")
        n_embd = meta_int(f"{arch}.embedding_length")
        n_head = meta_int(f"{arch}.attention.head_count")
        n_head_kv = meta_int(f"{arch}.attention.head_count_kv") or n_head
        n_ctx_train = meta_int(f"{arch}.context_length")

        if not n_ctx_train:
            try:
                n_ctx_train = probe.n_ctx_train()
            except Exception:
                pass

        if not n_layer or not n_embd or not n_head:
            logger.info("GGUF header for %s didn't expose full hyperparameters "
                        "(arch=%r); will use a coarser VRAM estimate.", model_path, arch)
            return None

        return {
            "architecture": arch,
            "n_layer": n_layer,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_head_kv": n_head_kv,
            "n_ctx_train": n_ctx_train,
        }
    except Exception as e:
        logger.warning("Failed to parse GGUF header metadata for %s: %s", model_path, e)
        return None
    finally:
        del probe
        gc.collect()


def estimate_kv_bytes_per_token(hyper: dict, kv_dtype_bytes: int = 2) -> int:
    """Bytes of KV-cache VRAM consumed per token of context, summed across
    all layers (K + V combined). Uses head_count_kv (not head_count) so
    grouped-query-attention models (Llama-3, Mistral, Qwen2, etc.) aren't
    overestimated. kv_dtype_bytes=2 assumes llama.cpp's default f16 KV
    cache (a conservative-but-reasonable default; quantized KV would use
    less)."""
    n_layer = hyper["n_layer"]
    n_embd = hyper["n_embd"]
    n_head = hyper["n_head"]
    n_head_kv = hyper.get("n_head_kv") or n_head
    head_dim = n_embd / n_head
    kv_embd = head_dim * n_head_kv  # per-layer K (or V) dimension
    return int(n_layer * kv_embd * 2 * kv_dtype_bytes)  # *2 for K and V


def resolve_desired_n_ctx(configured: Any, hyper: Optional[dict],
                           fallback_max: int = 32768) -> int:
    """Turns the configured 'gpu.n_ctx' / per-model 'n_ctx' value into a
    concrete integer context target. This is INFERENCE context sizing only
    (how many tokens of KV-cache to allocate for generation) -- it has
    nothing to do with training sequence length or batch size.

    A configured value of "max" (the default) means: always aim for the
    largest context window this specific model was itself trained with,
    read straight from its own GGUF header (n_ctx_train) via `hyper`, so
    every model gets to use as much context as it actually supports rather
    than a fixed guess. If the header didn't expose a trained context
    length (probe failed / unusual GGUF) and no explicit override was
    configured, falls back to `fallback_max` as a generous default -- the
    caller still sizes down further to fit VRAM. An explicit integer in
    config is always honored as-is instead."""
    if isinstance(configured, str) and configured.strip().lower() == "max":
        if hyper and hyper.get("n_ctx_train"):
            return int(hyper["n_ctx_train"])
        logger.info("Could not read the model's trained max context length from its "
                    "GGUF header; falling back to n_ctx=%d as the target before VRAM sizing.",
                    fallback_max)
        return fallback_max
    return int(configured)


def compute_dynamic_n_ctx(model_path: str, desired_n_ctx: int, n_gpu_layers: int,
                           n_batch: int, vram: VramManager, config: dict,
                           hyper: Optional[dict] = None) -> int:
    """Figures out the largest context size (up to `desired_n_ctx`, and
    capped at the model's own trained max context if known) that should fit
    in currently-free VRAM while preserving the configured headroom
    fraction. This budgets INFERENCE-time VRAM only -- model weights (or
    the fraction of them offloaded to GPU) plus KV-cache -- never gradients
    or optimizer state, since this worker only ever runs forward-pass
    generation. Falls back to `desired_n_ctx` unchanged if there's no GPU in
    play (nothing to size against).

    `hyper` can be passed in if the GGUF header was already probed by the
    caller (e.g. to resolve "max" via resolve_desired_n_ctx), to avoid
    probing the same file twice; otherwise it's probed here."""
    if not vram.relevant_gpu_indices():
        return desired_n_ctx

    vram_cfg = config.get("vram_management", {})
    min_ctx = vram_cfg.get("min_n_ctx", 256)

    if hyper is None:
        hyper = probe_gguf_hyperparams(model_path)
    usable_free = vram.usable_free_bytes()
    file_size = os.path.getsize(model_path)

    if hyper:
        n_layer = hyper["n_layer"]
        gpu_layers_used = n_layer if n_gpu_layers < 0 else min(n_gpu_layers, n_layer)
        weight_bytes = int(file_size * (gpu_layers_used / n_layer)) if n_layer else file_size
        kv_per_token = estimate_kv_bytes_per_token(hyper)
        n_ctx_cap = hyper.get("n_ctx_train") or desired_n_ctx
    else:
        # No header info -- assume all weights land on GPU and fall back to
        # a coarse, deliberately generous per-token KV estimate so we don't
        # undershoot and OOM anyway.
        weight_bytes = file_size
        kv_per_token = 128 * 1024
        n_ctx_cap = desired_n_ctx

    compute_overhead = max(512 * 1024 ** 2, n_batch * 1024 * 1024)
    available_for_kv = usable_free - weight_bytes - compute_overhead

    if available_for_kv <= 0:
        max_ctx_by_vram = 0
    else:
        max_ctx_by_vram = int(available_for_kv // max(kv_per_token, 1))

    final_ctx = min(desired_n_ctx, n_ctx_cap or desired_n_ctx)
    if max_ctx_by_vram > 0:
        final_ctx = min(final_ctx, max_ctx_by_vram)
    else:
        final_ctx = min(final_ctx, min_ctx)
    final_ctx = max(final_ctx, min_ctx)

    if final_ctx < desired_n_ctx:
        logger.info(
            "Sizing context for VRAM headroom: requested n_ctx=%d, using n_ctx=%d "
            "(usable free VRAM=%.2f GB, est. weight VRAM=%.2f GB, est. KV/token=%d bytes).",
            desired_n_ctx, final_ctx, usable_free / (1024 ** 3),
            weight_bytes / (1024 ** 3), kv_per_token)
    return final_ctx


# --------------------------------------------------------------------------- #
# Step 6: model manager (download/cache/load/unload, VRAM-aware)
# --------------------------------------------------------------------------- #


class ModelManager:
    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = config["model_cache_dir"]
        self.current_alias: Optional[str] = None
        self.current_local_path: Optional[str] = None
        self.llm: Optional[Llama] = None
        self.disk_cache = DiskCacheManager(self.cache_dir, config)
        self.vram = VramManager(config)
        self._load_lock = threading.Lock()  # serialize concurrent ensure_loaded calls

    def registry(self) -> Dict[str, Any]:
        return self.config.get("models", {})

    def _protected_paths(self) -> Set[str]:
        """Files that must never be evicted right now because they back the
        model currently loaded in memory."""
        return {self.current_local_path} if self.current_local_path else set()

    def _remote_file_size(self, repo: str, filename: str, token: Optional[str]) -> Optional[int]:
        try:
            url = hf_hub_url(repo_id=repo, filename=filename)
            meta = get_hf_file_metadata(url, token=token)
            return getattr(meta, "size", None)
        except Exception as e:
            logger.warning("Could not fetch remote size for %s/%s: %s", repo, filename, e)
            return None

    def resolve_local_path(self, alias: str) -> str:
        entry = self.registry().get(alias)
        if entry is None:
            raise ValueError(f"Model alias '{alias}' is not present in the model registry. "
                              f"Add it to inference_server_config.json under 'models'.")
        # Local file path support -- not subject to disk-cache eviction,
        # since we didn't download it and don't own its lifecycle.
        if "path" in entry:
            path = entry["path"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Configured local model path does not exist: {path}")
            return path

        # Hugging Face repo/file support, with on-disk caching + LRU eviction.
        repo = entry["repo"]
        filename = entry["file"]
        cache_key = f"{repo}/{filename}"
        local_target = os.path.join(self.cache_dir, repo.replace("/", "__"), filename)

        if os.path.exists(local_target):
            logger.info("Using cached model file for '%s' at %s", alias, local_target)
            size = os.path.getsize(local_target)
            self.disk_cache.touch(cache_key, local_target, size)
            return local_target

        dl_cfg = self.config.get("download", {})
        max_retries = dl_cfg.get("max_retries", 4)
        backoff = dl_cfg.get("retry_backoff_seconds", 5)
        token = self.config.get("huggingface_token") or None

        disk_cfg = self.config.get("disk_management", {})
        if disk_cfg.get("enabled", True):
            remote_size = self._remote_file_size(repo, filename, token)
            if remote_size:
                ok = self.disk_cache.ensure_space_for(remote_size, protect_paths=self._protected_paths())
                if not ok:
                    free_gb = self.disk_cache.free_space_bytes(self.cache_dir) / (1024 ** 3)
                    raise RuntimeError(
                        f"Not enough disk space to download '{alias}' "
                        f"({remote_size / (1024 ** 3):.2f} GB needed, only "
                        f"{free_gb:.2f} GB free even after evicting all evictable "
                        f"cached models).")
            else:
                logger.info("Proceeding without a pre-download space check for '%s' "
                            "(remote size unknown).", alias)

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Downloading '%s' from %s (attempt %d/%d)...",
                            filename, repo, attempt, max_retries)
                path = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=os.path.join(self.cache_dir, repo.replace("/", "__")),
                    token=token,
                )
                logger.info("Downloaded model to %s", path)
                size = os.path.getsize(path)
                self.disk_cache.touch(cache_key, path, size)
                return path
            except Exception as e:
                last_err = e
                logger.warning("Download attempt %d failed: %s", attempt, e)
                # If a download failed partway through disk exhaustion, try
                # freeing more room before the next attempt too.
                if disk_cfg.get("enabled", True):
                    remote_size = self._remote_file_size(repo, filename, token)
                    if remote_size:
                        self.disk_cache.ensure_space_for(remote_size, protect_paths=self._protected_paths())
                time.sleep(backoff * attempt)
        raise RuntimeError(f"Failed to download model '{alias}' after {max_retries} attempts: {last_err}")

    def unload(self):
        if self.llm is not None:
            logger.info("Unloading current model '%s' to free VRAM.", self.current_alias)
            del self.llm
            self.llm = None
            self.current_alias = None
            self.current_local_path = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            # Give the driver a brief moment to actually reclaim VRAM before
            # anything queries free memory to size the next model's context.
            time.sleep(0.5)

    def _build_load_kwargs(self, alias: str, entry: dict, n_ctx: int) -> dict:
        gpu_cfg = self.config.get("gpu", {})
        n_gpu_layers = entry.get("n_gpu_layers", gpu_cfg.get("n_gpu_layers", -1)) if GPU_COUNT > 0 else 0
        n_batch = entry.get("n_batch", gpu_cfg.get("n_batch", 512))
        n_threads = entry.get("n_threads", gpu_cfg.get("n_threads", 0)) or os.cpu_count() or 4
        return dict(
            model_path=self.current_local_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def _try_load(self, kwargs: dict, attempt_dual: bool) -> Llama:
        if attempt_dual:
            try:
                split = [1.0 / GPU_COUNT] * GPU_COUNT
                return Llama(tensor_split=split, **kwargs)
            except Exception as e:
                if is_cuda_oom_error(e):
                    raise
                logger.warning("Dual-GPU split failed/did not help (%s); "
                                "falling back to single-GPU/CPU load.", e)
        return Llama(**kwargs)

    def ensure_loaded(self, alias: str) -> Llama:
        with self._load_lock:
            if self.current_alias == alias and self.llm is not None:
                return self.llm

            # Switching models (or first load): release whatever's currently
            # loaded FIRST so its VRAM is freed and counted as available
            # before we size the incoming model's context window. This is
            # what lets a second, larger model reclaim VRAM a first model
            # was occupying instead of trying (and failing) to fit both at
            # once -- only one model is ever kept resident at a time.
            if self.llm is not None:
                logger.info("Switching from '%s' to '%s' -- unloading current model first.",
                            self.current_alias, alias)
                self.unload()

            model_path = self.resolve_local_path(alias)
            self.current_local_path = model_path  # protects from disk eviction while loading/in-use

            entry = self.registry().get(alias, {})
            gpu_cfg = self.config.get("gpu", {})
            configured_n_ctx = entry.get("n_ctx", gpu_cfg.get("n_ctx", "max"))
            n_gpu_layers = entry.get("n_gpu_layers", gpu_cfg.get("n_gpu_layers", -1)) if GPU_COUNT > 0 else 0
            n_batch = entry.get("n_batch", gpu_cfg.get("n_batch", 512))

            # Probe the GGUF header once (cheap, CPU-only, no GPU needed) so
            # we know this model's own trained max context length -- used
            # both to resolve "max" and, on GPU workers, to size the
            # KV-cache. This is purely inference-time context sizing.
            hyper = probe_gguf_hyperparams(model_path)
            desired_n_ctx = resolve_desired_n_ctx(configured_n_ctx, hyper)
            logger.info(
                "Context resolution for '%s': configured gpu/model n_ctx=%r, "
                "GGUF header n_ctx_train=%s, resolved desired_n_ctx=%d (before any "
                "VRAM-based shrinking below).",
                alias, configured_n_ctx,
                hyper.get("n_ctx_train") if hyper else "unknown (header probe failed)",
                desired_n_ctx)

            n_ctx = desired_n_ctx
            if GPU_COUNT > 0:
                n_ctx = compute_dynamic_n_ctx(model_path, desired_n_ctx, n_gpu_layers,
                                               n_batch, self.vram, self.config, hyper=hyper)
            # CPU-only workers have nothing to size against, so they always
            # get the model's full trained max context (or the configured
            # override) unchanged.

            attempt_dual = GPU_COUNT >= 2 and gpu_cfg.get("attempt_dual_gpu_split", True)
            vram_cfg = self.config.get("vram_management", {})
            min_ctx = vram_cfg.get("min_n_ctx", 256)
            shrink_factor = vram_cfg.get("oom_shrink_factor", 0.75)
            max_retries = vram_cfg.get("max_oom_retries", 4)

            attempts = 0
            while True:
                kwargs = self._build_load_kwargs(alias, entry, n_ctx)
                try:
                    logger.info("Loading '%s' (n_ctx=%d, n_gpu_layers=%s)...",
                                alias, n_ctx, kwargs["n_gpu_layers"])
                    self.llm = self._try_load(kwargs, attempt_dual)
                    self.current_alias = alias
                    logger.info("Loaded '%s' with n_ctx=%d.", alias, n_ctx)
                    return self.llm
                except Exception as e:
                    oom = is_cuda_oom_error(e)
                    attempts += 1
                    if not oom or attempts >= max_retries or n_ctx <= min_ctx:
                        logger.error("Failed to load '%s': %s", alias, e)
                        self.current_local_path = None
                        raise
                    new_ctx = max(min_ctx, int(n_ctx * shrink_factor))
                    logger.warning(
                        "CUDA OOM while loading '%s' at n_ctx=%d; retrying with n_ctx=%d "
                        "(attempt %d/%d).", alias, n_ctx, new_ctx, attempts, max_retries)
                    n_ctx = new_ctx
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    time.sleep(0.5)


MODELS = ModelManager(CONFIG)


async def disk_monitor_loop():
    """Periodically checks free disk space independent of any download in
    progress, and evicts least-recently-used cached models if free space
    drops below the configured minimum. This catches disk pressure from
    causes other than 'about to download a model' (HF hub temp files,
    logs, other notebook output, etc.)."""
    disk_cfg = CONFIG.get("disk_management", {})
    if not disk_cfg.get("enabled", True):
        return
    interval = disk_cfg.get("monitor_interval_seconds", 30)
    min_free_bytes = int(disk_cfg.get("min_free_space_gb", 5) * (1024 ** 3))
    while True:
        await asyncio.sleep(interval)
        try:
            free = MODELS.disk_cache.free_space_bytes(MODELS.cache_dir)
            if free < min_free_bytes:
                logger.info(
                    "Disk monitor: %.2f GB free is below the configured %.2f GB minimum; "
                    "evicting least-recently-used cached models.",
                    free / (1024 ** 3), min_free_bytes / (1024 ** 3))
                MODELS.disk_cache.ensure_space_for(
                    min_free_bytes, protect_paths=MODELS._protected_paths())
        except Exception as e:
            logger.warning("Disk monitor loop error: %s", e)

# --------------------------------------------------------------------------- #
# Step 7: inference execution
# --------------------------------------------------------------------------- #


def build_chat_prompt_messages(messages: list) -> list:
    # llama-cpp-python's create_chat_completion accepts OpenAI-style message
    # dicts directly, so we pass them through mostly as-is.
    return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]


def build_gen_kwargs(payload: dict) -> dict:
    params = payload.get("params", {})
    defaults = CONFIG.get("default_generation", {})
    gen_kwargs = dict(
        temperature=params.get("temperature", defaults.get("temperature", 0.8)),
        top_p=params.get("top_p", defaults.get("top_p", 0.95)),
        max_tokens=params.get("max_tokens", defaults.get("max_tokens", 512)),
        stop=params.get("stop"),
        presence_penalty=params.get("presence_penalty", 0.0),
        frequency_penalty=params.get("frequency_penalty", 0.0),
    )
    if params.get("seed") is not None:
        gen_kwargs["seed"] = params["seed"]
    return gen_kwargs


def is_cuda_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


def recover_from_oom(model_alias: str, e: Exception) -> None:
    logger.error("CUDA OOM while running '%s': %s", model_alias, e)
    MODELS.unload()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_inference(payload: dict) -> dict:
    """Non-streaming path: runs synchronously (blocking) -- called from a
    worker thread via run_in_executor. Returns the full completion at once."""
    model_alias = payload["model"]
    gen_kwargs = build_gen_kwargs(payload)

    try:
        llm = MODELS.ensure_loaded(model_alias)
    except Exception as e:
        return {"error": f"model_load_failed: {e}"}

    try:
        if payload["kind"] == "chat":
            messages = build_chat_prompt_messages(payload["messages"])
            out = llm.create_chat_completion(messages=messages, **gen_kwargs)
            choice = out["choices"][0]
            text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
        else:
            prompt = payload["prompt"]
            out = llm(prompt=prompt, **gen_kwargs)
            choice = out["choices"][0]
            text = choice["text"]
            finish_reason = choice.get("finish_reason", "stop")

        usage = out.get("usage", {})
        return {
            "result": {
                "text": text,
                "finish_reason": finish_reason,
                "usage": usage,
            }
        }
    except RuntimeError as e:
        if is_cuda_oom_error(e):
            recover_from_oom(model_alias, e)
            return {"error": "cuda_out_of_memory: the model ran out of GPU memory for this request"}
        logger.error("Runtime error during inference: %s\n%s", e, traceback.format_exc())
        return {"error": f"inference_error: {e}"}
    except Exception as e:
        logger.error("Unexpected error during inference: %s\n%s", e, traceback.format_exc())
        return {"error": f"inference_error: {e}"}


def run_inference_streaming(payload: dict, on_token):
    """True token-by-token streaming path. Runs synchronously (blocking) in
    a background thread; calls on_token(delta_text, finish_reason) for every
    token/chunk llama.cpp produces, as it produces it. Raises on failure so
    the caller can distinguish a clean finish from an error.

    llama-cpp-python's stream=True mode already yields OpenAI-shaped chunk
    dicts, which is why the mapping below is so direct.
    """
    model_alias = payload["model"]
    gen_kwargs = build_gen_kwargs(payload)
    llm = MODELS.ensure_loaded(model_alias)  # let model-load errors propagate

    if payload["kind"] == "chat":
        messages = build_chat_prompt_messages(payload["messages"])
        stream = llm.create_chat_completion(messages=messages, stream=True, **gen_kwargs)
        for chunk in stream:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {}) or {}
            text = delta.get("content", "") or ""
            finish_reason = choice.get("finish_reason")
            if text or finish_reason:
                on_token(text, finish_reason)
    else:
        prompt = payload["prompt"]
        stream = llm(prompt=prompt, stream=True, **gen_kwargs)
        for chunk in stream:
            choice = chunk["choices"][0]
            text = choice.get("text", "") or ""
            finish_reason = choice.get("finish_reason")
            if text or finish_reason:
                on_token(text, finish_reason)


# --------------------------------------------------------------------------- #
# Step 8: transport layer -- WebSocket primary, long-poll fallback
# --------------------------------------------------------------------------- #


class ProxyClient:
    def __init__(self, config: dict):
        self.config = config
        self.proxy_url = config["proxy_url"].rstrip("/")
        self.ws_url = self.proxy_url.replace("http://", "ws://").replace("https://", "wss://") \
            + config["transport"]["ws_path"]
        self.worker_id = config["worker_id"]
        self.secret = config["worker_shared_secret"]
        self.seen_job_ids: set = set()  # de-dupe protection across retries
        self.ws_failures = 0
        self.loop = asyncio.get_event_loop()

    def _sign(self) -> Dict[str, str]:
        ts = str(time.time())
        sig = hmac.new(self.secret.encode(), f"{self.worker_id}:{ts}".encode(),
                        hashlib.sha256).hexdigest()
        return {"worker_id": self.worker_id, "timestamp": ts, "signature": sig}

    def _check_and_mark_duplicate(self, job_id: str) -> bool:
        """Returns True if this job_id was already processed (duplicate
        delivery -- e.g. a redelivery after a dropped ack). Safe to ignore."""
        if job_id in self.seen_job_ids:
            logger.info("Duplicate delivery of job %s ignored (already processed).", job_id)
            return True
        self.seen_job_ids.add(job_id)
        if len(self.seen_job_ids) > 10000:
            self.seen_job_ids = set(list(self.seen_job_ids)[-2000:])
        return False

    async def process_job(self, job_id: str, payload: dict) -> dict:
        if self._check_and_mark_duplicate(job_id):
            return {"job_id": job_id, "duplicate": True}
        result = await self.loop.run_in_executor(None, run_inference, payload)
        return {"job_id": job_id, **result}

    async def process_stream_job(self, ws, job_id: str, payload: dict) -> None:
        """Runs generation in a background thread, forwarding each token to
        the proxy as a 'stream_chunk' WS message the instant it's produced,
        then closes out with 'stream_done' (or 'error' on failure)."""
        if self._check_and_mark_duplicate(job_id):
            await ws.send(json.dumps({
                "type": "stream_done", "job_id": job_id, "result": {"duplicate": True},
            }))
            return

        queue: "asyncio.Queue" = asyncio.Queue()
        loop = self.loop
        model_alias = payload.get("model", "?")

        def on_token(delta: str, finish_reason: Optional[str]) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"delta": delta, "finish_reason": finish_reason})

        def worker_thread() -> None:
            try:
                run_inference_streaming(payload, on_token)
                loop.call_soon_threadsafe(queue.put_nowait, {"__done__": True})
            except RuntimeError as e:
                if is_cuda_oom_error(e):
                    recover_from_oom(model_alias, e)
                    err = "cuda_out_of_memory: the model ran out of GPU memory for this request"
                else:
                    logger.error("Runtime error during streaming inference: %s\n%s",
                                  e, traceback.format_exc())
                    err = f"inference_error: {e}"
                loop.call_soon_threadsafe(queue.put_nowait, {"__error__": err})
            except Exception as e:
                logger.error("Unexpected error during streaming inference: %s\n%s",
                              e, traceback.format_exc())
                loop.call_soon_threadsafe(queue.put_nowait, {"__error__": f"inference_error: {e}"})

        threading.Thread(target=worker_thread, daemon=True).start()

        completion_tokens = 0
        try:
            while True:
                item = await queue.get()
                if item.get("__done__"):
                    await ws.send(json.dumps({
                        "type": "stream_done", "job_id": job_id,
                        "result": {"usage": {"completion_tokens": completion_tokens}},
                    }))
                    return
                if "__error__" in item:
                    await ws.send(json.dumps({
                        "type": "error", "job_id": job_id, "error": item["__error__"],
                    }))
                    return
                if item.get("delta"):
                    completion_tokens += 1
                await ws.send(json.dumps({
                    "type": "stream_chunk", "job_id": job_id,
                    "delta": item.get("delta", ""), "finish_reason": item.get("finish_reason"),
                }))
        except Exception as e:
            # Most likely the WS connection itself dropped mid-stream. Not
            # much we can do to notify the proxy at this point; the proxy's
            # own disconnect handler / watchdog will fail/requeue this job.
            logger.warning("Lost connection while streaming job %s: %s", job_id, e)

    # ---------------------- WebSocket transport ---------------------- #

    async def run_websocket_forever(self):
        t_cfg = self.config["transport"]
        backoff = t_cfg["reconnect_initial_backoff_seconds"]
        max_backoff = t_cfg["reconnect_max_backoff_seconds"]

        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    hello = {"type": "hello", "models": self.config.get("models", {}), **self._sign()}
                    await ws.send(json.dumps(hello))
                    ack_raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    ack = json.loads(ack_raw)
                    if ack.get("type") != "hello_ack":
                        raise ConnectionError(f"Handshake rejected: {ack}")

                    logger.info("WebSocket connected and authenticated with proxy.")
                    self.ws_failures = 0
                    backoff = t_cfg["reconnect_initial_backoff_seconds"]

                    heartbeat_task = asyncio.create_task(self._ws_heartbeat_loop(ws))
                    try:
                        async for raw in ws:
                            msg = json.loads(raw)
                            if msg.get("type") == "job":
                                payload = msg["payload"]
                                await ws.send(json.dumps({"type": "ack", "job_id": msg["job_id"]}))
                                if payload.get("stream"):
                                    # True token-by-token streaming: this
                                    # call sends its own stream_chunk/
                                    # stream_done/error messages as it goes.
                                    await self.process_stream_job(ws, msg["job_id"], payload)
                                else:
                                    result = await self.process_job(msg["job_id"], payload)
                                    await ws.send(json.dumps({"type": "result", **result}))
                            elif msg.get("type") == "heartbeat_ack":
                                pass
                            else:
                                logger.debug("Unhandled WS message: %s", msg)
                    finally:
                        heartbeat_task.cancel()

            except (asyncio.CancelledError,):
                raise
            except Exception as e:
                self.ws_failures += 1
                logger.warning("WebSocket connection failed/dropped (%s). Failure count=%d",
                                e, self.ws_failures)
                if self.ws_failures >= t_cfg["ws_failure_threshold_before_poll_fallback"]:
                    logger.info("Too many WS failures, temporarily switching to long-poll fallback.")
                    return  # let the caller switch transport
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def _ws_heartbeat_loop(self, ws):
        interval = self.config["transport"]["heartbeat_interval_seconds"]
        while True:
            await asyncio.sleep(interval)
            try:
                await ws.send(json.dumps({
                    "type": "heartbeat",
                    "status": "idle" if MODELS.llm is None else "idle",
                    "current_model": MODELS.current_alias,
                }))
            except Exception:
                return

    # ---------------------- Long-poll transport ---------------------- #

    async def run_long_poll_forever(self, stop_after_seconds: float):
        """Runs the long-poll loop for a while, then returns so the caller
        can periodically retry upgrading back to WebSocket."""
        t_cfg = self.config["transport"]
        # Register so the proxy knows we exist even before the first poll returns.
        try:
            requests.post(f"{self.proxy_url}/worker/register", json={
                "models": self.config.get("models", {}), **self._sign()
            }, timeout=15)
        except Exception as e:
            logger.warning("Long-poll registration failed: %s", e)

        end_time = time.time() + stop_after_seconds
        while time.time() < end_time:
            try:
                params = {**self._sign(), "worker_id": self.worker_id}
                # requests is sync; run it off the event loop so heartbeats/etc
                # elsewhere (none currently) aren't blocked longer than needed.
                resp = await self.loop.run_in_executor(
                    None,
                    lambda: requests.get(f"{self.proxy_url}/worker/poll", params=params,
                                          timeout=t_cfg["long_poll_wait_seconds"] + 15),
                )
                if resp.status_code != 200:
                    logger.warning("Long-poll returned HTTP %s: %s", resp.status_code, resp.text[:200])
                    await asyncio.sleep(t_cfg["poll_interval_seconds"])
                    continue
                data = resp.json()
                job = data.get("job")
                if job is None:
                    # No work right now -- send a lightweight heartbeat and loop.
                    await self._poll_heartbeat()
                    continue
                result = await self.process_job(job["job_id"], job["payload"])
                await self.loop.run_in_executor(
                    None,
                    lambda: requests.post(f"{self.proxy_url}/worker/result",
                                           json={**self._sign(), **result}, timeout=30),
                )
            except Exception as e:
                logger.warning("Long-poll cycle error: %s", e)
                await asyncio.sleep(t_cfg["poll_interval_seconds"])

    async def _poll_heartbeat(self):
        try:
            await self.loop.run_in_executor(
                None,
                lambda: requests.post(f"{self.proxy_url}/worker/heartbeat", json={
                    **self._sign(),
                    "status": "idle" if MODELS.llm is None else "idle",
                    "current_model": MODELS.current_alias,
                    "models": self.config.get("models", {}),
                }, timeout=15),
            )
        except Exception:
            pass

    # ---------------------- Shutdown ---------------------- #

    async def deregister(self):
        """Best-effort 'I'm going away' signal so the proxy stops treating
        us as online immediately, instead of waiting on the WS close event
        (WS transport) or the offline-after-seconds grace window (long-poll
        transport). Safe to call even if we're not currently connected."""
        try:
            await self.loop.run_in_executor(
                None,
                lambda: requests.post(f"{self.proxy_url}/worker/deregister",
                                       json=self._sign(), timeout=10),
            )
            logger.info("Deregistered from proxy.")
        except Exception as e:
            logger.warning("Deregistration call failed (proxy may already consider us offline): %s", e)

    # ---------------------- Top-level orchestration ---------------------- #

    async def run_forever(self):
        """WebSocket-first with automatic long-poll fallback, and automatic
        re-attempts to upgrade back to WebSocket over time."""
        prefer_ws = self.config["transport"].get("prefer_websocket", True)
        while True:
            if prefer_ws:
                try:
                    await self.run_websocket_forever()
                except Exception as e:
                    logger.error("WebSocket loop crashed: %s", e)
                logger.info("Falling back to long-polling for a while before retrying WebSocket...")
            await self.run_long_poll_forever(stop_after_seconds=120)
            logger.info("Attempting to upgrade back to WebSocket transport...")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


async def main():
    logger.info("Starting GGUF Kaggle worker '%s' -> proxy %s",
                CONFIG["worker_id"], CONFIG["proxy_url"])
    client = ProxyClient(CONFIG)
    monitor_task = asyncio.create_task(disk_monitor_loop())
    try:
        while True:
            try:
                await client.run_forever()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error("Top-level worker loop crashed, restarting in 5s: %s\n%s",
                              e, traceback.format_exc())
                await asyncio.sleep(5)
    except KeyboardInterrupt:
        logger.info("Shutting down worker (KeyboardInterrupt).")
        await client.deregister()
        MODELS.unload()
        return
    finally:
        monitor_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        MODELS.unload()
        print("[worker] Stopped.")
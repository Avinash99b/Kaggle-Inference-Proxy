"""
inference_server_kaggle.py
================================================================================
Single-model Kaggle notebook inference worker.

Adapted from `run_model_with_proxy.py` for SINGLE-MODEL mode: loads exactly
one GGUF model at startup (selected via the TARGET_MODEL env var), connects
outbound to a proxy (WebSocket first, HTTP long-poll fallback), serves
chat/text completion jobs (including token streaming), reports GPU
utilization every 5 seconds via heartbeat, and shuts down cleanly on
KeyboardInterrupt (unloading the model and deregistering from the proxy).

There is no model-switching in this build: once TARGET_MODEL is loaded it
stays loaded for the lifetime of the process. If you need a different
model, restart the notebook cell with a different TARGET_MODEL.

Run this in a Kaggle notebook (ideally with the "GPU T4 x2" accelerator
enabled):

    export TARGET_MODEL="general-qwen3-8b"
    export PROXY_URL="https://kaggle-inference-proxy.onrender.com"
    export WORKER_ID="kaggle-account-1"
    export WORKER_SECRET="shared-secret-key"
    python inference_server_kaggle.py
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
from typing import Any, Dict, List, Optional, Set

# --------------------------------------------------------------------------- #
# Step -1: consolidate Hugging Face's own cache into our tracked cache dir.
#
# Same fix as the multi-model reference: HF's default cache
# (~/.cache/huggingface) lives on the same disk quota as everything else on
# Kaggle. Point HF_HOME/HF_HUB_CACHE at our own cache dir *before*
# huggingface_hub is imported (it reads these env vars once at import time),
# and never pass local_dir=... to hf_hub_download, so there is only ever one
# copy of each model blob on disk, and our eviction logic (LRU) can actually
# find and free it.
# --------------------------------------------------------------------------- #
_DEFAULT_MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/gguf_models")
os.environ.setdefault("HF_HOME", os.path.join(_DEFAULT_MODEL_CACHE_DIR, ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_DEFAULT_MODEL_CACHE_DIR, ".hf_cache", "hub"))
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)


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
# Step 1: configuration
#
# Single-model mode reads its identity/target from environment variables
# (set by the orchestrator/dashboard), and keeps a model REGISTRY (repo/file
# per alias) so TARGET_MODEL only needs to name an alias. Everything else
# (VRAM/disk/transport tuning) still comes from a local JSON config file,
# same as the reference script, so operators can tune it without touching
# code.
# --------------------------------------------------------------------------- #

CONFIG_PATH = os.environ.get("WORKER_CONFIG_PATH", "inference_server_config.json")

DEFAULT_CONFIG = {
    "model_cache_dir": _DEFAULT_MODEL_CACHE_DIR,
    "models": {
        "general-qwen3-30b-a3b": {"repo": "Qwen/Qwen3-30B-A3B-GGUF", "file": "Qwen3-30B-A3B-Q4_K_M.gguf"},
        "general-qwen3-14b": {"repo": "Qwen/Qwen3-14B-GGUF", "file": "Qwen3-14B-Q8_0.gguf"},
        "general-qwen3-8b": {"repo": "Qwen/Qwen3-8B-GGUF", "file": "Qwen3-8B-Q8_0.gguf"},
        "coding-qwen3-coder-30b-a3b": {
            "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            "file": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        },
        "coding-qwen2-5-coder-14b": {
            "repo": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
            "file": "qwen2.5-coder-14b-instruct-q8_0.gguf",
        },
        "coding-qwen2-5-coder-7b": {
            "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "file": "qwen2.5-coder-7b-instruct-q8_0.gguf",
        },
        "fast-rp-hermes-3-llama-8b": {
            "repo": "bartowski/Hermes-3-Llama-3.1-8B-GGUF",
            "file": "Hermes-3-Llama-3.1-8B-Q8_0.gguf",
        },
    },
    "default_generation": {"temperature": 0.8, "top_p": 0.95, "max_tokens": 512},
    "transport": {
        "prefer_websocket": True,
        "ws_path": "/worker/ws",
        "heartbeat_interval_seconds": 5,
        "reconnect_initial_backoff_seconds": 2,
        "reconnect_max_backoff_seconds": 60,
        "ws_failure_threshold_before_poll_fallback": 3,
        "poll_interval_seconds": 2,
        "long_poll_wait_seconds": 40,
    },
    "download": {"max_retries": 4, "retry_backoff_seconds": 5},
    "disk_management": {
        "enabled": True,
        "safety_margin_gb": 2,
        "min_free_space_gb": 5,
        "monitor_interval_seconds": 30,
    },
    "vram_management": {
        "headroom_fraction": 0.05,
        "min_n_ctx": 256,
        "oom_shrink_factor": 0.75,
        "max_oom_retries": 4,
        "max_auto_n_ctx": 32768,
    },
    "backend_build": {
        "prefer_prebuilt_wheel": True,
        "cmake_cuda_args": "-DGGML_CUDA=on",
    },
    "gpu": {
        "attempt_dual_gpu_split": True,
        "n_ctx": "max",
        "n_batch": 512,
        "n_threads": 0,
        "n_gpu_layers": -1,
    },
    "logging": {"level": "INFO", "file": "/tmp/inference_server.log"},
}


def load_or_create_config(path: str, defaults: dict) -> dict:
    abs_path = os.path.abspath(path)
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(defaults, f, indent=2)
        print(f"[worker] No config found at '{abs_path}'. Created a default config there.")
        return json.loads(json.dumps(defaults))
    with open(path, "r") as f:
        cfg = json.load(f)
    merged = json.loads(json.dumps(defaults))
    merged.update(cfg)
    for k, v in defaults.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merged[k] = {**v, **cfg[k]}
    return merged


CONFIG = load_or_create_config(CONFIG_PATH, DEFAULT_CONFIG)

# --- Env-var overrides (these are what the orchestrator/dashboard sets) --- #
TARGET_MODEL = os.environ.get("TARGET_MODEL")
PROXY_URL = os.environ.get("PROXY_URL", "https://kaggle-inference-proxy.onrender.com")
WORKER_ID = os.environ.get("WORKER_ID", "kaggle-worker-1")
WORKER_SECRET = os.environ.get("WORKER_SECRET", "change-me-worker-secret")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
WORKER_LOG_LEVEL = os.environ.get("WORKER_LOG_LEVEL", CONFIG["logging"].get("level", "INFO"))

if not TARGET_MODEL:
    print("[worker] FATAL: TARGET_MODEL environment variable is required (must be a key "
          "under 'models' in the config, e.g. 'general-qwen3-8b').")
    sys.exit(1)

CONFIG["worker_id"] = WORKER_ID
CONFIG["worker_shared_secret"] = WORKER_SECRET
CONFIG["proxy_url"] = PROXY_URL
CONFIG["huggingface_token"] = HUGGINGFACE_TOKEN
CONFIG["logging"]["level"] = WORKER_LOG_LEVEL

if CONFIG["model_cache_dir"] != _DEFAULT_MODEL_CACHE_DIR:
    _hf_cache = os.path.join(CONFIG["model_cache_dir"], ".hf_cache", "hub")
    os.makedirs(_hf_cache, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = _hf_cache
    import huggingface_hub.constants as _hf_constants
    _hf_constants.HF_HUB_CACHE = _hf_cache
    _hf_constants.HUGGINGFACE_HUB_CACHE = _hf_cache

if TARGET_MODEL not in CONFIG.get("models", {}):
    print(f"[worker] FATAL: TARGET_MODEL '{TARGET_MODEL}' is not in the model registry. "
          f"Known aliases: {sorted(CONFIG.get('models', {}).keys())}")
    sys.exit(1)

logging.basicConfig(
    level=getattr(logging, CONFIG["logging"].get("level", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(CONFIG["logging"].get("file", "/tmp/inference_server.log"))],
)
logger = logging.getLogger("worker")

os.makedirs(CONFIG["model_cache_dir"], exist_ok=True)

logger.info("Starting inference server worker '%s' -> proxy %s (target model: %s)",
            WORKER_ID, PROXY_URL, TARGET_MODEL)

# --------------------------------------------------------------------------- #
# Step 2: GPU detection
# --------------------------------------------------------------------------- #


def detect_gpus() -> int:
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
    """Per-GPU memory + utilization stats via nvidia-smi. [] if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
        stats = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            idx, total, used, free, util = parts
            stats.append({
                "index": int(idx),
                "total_mb": int(total),
                "used_mb": int(used),
                "free_mb": int(free),
                "utilization_percent": int(util),
            })
        return stats
    except Exception as e:
        logger.debug("nvidia-smi memory query failed: %s", e)
        return []


def aggregate_gpu_stats() -> dict:
    """Summed/averaged GPU stats across all detected GPUs, for heartbeat
    reporting. Returns zeros if no GPU is present or nvidia-smi is unavailable."""
    stats = query_gpu_memory()
    if not stats:
        return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "utilization_percent": 0}
    total = sum(s["total_mb"] for s in stats)
    used = sum(s["used_mb"] for s in stats)
    free = sum(s["free_mb"] for s in stats)
    avg_util = int(sum(s["utilization_percent"] for s in stats) / len(stats))
    return {"total_mb": total, "used_mb": used, "free_mb": free, "utilization_percent": avg_util}


class VramManager:
    """Tracks free VRAM to size context windows and keep a safety headroom."""

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
        totals = self.totals()
        return max(totals["free_bytes"] - self.headroom_bytes(), 0)


# --------------------------------------------------------------------------- #
# Step 3: install/build the CUDA-enabled llama.cpp backend
# --------------------------------------------------------------------------- #


def _llama_cpp_has_gpu_support() -> bool:
    """A successful `import llama_cpp` proves nothing about GPU support --
    a CPU-only build imports fine too. Always verify via the actual
    offload-support query before trusting an install."""
    try:
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as e:
        logger.debug("Could not query llama_cpp GPU offload support: %s", e)
        return False


def ensure_llama_cpp_backend():
    """Installs llama-cpp-python with CUDA support when a GPU is present.
    Every install path is followed by an explicit GPU-offload check and
    falls through to a more forceful strategy if that check fails --
    `import` succeeding is never treated as "done" on its own, since pip can
    silently fall back to a CPU-only source build if a CUDA wheel index
    doesn't have a match for the resolved version."""
    if GPU_COUNT == 0:
        logger.info("No GPU detected; installing CPU-only llama-cpp-python.")
        try:
            import llama_cpp  # noqa
            return
        except ImportError:
            pass
        _pip_install("llama-cpp-python")
        return

    try:
        import llama_cpp  # noqa
        if _llama_cpp_has_gpu_support():
            logger.info("llama-cpp-python already importable with CUDA support.")
            return
        logger.warning("llama-cpp-python is importable but was NOT built with CUDA "
                        "support even though %d GPU(s) are present; reinstalling.", GPU_COUNT)
    except ImportError:
        pass

    cfg = CONFIG["backend_build"]

    if cfg.get("prefer_prebuilt_wheel", True):
        try:
            logger.info("Attempting prebuilt CUDA wheel for llama-cpp-python...")
            _pip_install(
                "llama-cpp-python", "--force-reinstall", "--no-cache-dir",
                extra_index="https://abetlen.github.io/llama-cpp-python/whl/cu121",
            )
            if _llama_cpp_has_gpu_support():
                logger.info("Installed prebuilt CUDA llama-cpp-python wheel (verified GPU offload).")
                return
            logger.warning("Prebuilt-wheel install completed but GPU offload is NOT available; "
                            "trying an explicit CUDA source build instead.")
        except Exception as e:
            logger.warning("Prebuilt CUDA wheel install failed (%s); will try source build.", e)

    try:
        logger.info("Building llama-cpp-python from source with CUDA (%s)...",
                    cfg.get("cmake_cuda_args"))
        _pip_install(
            "llama-cpp-python", "--force-reinstall", "--no-cache-dir",
            env={"CMAKE_ARGS": cfg.get("cmake_cuda_args", "-DGGML_CUDA=on"),
                 "FORCE_CMAKE": "1"},
        )
        if _llama_cpp_has_gpu_support():
            logger.info("Built CUDA-enabled llama-cpp-python from source (verified GPU offload).")
            return
        logger.error("CUDA source build completed but GPU offload is still NOT available. "
                     "Falling back to CPU-only inference.")
    except Exception as e:
        logger.error("CUDA source build failed (%s). Falling back to CPU-only llama-cpp-python.", e)

    import llama_cpp  # noqa


ensure_llama_cpp_backend()
from llama_cpp import Llama  # noqa: E402

# --------------------------------------------------------------------------- #
# Step 4: disk cache manager (LRU eviction)
#
# Single-model mode still needs this: the target model has to be downloaded
# (and old, no-longer-needed cached blobs from a previous TARGET_MODEL run
# in this same notebook instance may need to be evicted to make room), but
# there is no "switch models while running" case anymore, so entries are
# never protected mid-run except the one model currently loaded.
# --------------------------------------------------------------------------- #


class DiskCacheManager:
    def __init__(self, cache_dir: str, config: dict):
        self.cache_dir = cache_dir
        self.config = config
        self.metadata_path = os.path.join(cache_dir, "cache_metadata.json")
        self._lock = threading.Lock()
        self.entries: Dict[str, dict] = self._load_metadata()
        self._reconcile_with_disk()

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
        with self._lock:
            missing = [k for k, v in self.entries.items()
                       if not os.path.exists(v.get("path", ""))]
            for k in missing:
                del self.entries[k]
            if missing:
                self._save_metadata()

    @staticmethod
    def free_space_bytes(path: str) -> int:
        return shutil.disk_usage(path).free

    def touch(self, cache_key: str, local_path: str, size_bytes: int) -> None:
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
        real_path = os.path.realpath(path) if os.path.islink(path) else path
        for p in {path, real_path}:
            try:
                if os.path.exists(p) or os.path.islink(p):
                    os.remove(p)
            except Exception as e:
                logger.warning("Failed to remove cached file '%s' for '%s': %s", p, cache_key, e)
        logger.info("Evicted cached model '%s' (%.2f GB) to free disk space.",
                    cache_key, entry.get("size_bytes", 0) / (1024 ** 3))
        parent = os.path.dirname(path)
        try:
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except Exception:
            pass

    def ensure_space_for(self, required_bytes: int, protect_paths: Optional[Set[str]] = None) -> bool:
        protect_paths = protect_paths or set()
        disk_cfg = self.config.get("disk_management", {})
        margin_bytes = int(disk_cfg.get("safety_margin_gb", 2) * (1024 ** 3))
        target = required_bytes + margin_bytes

        free = self.free_space_bytes(self.cache_dir)
        if free >= target:
            return True

        logger.info("Low disk space: %.2f GB free, need %.2f GB. Evicting LRU cached models...",
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
                "Only %.2f GB free after evicting everything evictable (%.2f GB requested "
                "including margin). Proceeding only if the raw file itself still fits.",
                free / (1024 ** 3), target / (1024 ** 3))
            return free >= required_bytes
        return True


# --------------------------------------------------------------------------- #
# Step 5: GGUF header introspection (for VRAM/KV-cache estimation)
# --------------------------------------------------------------------------- #


def probe_gguf_hyperparams(model_path: str) -> Optional[dict]:
    try:
        probe = Llama(model_path=model_path, vocab_only=True, n_ctx=8,
                       n_gpu_layers=0, verbose=True)
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
                        "(arch=%r); using a coarser VRAM estimate.", model_path, arch)
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
    n_layer = hyper["n_layer"]
    n_embd = hyper["n_embd"]
    n_head = hyper["n_head"]
    n_head_kv = hyper.get("n_head_kv") or n_head
    head_dim = n_embd / n_head
    kv_embd = head_dim * n_head_kv
    return int(n_layer * kv_embd * 2 * kv_dtype_bytes)


def resolve_desired_n_ctx(configured: Any, hyper: Optional[dict],
                           fallback_max: int = 32768) -> int:
    if isinstance(configured, str) and configured.strip().lower() == "max":
        if hyper and hyper.get("n_ctx_train"):
            return int(hyper["n_ctx_train"])
        logger.info("Could not read the model's trained max context from its GGUF header; "
                    "falling back to n_ctx=%d before VRAM sizing.", fallback_max)
        return fallback_max
    return int(configured)


def compute_dynamic_n_ctx(model_path: str, desired_n_ctx: int, n_gpu_layers: int,
                           n_batch: int, vram: VramManager, config: dict,
                           hyper: Optional[dict] = None) -> int:
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
        weight_bytes = file_size
        kv_per_token = 128 * 1024
        n_ctx_cap = desired_n_ctx

    flat_overhead = max(512 * 1024 ** 2, n_batch * 1024 * 1024)
    per_token_compute_overhead = 2 * 1024  # ~2KB/token, conservative estimate
    available_for_kv = usable_free - weight_bytes - flat_overhead

    if available_for_kv <= 0:
        max_ctx_by_vram = 0
    else:
        max_ctx_by_vram = int(available_for_kv // max(kv_per_token + per_token_compute_overhead, 1))

    final_ctx = min(desired_n_ctx, n_ctx_cap or desired_n_ctx)

    max_auto_n_ctx = vram_cfg.get("max_auto_n_ctx")
    if max_auto_n_ctx:
        final_ctx = min(final_ctx, int(max_auto_n_ctx))

    if max_ctx_by_vram > 0:
        final_ctx = min(final_ctx, max_ctx_by_vram)
    else:
        final_ctx = min(final_ctx, min_ctx)
    final_ctx = max(final_ctx, min_ctx)

    if final_ctx < desired_n_ctx:
        logger.info(
            "Sizing context for VRAM/safety headroom: requested n_ctx=%d, using n_ctx=%d "
            "(usable free VRAM=%.2f GB, est. weight VRAM=%.2f GB, est. KV/token=%d bytes).",
            desired_n_ctx, final_ctx, usable_free / (1024 ** 3),
            weight_bytes / (1024 ** 3), kv_per_token)
    return final_ctx


def is_cuda_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


# --------------------------------------------------------------------------- #
# Step 6: single-model manager (download/load ONE model at startup)
#
# No `ensure_loaded(alias)` / model-switching here. `load_startup_model()`
# resolves + downloads + loads TARGET_MODEL exactly once; after that the
# model just stays resident until process shutdown, at which point
# `unload()` is called once for a clean exit.
# --------------------------------------------------------------------------- #


class ModelManager:
    def __init__(self, config: dict, target_alias: str):
        self.config = config
        self.target_alias = target_alias
        self.cache_dir = config["model_cache_dir"]
        self.current_local_path: Optional[str] = None
        self.llm: Optional[Llama] = None
        self.status: str = "loading_model"  # "loading_model" | "idle" | "busy"
        self.disk_cache = DiskCacheManager(self.cache_dir, config)
        self.vram = VramManager(config)
        self._load_lock = threading.Lock()

    def registry(self) -> Dict[str, Any]:
        return self.config.get("models", {})

    def _remote_file_size(self, repo: str, filename: str, token: Optional[str]) -> Optional[int]:
        try:
            url = hf_hub_url(repo_id=repo, filename=filename)
            meta = get_hf_file_metadata(url, token=token)
            return getattr(meta, "size", None)
        except Exception as e:
            logger.warning("Could not fetch remote size for %s/%s: %s", repo, filename, e)
            return None

    def _resolve_local_path(self) -> str:
        entry = self.registry().get(self.target_alias)
        if entry is None:
            raise ValueError(f"Model alias '{self.target_alias}' is not in the model registry.")
        if "path" in entry:
            path = entry["path"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Configured local model path does not exist: {path}")
            return path

        repo = entry["repo"]
        filename = entry["file"]
        cache_key = f"{repo}/{filename}"
        token = self.config.get("huggingface_token") or None

        dl_cfg = self.config.get("download", {})
        max_retries = dl_cfg.get("max_retries", 4)
        backoff = dl_cfg.get("retry_backoff_seconds", 5)

        already_cached = cache_key in self.disk_cache.entries and \
            os.path.exists(self.disk_cache.entries[cache_key]["path"])
        if not already_cached:
            disk_cfg = self.config.get("disk_management", {})
            if disk_cfg.get("enabled", True):
                remote_size = self._remote_file_size(repo, filename, token)
                if remote_size:
                    ok = self.disk_cache.ensure_space_for(remote_size)
                    if not ok:
                        free_gb = self.disk_cache.free_space_bytes(self.cache_dir) / (1024 ** 3)
                        raise RuntimeError(
                            f"Not enough disk space to download '{self.target_alias}' "
                            f"({remote_size / (1024 ** 3):.2f} GB needed, only "
                            f"{free_gb:.2f} GB free even after evicting all evictable cached models).")
                else:
                    logger.info("Proceeding without a pre-download space check for '%s' "
                                "(remote size unknown).", self.target_alias)

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Resolving '%s' from %s (attempt %d/%d)...", filename, repo, attempt, max_retries)
                path = hf_hub_download(repo_id=repo, filename=filename, cache_dir=self.cache_dir, token=token)
                logger.info("Model resolved at %s", path)
                size = os.path.getsize(path)
                self.disk_cache.touch(cache_key, path, size)
                return path
            except Exception as e:
                last_err = e
                logger.warning("Download attempt %d failed: %s", attempt, e)
                disk_cfg = self.config.get("disk_management", {})
                if disk_cfg.get("enabled", True):
                    remote_size = self._remote_file_size(repo, filename, token)
                    if remote_size:
                        self.disk_cache.ensure_space_for(remote_size)
                time.sleep(backoff * attempt)
        raise RuntimeError(f"Failed to download model '{self.target_alias}' after {max_retries} attempts: {last_err}")

    def _build_load_kwargs(self, entry: dict, n_ctx: int) -> dict:
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
                logger.warning("Dual-GPU split failed/did not help (%s); falling back.", e)
        return Llama(**kwargs)

    def load_startup_model(self) -> Llama:
        """Downloads (if needed) and loads TARGET_MODEL exactly once. Meant
        to be called a single time during startup, before the proxy
        connection is established."""
        with self._load_lock:
            self.status = "loading_model"
            model_path = self._resolve_local_path()
            self.current_local_path = model_path

            entry = self.registry().get(self.target_alias, {})
            gpu_cfg = self.config.get("gpu", {})
            configured_n_ctx = entry.get("n_ctx", gpu_cfg.get("n_ctx", "max"))
            n_gpu_layers = entry.get("n_gpu_layers", gpu_cfg.get("n_gpu_layers", -1)) if GPU_COUNT > 0 else 0
            n_batch = entry.get("n_batch", gpu_cfg.get("n_batch", 512))

            hyper = probe_gguf_hyperparams(model_path)
            desired_n_ctx = resolve_desired_n_ctx(configured_n_ctx, hyper)
            logger.info(
                "Context resolution for '%s': configured n_ctx=%r, GGUF n_ctx_train=%s, "
                "resolved desired_n_ctx=%d (before VRAM/safety shrinking).",
                self.target_alias, configured_n_ctx,
                hyper.get("n_ctx_train") if hyper else "unknown", desired_n_ctx)
            if hyper:
                logger.info("GGUF header: n_ctx_train=%s, n_layer=%s, n_embd=%s",
                            hyper.get("n_ctx_train"), hyper.get("n_layer"), hyper.get("n_embd"))

            n_ctx = desired_n_ctx
            if GPU_COUNT > 0:
                n_ctx = compute_dynamic_n_ctx(model_path, desired_n_ctx, n_gpu_layers,
                                               n_batch, self.vram, self.config, hyper=hyper)

            attempt_dual = GPU_COUNT >= 2 and gpu_cfg.get("attempt_dual_gpu_split", True)
            vram_cfg = self.config.get("vram_management", {})
            min_ctx = vram_cfg.get("min_n_ctx", 256)
            shrink_factor = vram_cfg.get("oom_shrink_factor", 0.75)
            max_retries = vram_cfg.get("max_oom_retries", 4)

            attempts = 0
            while True:
                kwargs = self._build_load_kwargs(entry, n_ctx)
                try:
                    logger.info("Loading '%s' (n_ctx=%d, n_gpu_layers=%s)...",
                                self.target_alias, n_ctx, kwargs["n_gpu_layers"])
                    self.llm = self._try_load(kwargs, attempt_dual)
                    self.status = "idle"
                    gpu_stats = aggregate_gpu_stats()
                    logger.info("Loaded '%s' with n_ctx=%d (GPU: %.1fGB/%.1fGB used).",
                                self.target_alias, n_ctx,
                                gpu_stats["used_mb"] / 1024, gpu_stats["total_mb"] / 1024)
                    return self.llm
                except Exception as e:
                    oom = is_cuda_oom_error(e)
                    attempts += 1
                    if not oom or attempts >= max_retries or n_ctx <= min_ctx:
                        logger.error("Failed to load '%s': %s", self.target_alias, e)
                        self.status = "idle"
                        raise
                    new_ctx = max(min_ctx, int(n_ctx * shrink_factor))
                    logger.warning("CUDA OOM loading '%s' at n_ctx=%d; retrying with n_ctx=%d (%d/%d).",
                                    self.target_alias, n_ctx, new_ctx, attempts, max_retries)
                    n_ctx = new_ctx
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    time.sleep(0.5)

    def unload(self):
        """Called once, on shutdown. There is no reload afterwards."""
        if self.llm is not None:
            logger.info("Unloading model '%s'.", self.target_alias)
            del self.llm
            self.llm = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


MODELS = ModelManager(CONFIG, TARGET_MODEL)


async def disk_monitor_loop():
    """Periodically evicts LRU cached models (other than the currently
    loaded one's backing file) if free space drops below the configured
    minimum."""
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
                protect = {MODELS.current_local_path} if MODELS.current_local_path else set()
                logger.info("Disk monitor: %.2f GB free < %.2f GB minimum; evicting LRU cached models.",
                            free / (1024 ** 3), min_free_bytes / (1024 ** 3))
                MODELS.disk_cache.ensure_space_for(min_free_bytes, protect_paths=protect)
        except Exception as e:
            logger.warning("Disk monitor loop error: %s", e)

# --------------------------------------------------------------------------- #
# Step 7: inference execution
# --------------------------------------------------------------------------- #


def build_chat_prompt_messages(messages: list) -> list:
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


def recover_from_oom(model_alias: str, e: Exception) -> None:
    """Single-model mode has nowhere to fall back to: we don't unload+reload
    mid-run. We just free what CUDA cache we can and surface the error --
    the caller returns a cuda_out_of_memory error for this request only."""
    logger.error("CUDA OOM while running '%s': %s", model_alias, e)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_inference(payload: dict) -> dict:
    """Non-streaming path: blocking, called via run_in_executor."""
    model_alias = payload.get("model", TARGET_MODEL)
    if model_alias != TARGET_MODEL:
        return {"error": f"model_not_available: this worker only serves '{TARGET_MODEL}' "
                          f"(requested '{model_alias}')"}
    gen_kwargs = build_gen_kwargs(payload)

    if MODELS.llm is None:
        return {"error": "model_load_failed: model is not loaded"}

    MODELS.status = "busy"
    start = time.time()
    try:
        if payload["kind"] == "chat":
            messages = build_chat_prompt_messages(payload["messages"])
            out = MODELS.llm.create_chat_completion(messages=messages, **gen_kwargs)
            choice = out["choices"][0]
            text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
        else:
            prompt = payload["prompt"]
            out = MODELS.llm(prompt=prompt, **gen_kwargs)
            choice = out["choices"][0]
            text = choice["text"]
            finish_reason = choice.get("finish_reason", "stop")

        usage = out.get("usage", {})
        elapsed = time.time() - start
        logger.info("Job for '%s' completed in %.2fs (completion_tokens=%s).",
                    model_alias, elapsed, usage.get("completion_tokens"))
        return {"result": {"text": text, "finish_reason": finish_reason, "usage": usage}}
    except RuntimeError as e:
        if is_cuda_oom_error(e):
            recover_from_oom(model_alias, e)
            return {"error": "cuda_out_of_memory: the model ran out of GPU memory for this request"}
        logger.error("Runtime error during inference: %s\n%s", e, traceback.format_exc())
        return {"error": f"inference_error: {e}"}
    except Exception as e:
        logger.error("Unexpected error during inference: %s\n%s", e, traceback.format_exc())
        return {"error": f"inference_error: {e}"}
    finally:
        MODELS.status = "idle"


def run_inference_streaming(payload: dict, on_token):
    """Token-by-token streaming path (blocking; run in a background thread).
    Calls on_token(delta_text, finish_reason) as tokens are produced."""
    model_alias = payload.get("model", TARGET_MODEL)
    if model_alias != TARGET_MODEL:
        raise RuntimeError(f"model_not_available: this worker only serves '{TARGET_MODEL}'")
    if MODELS.llm is None:
        raise RuntimeError("model_load_failed: model is not loaded")

    gen_kwargs = build_gen_kwargs(payload)
    MODELS.status = "busy"
    start = time.time()
    token_count = 0
    try:
        if payload["kind"] == "chat":
            messages = build_chat_prompt_messages(payload["messages"])
            stream = MODELS.llm.create_chat_completion(messages=messages, stream=True, **gen_kwargs)
            for chunk in stream:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {}) or {}
                text = delta.get("content", "") or ""
                finish_reason = choice.get("finish_reason")
                if text:
                    token_count += 1
                if text or finish_reason:
                    on_token(text, finish_reason)
        else:
            prompt = payload["prompt"]
            stream = MODELS.llm(prompt=prompt, stream=True, **gen_kwargs)
            for chunk in stream:
                choice = chunk["choices"][0]
                text = choice.get("text", "") or ""
                finish_reason = choice.get("finish_reason")
                if text:
                    token_count += 1
                if text or finish_reason:
                    on_token(text, finish_reason)
        logger.info("Streaming job for '%s' completed in %.2fs (~%d tokens).",
                    model_alias, time.time() - start, token_count)
    finally:
        MODELS.status = "idle"


# --------------------------------------------------------------------------- #
# Step 8: transport layer -- WebSocket primary, long-poll fallback
# --------------------------------------------------------------------------- #


class ProxyClient:
    def __init__(self, config: dict, target_model: str):
        self.config = config
        self.target_model = target_model
        self.proxy_url = config["proxy_url"].rstrip("/")
        self.ws_url = self.proxy_url.replace("http://", "ws://").replace("https://", "wss://") \
            + config["transport"]["ws_path"]
        self.worker_id = config["worker_id"]
        self.secret = config["worker_shared_secret"]
        self.seen_job_ids: set = set()
        self.ws_failures = 0
        self.loop = asyncio.get_event_loop()

    def _sign(self) -> Dict[str, str]:
        ts = str(time.time())
        sig = hmac.new(self.secret.encode(), f"{self.worker_id}:{ts}".encode(),
                        hashlib.sha256).hexdigest()
        return {"worker_id": self.worker_id, "timestamp": ts, "signature": sig}

    def _heartbeat_payload(self) -> dict:
        gpu_stats = aggregate_gpu_stats()
        return {
            "type": "heartbeat",
            "worker_id": self.worker_id,
            "timestamp": str(time.time()),
            "gpu_stats": gpu_stats,
            "current_model": self.target_model,
            "status": MODELS.status,
        }

    def _check_and_mark_duplicate(self, job_id: str) -> bool:
        if job_id in self.seen_job_ids:
            logger.info("Duplicate delivery of job %s ignored.", job_id)
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
        if self._check_and_mark_duplicate(job_id):
            await ws.send(json.dumps({
                "type": "stream_done", "job_id": job_id, "result": {"duplicate": True},
            }))
            return

        queue: "asyncio.Queue" = asyncio.Queue()
        loop = self.loop

        def on_token(delta: str, finish_reason: Optional[str]) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"delta": delta, "finish_reason": finish_reason})

        def worker_thread() -> None:
            try:
                run_inference_streaming(payload, on_token)
                loop.call_soon_threadsafe(queue.put_nowait, {"__done__": True})
            except RuntimeError as e:
                if is_cuda_oom_error(e):
                    recover_from_oom(self.target_model, e)
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
            logger.warning("Lost connection while streaming job %s: %s", job_id, e)

    async def run_websocket_forever(self):
        t_cfg = self.config["transport"]
        backoff = t_cfg["reconnect_initial_backoff_seconds"]
        max_backoff = t_cfg["reconnect_max_backoff_seconds"]

        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    hello = {"type": "hello", "models": {self.target_model: self.config["models"][self.target_model]},
                             **self._sign()}
                    await ws.send(json.dumps(hello))
                    ack_raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    ack = json.loads(ack_raw)
                    if ack.get("type") != "hello_ack":
                        raise ConnectionError(f"Handshake rejected: {ack}")

                    logger.info("WebSocket connected and authenticated with proxy.")
                    logger.info("Ready to serve inference requests.")
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
                    logger.info("Too many WS failures, switching to long-poll fallback.")
                    return
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def _ws_heartbeat_loop(self, ws):
        interval = self.config["transport"].get("heartbeat_interval_seconds", 5)
        while True:
            await asyncio.sleep(interval)
            try:
                await ws.send(json.dumps(self._heartbeat_payload()))
            except Exception:
                return

    async def run_long_poll_forever(self, stop_after_seconds: float):
        """Runs long-poll for a while (including its own periodic heartbeat
        posts), then returns so the caller can retry upgrading back to WS."""
        t_cfg = self.config["transport"]
        try:
            requests.post(f"{self.proxy_url}/worker/register", json={
                "models": {self.target_model: self.config["models"][self.target_model]},
                **self._sign(),
            }, timeout=15)
        except Exception as e:
            logger.warning("Long-poll registration failed: %s", e)

        heartbeat_interval = t_cfg.get("heartbeat_interval_seconds", 5)
        last_heartbeat = 0.0
        end_time = time.time() + stop_after_seconds
        while time.time() < end_time:
            try:
                if time.time() - last_heartbeat >= heartbeat_interval:
                    await self._poll_heartbeat()
                    last_heartbeat = time.time()

                params = {**self._sign(), "worker_id": self.worker_id}
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
                lambda: requests.post(f"{self.proxy_url}/worker/heartbeat",
                                       json=self._heartbeat_payload(), timeout=15),
            )
        except Exception:
            pass

    async def deregister(self):
        """Tells the proxy we're going away so it stops waiting on us."""
        try:
            await self.loop.run_in_executor(
                None,
                lambda: requests.post(f"{self.proxy_url}/worker/deregister",
                                       json=self._sign(), timeout=10),
            )
            logger.info("Deregistered from proxy.")
        except Exception as e:
            logger.warning("Deregistration call failed (proxy may already consider us offline): %s", e)

    async def run_forever(self):
        prefer_ws = self.config["transport"].get("prefer_websocket", True)
        while True:
            if prefer_ws:
                try:
                    await self.run_websocket_forever()
                except Exception as e:
                    logger.error("WebSocket loop crashed: %s", e)
                logger.info("Falling back to long-polling before retrying WebSocket...")
            await self.run_long_poll_forever(stop_after_seconds=120)
            logger.info("Attempting to upgrade back to WebSocket transport...")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


async def main():
    # Step 1/2: download + load the single target model BEFORE connecting to
    # the proxy, so the worker never advertises itself as ready until it
    # actually has something to serve.
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, MODELS.load_startup_model)
    except Exception as e:
        logger.error("FATAL: could not load target model '%s': %s\n%s",
                     TARGET_MODEL, e, traceback.format_exc())
        sys.exit(1)

    client = ProxyClient(CONFIG, TARGET_MODEL)
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
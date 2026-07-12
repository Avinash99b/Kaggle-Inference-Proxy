"""
run_model_with_proxy.py
================================================================================
Private Kaggle GGUF inference worker.

Run this in a Kaggle notebook (ideally with the "GPU T4 x2" accelerator
enabled). It installs dependencies, builds a CUDA llama.cpp backend, connects
outbound to the proxy (WebSocket first, long-poll fallback), downloads/caches
GGUF models from Hugging Face on demand, runs generation, and reports results
back. It manages disk space (LRU eviction) and VRAM (dynamic context sizing)
so it can run unattended.

Just run:
    python run_model_with_proxy.py
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
# Step -1: consolidate Hugging Face's own cache into our tracked cache dir.
#
# BUG FIX: hf_hub_download's default cache lives under ~/.cache/huggingface,
# which on Kaggle sits on the SAME small root volume as everything else
# (despite /tmp "feeling" bigger, `df` reports one shared quota for the
# instance). Our old code downloaded with `local_dir=...`, which made
# huggingface_hub materialize a full second copy of every model file
# outside of that hidden cache. Two problems resulted:
#   1. Every model was stored twice on disk (double the space per model).
#   2. Our eviction logic only ever deleted the `local_dir` copy it knew
#      about -- the hidden ~/.cache/huggingface blob was never freed, so
#      disk usage crept up over time even though our own accounting said
#      there was room, eventually blowing through the real quota
#      (the "59.1GiB used / 57.6GiB max" crash).
#
# Fix: point HF's cache at a subdirectory of our own (tracked, evictable)
# cache dir, and stop using `local_dir` so huggingface_hub uses its normal
# single-copy, symlinked cache instead of materializing a duplicate file.
# This must happen before huggingface_hub is imported, since it reads these
# env vars once at import time.
# --------------------------------------------------------------------------- #
_DEFAULT_MODEL_CACHE_DIR = "/tmp/gguf_models"
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
# Step 1: config load / generate
# --------------------------------------------------------------------------- #

CONFIG_PATH = os.environ.get("WORKER_CONFIG_PATH", "inference_server_config.json")

DEFAULT_CONFIG = {
    "proxy_url": "https://kaggle-inference-proxy.onrender.com",
    "worker_id": "kaggle-worker-1",
    "worker_shared_secret": "change-me-worker-secret",
    "huggingface_token": "",
    "model_cache_dir": _DEFAULT_MODEL_CACHE_DIR,
    "models": {
        "general-qwen3-30b-a3b": {
            "repo": "Qwen/Qwen3-30B-A3B-GGUF",
            "file": "Qwen3-30B-A3B-Q4_K_M.gguf",
        },
        "general-qwen3-14b": {
            "repo": "Qwen/Qwen3-14B-GGUF",
            "file": "Qwen3-14B-Q8_0.gguf",
        },
        "general-qwen3-8b": {
            "repo": "Qwen/Qwen3-8B-GGUF",
            "file": "Qwen3-8B-Q8_0.gguf",
        },
        "coding-qwen3-coder-30b-a3b": {
            "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            "file": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        },
        "coding-qwen3-coder-next": {
            "repo": "Qwen/Qwen3-Coder-Next-GGUF",
            "file": "Qwen3-Coder-Next-Q4_K_M-00001-of-00003.gguf",
        },
        "coding-qwen2-5-coder-14b": {
            "repo": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
            "file": "qwen2.5-coder-14b-instruct-q8_0.gguf",
        },
        "coding-qwen2-5-coder-7b": {
            "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "file": "qwen2.5-coder-7b-instruct-q8_0.gguf",
        },
        "rp-cydonia-24b-v4-3": {
            "repo": "bartowski/TheDrummer_Cydonia-24B-v4.3-GGUF",
            "file": "TheDrummer_Cydonia-24B-v4.3-Q4_K_M.gguf",
        },
        "rp-dolphin-mistral-24b-venice": {
            "repo": "bartowski/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-GGUF",
            "file": "cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_M.gguf",
        },
        "rp-dolphin-mistral-24b-venice-single-q4": {
            "repo": "JohnRoger/Dolphin-Mistral-24B-Venice-Edition-Q4_K_M-GGUF",
            "file": "dolphin-mistral-24b-venice-edition-q4_k_m.gguf",
        },
        "rp-hexis-pure-soul-24b": {
            "repo": "WasamiKirua/Hexis-Pure-Soul-24B-GGUF",
            "file": "Hexis-Pure-Soul-24B-Q5_K_M.gguf",
        },
        "rp-hexis-pure-soul-24b-i1": {
            "repo": "mradermacher/Hexis-Pure-Soul-24B-i1-GGUF",
            "file": "Hexis-Pure-Soul-24B-i1.Q5_K_M.gguf",
        },
        "rp-gemma3-27b-derestricted": {
            "repo": "mradermacher/Gemma-3-27B-Derestricted-GGUF",
            "file": "Gemma-3-27B-Derestricted.Q4_K_M.gguf",
        },
        "rp-gemma3-27b-derestricted-q5": {
            "repo": "mradermacher/Gemma-3-27B-Derestricted-GGUF",
            "file": "Gemma-3-27B-Derestricted.Q5_K_M.gguf",
        },
        "rp-gemma3-27b-heretic": {
            "repo": "mradermacher/Gemma-3-27B-Heretic-i1-GGUF",
            "file": "gemma-3-27b-heretic.i1-Q4_K_M.gguf",
        },
        "rp-gemma3-27b-abliterated": {
            "repo": "mlabonne/gemma-3-27b-it-abliterated-GGUF",
            "file": "gemma-3-27b-it-abliterated.Q4_K_M.gguf",
        },
        "rp-big-tiger-gemma-27b-v3": {
            "repo": "TheDrummer/Big-Tiger-Gemma-27B-v3-GGUF",
            "file": "Big-Tiger-Gemma-27B-v3-Q4_K_M.gguf",
        },
        "rp-qwen3-30b-a3b-abliterated": {
            "repo": "mradermacher/Qwen3-30B-A3B-abliterated-GGUF",
            "file": "Qwen3-30B-A3B-abliterated.Q4_K_M.gguf",
        },
        "rp-qwen3-6-27b-heretic-v2": {
            "repo": "Jolex98/Qwen3.6-27B-uncensored-heretic-v2-GGUF",
            "file": "Qwen3.6-27B-uncensored-heretic-v2-Q4_K_M.gguf",
        },
        "uncensored-general-gpt-oss-20b": {
            "repo": "bartowski/p-e-w_gpt-oss-20b-heretic-GGUF",
            "file": "p-e-w_gpt-oss-20b-heretic-Q4_K_M.gguf",
        },
        "uncensored-general-qwq-32b": {
            "repo": "bartowski/huihui-ai_QwQ-32B-abliterated-GGUF",
            "file": "huihui-ai_QwQ-32B-abliterated-Q4_K_M.gguf",
        },
        "legacy-rp-mythomax-l2-13b": {
            "repo": "TheBloke/MythoMax-L2-13B-GGUF",
            "file": "mythomax-l2-13b.Q8_0.gguf",
        },
        "fast-rp-hermes-3-llama-8b": {
            "repo": "bartowski/Hermes-3-Llama-3.1-8B-GGUF",
            "file": "Hermes-3-Llama-3.1-8B-Q8_0.gguf",
        },
        "fast-uncensored-dolphin-llama-8b": {
            "repo": "dphn/Dolphin3.0-Llama3.1-8B-GGUF",
            "file": "Dolphin3.0-Llama3.1-8B-Q8_0.gguf",
        },
    },
    "default_generation": {"temperature": 0.8, "top_p": 0.95, "max_tokens": 512},
    "transport": {
        "prefer_websocket": True,
        "ws_path": "/worker/ws",
        "heartbeat_interval_seconds": 15,
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
        # Fraction of total VRAM always kept free as headroom.
        "headroom_fraction": 0.05,
        "min_n_ctx": 256,
        "oom_shrink_factor": 0.75,
        "max_oom_retries": 4,
        # BUG FIX: "max" used to mean "the model's full trained context,
        # shrunk only as far as VRAM math says is needed" -- with no upper
        # bound. For models trained with huge native windows (e.g. this
        # Qwen3-Coder-1M GGUF), that VRAM math alone produced n_ctx=215017,
        # which is unusable in practice: llama.cpp's compute buffers
        # (attention/KQ scratch, batch buffers, etc.) scale with n_ctx too,
        # not just KV-cache, and were never budgeted for. The resulting
        # allocation blew past system RAM and Kaggle hard-restarted the
        # notebook. This cap bounds "max" auto-sizing to something that
        # actually fits real hardware; raise it explicitly per-model if you
        # know your box can handle more.
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
    "logging": {"level": "INFO", "file": "worker.log"},
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
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
    print(f"[worker] Loaded existing config from '{abs_path}' (last modified {mtime}). "
          f"An existing config is never auto-overwritten with new defaults -- delete/replace "
          f"it (or edit it directly) to pick up new script defaults.")
    return merged


CONFIG = load_or_create_config(CONFIG_PATH, DEFAULT_CONFIG)

# If the config overrides model_cache_dir, keep HF's cache inside it too --
# otherwise we're back to two caches on (possibly) two different volumes.
if CONFIG["model_cache_dir"] != _DEFAULT_MODEL_CACHE_DIR:
    _hf_cache = os.path.join(CONFIG["model_cache_dir"], ".hf_cache", "hub")
    os.makedirs(_hf_cache, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = _hf_cache
    import huggingface_hub.constants as _hf_constants
    _hf_constants.HF_HUB_CACHE = _hf_cache
    _hf_constants.HUGGINGFACE_HUB_CACHE = _hf_cache

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
    """Per-GPU memory stats via nvidia-smi, in MB. [] if unavailable."""
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
    """Checks whether the currently-importable llama_cpp build actually has
    CUDA offload compiled in, rather than trusting that installation
    "succeeded". A plain `import llama_cpp` succeeding proves nothing about
    GPU support -- a CPU-only build imports fine too."""
    try:
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as e:
        logger.debug("Could not query llama_cpp GPU offload support: %s", e)
        return False


def ensure_llama_cpp_backend():
    """Installs llama-cpp-python with CUDA support when a GPU is present.

    BUG FIX: the old version stopped as soon as `import llama_cpp` worked,
    even for the "prebuilt CUDA wheel" attempt. But `pip install
    llama-cpp-python --extra-index-url .../whl/cu121` doesn't guarantee a
    CUDA wheel gets used -- if that index doesn't have a matching wheel for
    whatever version pip resolves to, pip silently falls back to building
    the plain PyPI sdist from source with default (CPU-only) cmake flags.
    That's exactly what happened here: the log showed "Building wheel for
    llama-cpp-python ... done" during the supposed "prebuilt wheel" step,
    which should never need to build anything, and the code logged
    "Installed prebuilt CUDA wheel" regardless, purely because import
    succeeded. Net effect: every layer stayed on CPU (0 VRAM used, ~25GB
    system RAM, ~300% CPU during inference) despite n_gpu_layers=-1 asking
    to offload everything.

    Every install path below is now followed by an explicit GPU-offload
    check, and falls through to the next (more forceful) strategy if that
    check fails -- import succeeding is no longer treated as success.
    """
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
            logger.warning("Prebuilt-wheel install completed but GPU offload is NOT available "
                            "(it likely fell back to a CPU build from source); trying an "
                            "explicit CUDA source build instead.")
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
                     "Falling back to CPU-only inference -- this will be slow and will NOT "
                     "use VRAM. Check that the CUDA toolkit/driver matches this environment.")
    except Exception as e:
        logger.error("CUDA source build failed (%s). Falling back to CPU-only llama-cpp-python; "
                     "inference will run on CPU only.", e)

    import llama_cpp  # noqa


ensure_llama_cpp_backend()
from llama_cpp import Llama  # noqa: E402

# --------------------------------------------------------------------------- #
# Step 4: disk cache manager (LRU eviction)
# --------------------------------------------------------------------------- #


class DiskCacheManager:
    """Tracks cached GGUF files and last-used time so the least-recently-used
    model(s) can be evicted before disk fills up. Persists to a JSON file in
    the cache dir so LRU order survives a restart."""

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
        """Drop entries whose backing file (following symlinks) no longer exists."""
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
        # BUG FIX: `path` may be a symlink into the HF blob cache (this is
        # now the normal case, see resolve_local_path). Deleting only the
        # symlink would leave the real multi-GB blob on disk forever, which
        # was the root cause of disk usage silently outrunning what our own
        # eviction accounting believed it had freed. Resolve and remove the
        # real target too.
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
        """Evicts LRU cached models until `required_bytes` + safety margin is
        free, skipping anything in `protect_paths`. Returns False if that
        target still can't be met after evicting everything evictable."""
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
    """Reads GGUF header + vocab only (n_gpu_layers=0, vocab_only=True) to
    get architecture hyperparameters for a KV-cache estimate, without
    touching the GPU. Returns None if probing/parsing fails."""
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
    """Bytes of KV-cache VRAM per token, summed across all layers (K + V).
    Uses head_count_kv so GQA models (Llama-3, Mistral, Qwen2/3, etc.)
    aren't overestimated. kv_dtype_bytes=2 assumes llama.cpp's default f16
    KV cache."""
    n_layer = hyper["n_layer"]
    n_embd = hyper["n_embd"]
    n_head = hyper["n_head"]
    n_head_kv = hyper.get("n_head_kv") or n_head
    head_dim = n_embd / n_head
    kv_embd = head_dim * n_head_kv
    return int(n_layer * kv_embd * 2 * kv_dtype_bytes)


def resolve_desired_n_ctx(configured: Any, hyper: Optional[dict],
                           fallback_max: int = 32768) -> int:
    """Turns 'gpu.n_ctx' / per-model 'n_ctx' into a concrete integer target.

    "max" means: aim for the model's own trained context length (from its
    GGUF header). It is then ALSO capped by vram_management.max_auto_n_ctx
    (applied by the caller) -- auto-maximizing to a model's full native
    window (which can be 1M+ tokens) is not something real hardware can
    serve safely, regardless of what the raw VRAM-fit math says, so "max"
    is bounded rather than unlimited. An explicit integer in config is
    always honored as-is."""
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
    """Largest context (up to desired_n_ctx) that fits currently-free VRAM
    while preserving the configured headroom. Budgets inference-only VRAM:
    model weights (or the offloaded fraction) + KV-cache + a compute-buffer
    estimate. Falls back to desired_n_ctx unchanged if there's no GPU."""
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

    # BUG FIX: the old flat overhead (a fixed few hundred MB, independent of
    # context size) badly underestimated llama.cpp's compute/scratch
    # buffers (attention scratch, batch buffers, etc.), which DO grow with
    # n_ctx. That let this function "approve" a 215k-token context that fit
    # the naive weights+KV math but not the real allocation, crashing the
    # notebook. Add a per-token compute term on top of the flat batch
    # overhead so large contexts are sized more conservatively.
    flat_overhead = max(512 * 1024 ** 2, n_batch * 1024 * 1024)
    per_token_compute_overhead = 2 * 1024  # ~2KB/token, conservative estimate
    available_for_kv = usable_free - weight_bytes - flat_overhead

    if available_for_kv <= 0:
        max_ctx_by_vram = 0
    else:
        max_ctx_by_vram = int(available_for_kv // max(kv_per_token + per_token_compute_overhead, 1))

    final_ctx = min(desired_n_ctx, n_ctx_cap or desired_n_ctx)

    # BUG FIX: apply the configured hard ceiling for auto-"max" sizing.
    # Fixing the VRAM math alone isn't enough -- host RAM, load time, and
    # per-request latency all scale with n_ctx too, so "max" should never
    # try to reach a model's full native window unattended.
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
        self._load_lock = threading.Lock()

    def registry(self) -> Dict[str, Any]:
        return self.config.get("models", {})

    def _protected_paths(self) -> Set[str]:
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
            raise ValueError(f"Model alias '{alias}' is not in the model registry. "
                              f"Add it to inference_server_config.json under 'models'.")
        if "path" in entry:
            path = entry["path"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Configured local model path does not exist: {path}")
            return path

        repo = entry["repo"]
        filename = entry["file"]
        cache_key = f"{repo}/{filename}"
        token = self.config.get("huggingface_token") or None

        # BUG FIX: no more `local_dir=...`. Using cache_dir here lets
        # huggingface_hub manage a single, symlinked copy of each blob
        # (inside HF_HUB_CACHE, which we've pointed at our own cache_dir),
        # instead of materializing a second full copy per model. This call
        # is also cheap/idempotent when already cached -- it just verifies
        # and returns the existing path without re-downloading -- so we no
        # longer need to hand-roll an "already exists" check against a
        # manually-computed local_dir path.
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
                    ok = self.disk_cache.ensure_space_for(remote_size, protect_paths=self._protected_paths())
                    if not ok:
                        free_gb = self.disk_cache.free_space_bytes(self.cache_dir) / (1024 ** 3)
                        raise RuntimeError(
                            f"Not enough disk space to download '{alias}' "
                            f"({remote_size / (1024 ** 3):.2f} GB needed, only "
                            f"{free_gb:.2f} GB free even after evicting all evictable cached models).")
                else:
                    logger.info("Proceeding without a pre-download space check for '%s' "
                                "(remote size unknown).", alias)

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
            time.sleep(0.5)  # let the driver actually reclaim VRAM

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
                logger.warning("Dual-GPU split failed/did not help (%s); falling back.", e)
        return Llama(**kwargs)

    def ensure_loaded(self, alias: str) -> Llama:
        with self._load_lock:
            if self.current_alias == alias and self.llm is not None:
                return self.llm

            if self.llm is not None:
                logger.info("Switching from '%s' to '%s' -- unloading current model first.",
                            self.current_alias, alias)
                self.unload()

            model_path = self.resolve_local_path(alias)
            self.current_local_path = model_path

            entry = self.registry().get(alias, {})
            gpu_cfg = self.config.get("gpu", {})
            configured_n_ctx = entry.get("n_ctx", gpu_cfg.get("n_ctx", "max"))
            n_gpu_layers = entry.get("n_gpu_layers", gpu_cfg.get("n_gpu_layers", -1)) if GPU_COUNT > 0 else 0
            n_batch = entry.get("n_batch", gpu_cfg.get("n_batch", 512))

            hyper = probe_gguf_hyperparams(model_path)
            desired_n_ctx = resolve_desired_n_ctx(configured_n_ctx, hyper)
            logger.info(
                "Context resolution for '%s': configured n_ctx=%r, GGUF n_ctx_train=%s, "
                "resolved desired_n_ctx=%d (before VRAM/safety shrinking).",
                alias, configured_n_ctx,
                hyper.get("n_ctx_train") if hyper else "unknown", desired_n_ctx)

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
                    logger.warning("CUDA OOM loading '%s' at n_ctx=%d; retrying with n_ctx=%d (%d/%d).",
                                    alias, n_ctx, new_ctx, attempts, max_retries)
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
    """Periodically evicts LRU cached models if free space drops below the
    configured minimum, independent of any in-progress download."""
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
                logger.info("Disk monitor: %.2f GB free < %.2f GB minimum; evicting LRU cached models.",
                            free / (1024 ** 3), min_free_bytes / (1024 ** 3))
                MODELS.disk_cache.ensure_space_for(min_free_bytes, protect_paths=MODELS._protected_paths())
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
    """Non-streaming path: blocking, called via run_in_executor."""
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


def run_inference_streaming(payload: dict, on_token):
    """Token-by-token streaming path (blocking; run in a background thread).
    Calls on_token(delta_text, finish_reason) as tokens are produced."""
    model_alias = payload["model"]
    gen_kwargs = build_gen_kwargs(payload)
    llm = MODELS.ensure_loaded(model_alias)

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
        self.seen_job_ids: set = set()
        self.ws_failures = 0
        self.loop = asyncio.get_event_loop()

    def _sign(self) -> Dict[str, str]:
        ts = str(time.time())
        sig = hmac.new(self.secret.encode(), f"{self.worker_id}:{ts}".encode(),
                        hashlib.sha256).hexdigest()
        return {"worker_id": self.worker_id, "timestamp": ts, "signature": sig}

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
        """Streams tokens to the proxy as they're produced, then sends
        stream_done (or error) at the end."""
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
            logger.warning("Lost connection while streaming job %s: %s", job_id, e)

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

    async def run_long_poll_forever(self, stop_after_seconds: float):
        """Runs long-poll for a while, then returns so the caller can retry
        upgrading back to WebSocket."""
        t_cfg = self.config["transport"]
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
        """WebSocket-first with automatic long-poll fallback and periodic
        re-attempts to upgrade back to WebSocket."""
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
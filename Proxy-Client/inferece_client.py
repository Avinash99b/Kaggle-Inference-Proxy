"""
inference_client.py
================================================================================
Single-model Kaggle notebook inference worker.

Adapted from `run_model_with_proxy.py` for SINGLE-MODEL mode: loads exactly
one GGUF model at startup (selected via TARGET_MODEL_REPO and FILE_NAME env vars), connects
outbound to a proxy (WebSocket first, HTTP long-poll fallback), serves
chat/text completion jobs (including token streaming) and, optionally,
embeddings jobs, reports GPU utilization every 5 seconds via heartbeat, and
shuts down cleanly on KeyboardInterrupt (unloading the model and
deregistering from the proxy).

EMBEDDINGS
-------------------------------------------------------------------------------
Set ENABLE_EMBEDDINGS=1 to load the target model with llama.cpp's
embedding output enabled, which lets this worker also serve "embeddings"
jobs (routed from the proxy's POST /v1/embeddings) against the SAME loaded
model, alongside its normal chat/completion jobs. This is off by default:
it costs a bit of extra VRAM/load time, and if all you want is embeddings,
a dedicated embedding GGUF (nomic-embed-text, bge-*, etc.) run as its own
worker will generally give better embedding quality than repurposing a
chat model's hidden states.

max_tokens / generation length
-------------------------------------------------------------------------------
This worker no longer imposes its own default cap (previously 512) on
chat/completion jobs that don't specify a max_tokens. The proxy's
normalize_generation_params() resolves whichever max-tokens field a
client sent (max_tokens / max_completion_tokens / max_new_tokens /
max_output_tokens) down to a single canonical value, or None if the
client wants unlimited generation -- this worker forwards that straight
through to llama-cpp-python, which itself treats None (or <=0) as
"generate until EOS/stop/context limit" rather than requiring an explicit
cap. Set default_generation.max_tokens in inference_server_config.json if
you want this specific worker to default to a fixed cap when a payload
somehow arrives without a "params" object at all (bypassing the proxy).

There is no model-switching in this build: once the target model is loaded it
stays loaded for the lifetime of the process. If you need a different
model, restart the notebook cell with a different TARGET_MODEL_REPO/FILE_NAME pair.

CANCELLATION / HARD-KILL
-------------------------------------------------------------------------------
Each inference call (chat/completion, streaming or not) runs in a dedicated
child process (see inference_runner_proc.py), not in-process. This is what
makes cancellation actually work: llama.cpp's generation call is a blocking
native call with no cooperative-cancel hook, so the only reliable way to
stop a job that the client no longer wants (disconnected, or the proxy's
request_timeout_seconds elapsed) is to SIGKILL the process running it. The
proxy signals this by sending a `{"type": "cancel", "job_id": ...}` message
over the worker's WebSocket connection the moment the client disconnects or
the job times out server-side; this worker kills the matching subprocess
immediately, frees the GPU context, reports a cancellation error for that
job, and is ready for the next job right away -- instead of sitting there
for however long the orphaned generation would otherwise have taken.

Run this in a Kaggle notebook (ideally with the "GPU T4 x2" accelerator
enabled):

    export TARGET_MODEL_REPO="Qwen/Qwen3-8B-GGUF"
    export FILE_NAME="Qwen3-8B-Q8_0.gguf"
    export PROXY_URL="https://kaggle-inference-proxy.onrender.com"
    export WORKER_ID="kaggle-account-1"
    export WORKER_SECRET="shared-secret-key"
    export ENABLE_EMBEDDINGS="0"   # set to "1" to also serve /v1/embeddings
    python inference_client.py
"""
from __future__ import annotations

import asyncio
import concurrent.futures
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

print("Version 1 to see if kaggle cache is updated")
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
# Single-model mode reads its target repo/file directly from environment
# variables (set by the orchestrator/dashboard). Everything else
# (VRAM/disk/transport tuning) still comes from a local JSON config file,
# same as the reference script, so operators can tune it without touching
# code.
# --------------------------------------------------------------------------- #

CONFIG_PATH = os.environ.get("WORKER_CONFIG_PATH", "inference_server_config.json")

DEFAULT_CONFIG = {
    "model_cache_dir": _DEFAULT_MODEL_CACHE_DIR,
    "default_generation": {"temperature": 0.8, "top_p": 0.95, "max_tokens": None},  # None = unlimited
    "transport": {
        "prefer_websocket": True,
        "ws_path": "/worker/ws",
        "heartbeat_interval_seconds": 5,
        # websockets' own low-level ping/pong keepalive (separate from the
        # app-level "heartbeat" message sent every heartbeat_interval_seconds
        # above). Bumped up from the library default of 20/20: on Kaggle's
        # outbound network path, brief jitter/latency spikes under sustained
        # GPU load can delay a single pong past 20s even with no blocking
        # bug in this process, which was causing false-positive "keepalive
        # ping timeout" disconnects mid-stream. A larger window trades a
        # slightly slower detection of a genuinely dead connection for
        # tolerance of that jitter.
        "ws_ping_interval_seconds": 30,
        "ws_ping_timeout_seconds": 45,
        "reconnect_initial_backoff_seconds": 2,
        "reconnect_max_backoff_seconds": 60,
        "ws_failure_threshold_before_poll_fallback": 3,
        "poll_interval_seconds": 2,
        "long_poll_wait_seconds": 40,
        # Belt-and-suspenders local watchdog for a single streaming job: if
        # no token (and no done/error) arrives from the runner subprocess
        # for this long, the job is treated as stalled and force-cancelled
        # locally rather than waiting indefinitely for the proxy's own
        # request_timeout_seconds cancel to arrive. This matters because,
        # after the run_websocket_forever fix below, an in-flight job no
        # longer blocks the WS receive loop -- but without this timeout a
        # single wedged job could still sit forever waiting on its own
        # queue if a cancel frame were somehow lost.
        "stream_token_stall_timeout_seconds": 120,
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
    "cancellation": {
        # How long to wait for a subprocess to exit gracefully (SIGTERM)
        # before escalating to SIGKILL when a job is cancelled.
        "graceful_term_timeout_seconds": 2.0,
        # See PersistentRunner.job_lock_wait_timeout_seconds for why this
        # exists: a bound on how long a new job will wait to acquire the
        # runner's job_lock before treating whoever's holding it as wedged
        # and forcing an unconditional reset.
        "job_lock_wait_timeout_seconds": 30,
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
TARGET_MODEL_REPO = os.environ.get("TARGET_MODEL_REPO")
FILE_NAME = os.environ.get("FILE_NAME")
PROXY_URL = os.environ.get("PROXY_URL", "http://0.0.0.0:8000")
WORKER_ID = os.environ.get("WORKER_ID", "kaggle-worker-1")
WORKER_SECRET = os.environ.get("WORKER_SECRET", "change-me-worker-secret")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
WORKER_LOG_LEVEL = os.environ.get("WORKER_LOG_LEVEL", CONFIG["logging"].get("level", "INFO"))
# When set truthy, the target model is loaded with embedding=True (llama.cpp
# needs this at load time to build an embedding-capable compute graph), which
# enables serving "embeddings"-kind jobs against it in addition to normal
# chat/completion jobs. Off by default: enabling it on a large generation
# model wastes a bit of VRAM/load time for a capability most deployments
# won't use, and dedicated embedding GGUFs (e.g. nomic-embed-text, bge-*)
# give meaningfully better embedding quality than a chat model's hidden
# states repurposed as embeddings anyway.
ENABLE_EMBEDDINGS = os.environ.get("ENABLE_EMBEDDINGS", "").strip().lower() in ("1", "true", "yes")

if not TARGET_MODEL_REPO or not FILE_NAME:
    print("[worker] FATAL: TARGET_MODEL_REPO and FILE_NAME environment variables are required.")
    sys.exit(1)

TARGET_MODEL_ID = f"{TARGET_MODEL_REPO}/{FILE_NAME}"

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

logging.basicConfig(
    level=getattr(logging, CONFIG["logging"].get("level", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(CONFIG["logging"].get("file", "/tmp/inference_server.log"))],
)
logger = logging.getLogger("worker")

os.makedirs(CONFIG["model_cache_dir"], exist_ok=True)

logger.info("Starting inference server worker '%s' -> proxy %s (target model: %s)",
            WORKER_ID, PROXY_URL, TARGET_MODEL_ID)

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
# (and old, no-longer-needed cached blobs from a previous target model run
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
# resolves + downloads + probe-loads the target model exactly once, to
# determine the final n_ctx/n_gpu_layers/tensor_split settings. Actual
# inference then runs on a single persistent CHILD PROCESS (see
# PersistentRunner below) that loads the model once and serves jobs one
# at a time, so a stuck/orphaned job can be hard-killed (and the process
# respawned) without reloading the model on every normal request.
# --------------------------------------------------------------------------- #
def _debug(msg: str):
    print(msg)

class ModelManager:
    def __init__(self, config: dict, target_repo: str, file_name: str):
        self.config = config
        self.target_repo = target_repo
        self.file_name = file_name
        self.target_model_id = f"{target_repo}/{file_name}"
        self.cache_dir = config["model_cache_dir"]
        self.current_local_path: Optional[str] = None
        self.status: str = "loading_model"  # "loading_model" | "idle" | "busy"
        self.disk_cache = DiskCacheManager(self.cache_dir, config)
        self.vram = VramManager(config)
        self._load_lock = threading.Lock()

        # Resolved load parameters, computed once in load_startup_model()
        # and then reused whenever PersistentRunner (re)spawns its
        # subprocess. n_ctx is also what gets reported on the dashboard
        # via heartbeat.
        self.n_ctx: Optional[int] = None
        self.n_gpu_layers: Optional[int] = None
        self.n_batch: Optional[int] = None
        self.n_threads: Optional[int] = None
        self.tensor_split: Optional[List[float]] = None

    def _remote_file_size(self, repo: str, filename: str, token: Optional[str]) -> Optional[int]:
        try:
            url = hf_hub_url(repo_id=repo, filename=filename)
            meta = get_hf_file_metadata(url, token=token)
            return getattr(meta, "size", None)
        except Exception as e:
            logger.warning("Could not fetch remote size for %s/%s: %s", repo, filename, e)
            return None

    def _resolve_local_path(self) -> str:
        repo = self.target_repo
        filename = self.file_name
        cache_key = f"{repo}/{filename}"
        token = self.config.get("huggingface_token") or None

        dl_cfg = self.config.get("download", {})
        max_retries = dl_cfg.get("max_retries", 4)
        backoff = dl_cfg.get("retry_backoff_seconds", 5)

        already_cached = cache_key in self.disk_cache.entries and             os.path.exists(self.disk_cache.entries[cache_key]["path"])
        if not already_cached:
            disk_cfg = self.config.get("disk_management", {})
            if disk_cfg.get("enabled", True):
                remote_size = self._remote_file_size(repo, filename, token)
                if remote_size:
                    ok = self.disk_cache.ensure_space_for(remote_size)
                    if not ok:
                        free_gb = self.disk_cache.free_space_bytes(self.cache_dir) / (1024 ** 3)
                        raise RuntimeError(
                            f"Not enough disk space to download '{self.target_model_id}' "
                            f"({remote_size / (1024 ** 3):.2f} GB needed, only "
                            f"{free_gb:.2f} GB free even after evicting all evictable cached models).")
                else:
                    logger.info("Proceeding without a pre-download space check for '%s' "
                                "(remote size unknown).", self.target_model_id)

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
        raise RuntimeError(f"Failed to download model '{self.target_model_id}' after {max_retries} attempts: {last_err}")

    def _try_load_probe(self, n_ctx: int, n_gpu_layers: int, n_batch: int,
                         n_threads: int, attempt_dual: bool) -> None:
        """Loads the model in-process JUST to validate that these settings
        actually work (VRAM fits, no OOM), then immediately releases it.
        The real inference calls happen in a persistent child process via
        PersistentRunner,
        which re-load the model fresh every job -- this probe only exists
        to run the existing OOM-shrink retry loop once at startup instead
        of duplicating that logic in every child process."""
        kwargs = dict(
            model_path=self.current_local_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            # Match whatever the real per-job runner subprocess will load
            # with (see PersistentRunner.start()'s load_req) -- embedding
            # output changes the compute graph enough that a probe done
            # without this flag isn't a reliable predictor of whether the
            # real load will fit in VRAM.
            embedding=ENABLE_EMBEDDINGS,
            verbose=False,
        )
        llm = None
        try:
            if attempt_dual:
                try:
                    split = [1.0 / GPU_COUNT] * GPU_COUNT
                    llm = Llama(tensor_split=split, **kwargs)
                    _debug("=" * 80)
                    _debug("LLAMA CONFIG")
                    _debug("chat_format:", getattr(llm, "chat_format", None))
                    _debug("chat_handler:", type(getattr(llm, "chat_handler", None)).__name__)
                    _debug("=" * 80)
                    self.tensor_split = split
                except Exception as e:
                    if is_cuda_oom_error(e):
                        raise
                    logger.warning("Dual-GPU split failed/did not help (%s); falling back.", e)
                    llm = Llama(**kwargs)
                    self.tensor_split = None
            else:
                llm = Llama(**kwargs)
                self.tensor_split = None
        finally:
            if llm is not None:
                del llm
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    def load_startup_model(self) -> None:
        """
        Downloads (if needed) the target model and PROBES that it loads
        successfully with some (n_ctx, n_gpu_layers, ...) configuration,
        handling VRAM constraints, OOM errors, and gracefully degrading
        context size along the way. The resolved settings are stored on
        self (n_ctx, n_gpu_layers, n_batch, n_threads, tensor_split) and
        reused whenever PersistentRunner (re)spawns its subprocess -- there
        is no persistent in-process Llama object anymore; instead a single
        long-lived, killable child process handles all inference calls.
        """
        with self._load_lock:
            self.status = "loading_model"
            model_path = self._resolve_local_path()
            self.current_local_path = model_path

            gpu_cfg = self.config.get("gpu", {})
            configured_n_ctx = gpu_cfg.get("n_ctx", "max")
            n_gpu_layers = gpu_cfg.get("n_gpu_layers", -1) if GPU_COUNT > 0 else 0
            n_batch = gpu_cfg.get("n_batch", 512)
            n_threads = gpu_cfg.get("n_threads", 0) or os.cpu_count() or 4

            # Probe GGUF metadata
            hyper = probe_gguf_hyperparams(model_path)
            desired_n_ctx = resolve_desired_n_ctx(configured_n_ctx, hyper)

            logger.info(
                "Context resolution for '%s': configured n_ctx=%r, GGUF n_ctx_train=%s, "
                "resolved desired_n_ctx=%d (before VRAM/safety shrinking).",
                self.target_model_id, configured_n_ctx,
                hyper.get("n_ctx_train") if hyper else "unknown", desired_n_ctx)
            if hyper:
                logger.info("GGUF header: n_ctx_train=%s, n_layer=%s, n_embd=%s",
                            hyper.get("n_ctx_train"), hyper.get("n_layer"), hyper.get("n_embd"))

            # Compute initial context size with VRAM constraints
            n_ctx = desired_n_ctx
            if GPU_COUNT > 0:
                n_ctx = compute_dynamic_n_ctx(model_path, desired_n_ctx, n_gpu_layers,
                                            n_batch, self.vram, self.config, hyper=hyper)

            # GPU split configuration
            attempt_dual = GPU_COUNT >= 2 and gpu_cfg.get("attempt_dual_gpu_split", True)

            # VRAM management: be conservative on small GPUs
            vram_cfg = self.config.get("vram_management", {})
            total_vram_mb = self.vram.usable_free_bytes()/1024/1024

            # Adjust shrink factor and min context based on available VRAM
            if total_vram_mb < 4096:  # Less than 4GB
                shrink_factor = vram_cfg.get("oom_shrink_factor", 0.6)  # More aggressive
                min_ctx = vram_cfg.get("min_n_ctx", 128)  # Lower floor
            else:
                shrink_factor = vram_cfg.get("oom_shrink_factor", 0.75)
                min_ctx = vram_cfg.get("min_n_ctx", 256)

            max_retries = vram_cfg.get("max_oom_retries", 6)  # More retries for safety

            # **Preemptive sizing**: If we detect we're pushing it, start lower
            if GPU_COUNT > 0 and n_ctx > 8192 and total_vram_mb < 6000:
                preemptive_n_ctx = max(min_ctx, int(n_ctx * 0.5))
                logger.warning(
                    "Preemptive context reduction: starting with n_ctx=%d (was %d) "
                    "due to low VRAM (%.1f GB available). Attempting dual_gpu=%s",
                    preemptive_n_ctx, n_ctx, total_vram_mb / 1024, attempt_dual)
                n_ctx = preemptive_n_ctx

            attempts = 0
            last_exception = None

            while True:
                attempts += 1
                try:
                    logger.info("Probe-loading '%s' (attempt %d/%d, n_ctx=%d, n_gpu_layers=%s)...",
                                self.target_model_id, attempts, max_retries + 1,
                                n_ctx, n_gpu_layers)

                    self._try_load_probe(n_ctx, n_gpu_layers, n_batch, n_threads, attempt_dual)

                    # Success! Store resolved settings for per-job runners.
                    self.n_ctx = n_ctx
                    self.n_gpu_layers = n_gpu_layers
                    self.n_batch = n_batch
                    self.n_threads = n_threads
                    self.status = "idle"
                    gpu_stats = aggregate_gpu_stats()
                    logger.info("✓ Verified '%s' loads with n_ctx=%d. GPU: %.1f/%.1f GB used "
                                "(probe released; per-job runner processes will reload as needed).",
                                self.target_model_id, n_ctx,
                                gpu_stats["used_mb"] / 1024, gpu_stats["total_mb"] / 1024)
                    return

                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    is_oom = is_cuda_oom_error(e) or "context" in error_str or "memory" in error_str

                    logger.error("Load attempt %d/%d failed: %s", attempts, max_retries + 1, type(e).__name__)

                    # **Condition 1: Max retries exceeded**
                    if attempts >= max_retries + 1:
                        logger.error("✗ Exceeded max retries (%d). Giving up.", max_retries + 1)
                        self.status = "failed"
                        raise RuntimeError(
                            f"Failed to load model after {max_retries + 1} attempts. "
                            f"Last error: {e}") from e

                    # **Condition 2: Non-OOM error (can't recover)**
                    if not is_oom:
                        logger.error("✗ Non-recoverable error: %s", e)
                        self.status = "failed"
                        raise RuntimeError(f"Failed to load model (non-recoverable): {e}") from e

                    # **Condition 3: Already at minimum context**
                    if n_ctx <= min_ctx:
                        logger.error("✗ Reached minimum context (n_ctx=%d). Cannot shrink further.", min_ctx)
                        self.status = "failed"
                        raise RuntimeError(
                            f"OOM even at minimum context (n_ctx={min_ctx}). "
                            f"GPU may be insufficient for this model. "
                            f"Try: reduce n_gpu_layers, lower min_n_ctx, or use a smaller model.") from e

                    # **Recoverable OOM: Shrink and retry**
                    new_ctx = max(min_ctx, int(n_ctx * shrink_factor))
                    logger.warning(
                        "OOM at n_ctx=%d. Shrinking to n_ctx=%d (%.0f%%) and retrying (%d/%d)...",
                        n_ctx, new_ctx, (new_ctx / n_ctx) * 100, attempts, max_retries + 1)

                    # Clear GPU cache aggressively
                    self._clear_gpu_memory()
                    time.sleep(0.5)

                    n_ctx = new_ctx

    @staticmethod
    def _clear_gpu_memory():
        """Clear GPU memory caches across different libraries."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("Cleared PyTorch GPU cache")
        except Exception as e:
            logger.debug("PyTorch cache clear failed: %s", e)

        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            logger.debug("Cleared CuPy GPU cache")
        except Exception as e:
            logger.debug("CuPy cache clear failed: %s", e)

    def unload(self):
        """Called once, on shutdown. No persistent in-process model to
        release (the persistent runner subprocess cleans up its own GPU
        memory on exit), but we still clear whatever cache we can from
        this process."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


MODELS = ModelManager(CONFIG, TARGET_MODEL_REPO, FILE_NAME)

# Path to the child-process entrypoint script, expected alongside this file.
_RUNNER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_runner_proc.py")


class PersistentRunner:
    """Owns ONE long-lived subprocess running inference_runner_proc.py, which
    loads the model once and then serves jobs one at a time in a loop. This
    is what avoids paying a multi-GB model reload on every single request:
    the same warm process is reused across jobs, and is only killed +
    respawned when a job actually needs to be force-cancelled (client
    disconnected / proxy-side request_timeout_seconds elapsed) -- the NEXT
    job after that pays one reload, but every other job doesn't.

    Only one job may run at a time (job_lock), matching the underlying
    single GGUF execution context -- this mirrors the previous in-process
    behavior, just routed through a killable subprocess instead of a
    killable-in-name-only blocking call.
    """

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self._current_job_id: Optional[str] = None
        self._kill_flag = False
        self._state_lock = threading.Lock()   # guards proc/_current_job_id/_kill_flag/_cancelled_job_ids
        self.job_lock = threading.Lock()       # serializes job execution (one at a time)
        # job_ids that have been cancelled but were not (yet, or ever)
        # the currently-running job when the cancel arrived -- e.g. a job
        # still waiting for job_lock because a previous job hasn't
        # finished tearing down yet. Without this, a cancel that arrives
        # slightly early would be silently dropped ("not currently running
        # here") and the job would go on to run to completion anyway once
        # the lock freed up. Checked at the top of run_job(), both before
        # and after acquiring job_lock.
        self._cancelled_job_ids: set = set()
        # How long run_job() will wait to acquire job_lock before treating
        # the holder as wedged. In normal operation this lock is essentially
        # uncontended (the proxy only ever has one job in flight per worker
        # at a time), so the only time a second run_job() call has to wait
        # here at all is the brief handoff window after a job is cancelled
        # (graceful SIGTERM, then SIGKILL -- single-digit seconds). A wait
        # this long is only ever hit if that teardown itself is stuck (e.g.
        # a genuinely hung subprocess that doesn't die even to SIGKILL),
        # which is exactly the case force_reset() below exists to break out
        # of -- otherwise every job after that point queues up behind the
        # lock forever and the worker reports "busy" indefinitely with no
        # way for a job_id-keyed cancel to ever reach it.
        self.job_lock_wait_timeout_seconds = CONFIG.get("cancellation", {}).get(
            "job_lock_wait_timeout_seconds", 30)

    def start(self) -> None:
        """(Re)spawns the subprocess and blocks until it reports the model
        loaded (or failed to load). Raises RuntimeError on load failure."""
        with self._state_lock:
            self.proc = subprocess.Popen(
                [sys.executable, _RUNNER_SCRIPT_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._kill_flag = False
            self._current_job_id = None
            proc = self.proc

        vram_cfg = CONFIG.get("vram_management", {})
        load_req = {
            "type": "load",
            "model_path": MODELS.current_local_path,
            "n_ctx": MODELS.n_ctx,
            "n_batch": MODELS.n_batch,
            "n_threads": MODELS.n_threads,
            "n_gpu_layers": MODELS.n_gpu_layers,
            "tensor_split": MODELS.tensor_split,
            # Mirrors ENABLE_EMBEDDINGS -- must be set at load time (llama.cpp
            # needs a different compute graph for embedding output), so
            # every (re)spawn of the runner subprocess needs to pass it, not
            # just the initial one.
            "embedding": ENABLE_EMBEDDINGS,
            # The startup probe's n_ctx can go stale by the time a per-job
            # runner actually allocates its own llama_context (VRAM
            # fragmentation, other processes claiming memory between probe
            # time and now). Let the runner itself retry at a smaller
            # n_ctx on load failure instead of treating the parent's
            # single cached value as gospel.
            "min_ctx": vram_cfg.get("min_n_ctx", 256),
            "oom_shrink_factor": vram_cfg.get("oom_shrink_factor", 0.75),
            "max_load_retries": vram_cfg.get("max_oom_retries", 4),
        }
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(load_req) + "\n")
        proc.stdin.flush()

        line = proc.stdout.readline()
        if not line:
            stderr_tail = proc.stderr.read()[-2000:] if proc.stderr else ""
            raise RuntimeError(f"Runner subprocess exited during model load. stderr: {stderr_tail}")
        event = json.loads(line)
        if event.get("type") == "load_error":
            raise RuntimeError(f"Runner subprocess failed to load model: {event.get('error')}")
        if event.get("type") != "loaded":
            raise RuntimeError(f"Runner subprocess sent unexpected startup event: {event}")
        actual_n_ctx = event.get("n_ctx")
        if actual_n_ctx and actual_n_ctx != MODELS.n_ctx:
            logger.warning(
                "Runner subprocess loaded with n_ctx=%d after shrinking from the cached "
                "n_ctx=%d; adopting the smaller value for future respawns.",
                actual_n_ctx, MODELS.n_ctx)
            MODELS.n_ctx = actual_n_ctx
        logger.info("Persistent runner subprocess loaded and ready (pid=%s).", proc.pid)

    def _teardown(self, reason: str) -> None:
        """Hard-kill the current subprocess (SIGTERM then SIGKILL). Safe to
        call even if it's already dead. Does NOT touch job_lock -- callers
        that need to coordinate with an in-progress run_job() call do that
        separately (see force_reset(), which is safe to call even while
        another thread holds job_lock, and cancel_current_job())."""
        with self._state_lock:
            proc = self.proc
            self.proc = None
        if proc is None:
            return
        logger.warning("Killing runner subprocess (%s): pid=%s", reason, proc.pid)
        graceful_timeout = CONFIG.get("cancellation", {}).get("graceful_term_timeout_seconds", 2.0)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=graceful_timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Runner subprocess did not exit after SIGTERM; sending SIGKILL.")
                proc.kill()
                proc.wait(timeout=5)
        except Exception as e:
            logger.warning("Error while killing runner subprocess: %s", e)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def cancel_current_job(self, job_id: str) -> bool:
        """Called from a different thread than the one running run_job().
        Kills the subprocess right away if it's currently working on
        job_id. If job_id isn't the current job -- e.g. it's still queued
        behind job_lock waiting for an earlier job to finish tearing down
        -- it's recorded in _cancelled_job_ids instead, so run_job() bails
        out for it the moment it's about to start rather than letting a
        job we were explicitly told to abandon run to completion anyway.
        Returns True if either action was taken (subprocess killed now, or
        the job was marked so it never runs)."""
        with self._state_lock:
            self._cancelled_job_ids.add(job_id)
            if len(self._cancelled_job_ids) > 10000:
                self._cancelled_job_ids = set(list(self._cancelled_job_ids)[-2000:])
            is_current = self._current_job_id == job_id and self.proc is not None
            if is_current:
                self._kill_flag = True
        if is_current:
            self._teardown(reason=f"cancelled job {job_id}")
        return True

    def force_reset(self, reason: str) -> None:
        """Unconditionally kills whatever subprocess exists and clears all
        per-job bookkeeping, regardless of which job_id (if any) currently
        "owns" it. This is the escape hatch for the two situations where
        the normal job_id-matching cancel path can't be trusted to reach
        whatever is actually stuck:

          1. /kill-all -- an explicit operator-issued reset.
          2. run_job()'s own job_lock-wait timeout below, when a previous
             job's teardown appears to be wedged (subprocess not dying
             even to SIGKILL/wait()) and every job after it has been
             silently queuing up behind job_lock with no way for a
             job_id-keyed cancel to ever reach it.

        Deliberately does not acquire job_lock -- it must be safe to call
        while another thread is blocked holding it (that's the whole
        point). Killing self.proc from here still reaches the same OS
        process that thread's run_job() call is reading from, so its
        `for raw_line in proc.stdout:` loop observes EOF and that thread
        exits its own with-job_lock block on its own shortly after."""
        with self._state_lock:
            self._kill_flag = True
            self._current_job_id = None
            self._cancelled_job_ids.clear()
        self._teardown(reason=reason)

    def run_job(self, job_id: str, job_dict: dict, on_token=None) -> dict:
        """Runs exactly one job on the persistent subprocess, blocking until
        it completes, errors, or is cancelled. If the subprocess isn't
        currently alive (first call, or a previous cancel killed it), it is
        (re)spawned first -- this is the only point where job latency can
        include a model-load cost, and only for the job immediately
        following a cancellation.

        Returns {"result": {...}} or {"error": "..."}. Streaming jobs also
        invoke on_token(delta, finish_reason) as tokens arrive.
        """
        # Fast path: don't even wait for job_lock if this job was already
        # cancelled before its turn came up.
        with self._state_lock:
            if job_id in self._cancelled_job_ids:
                self._cancelled_job_ids.discard(job_id)
                return {"error": "cancelled: job was cancelled before it started running here"}

        acquired = self.job_lock.acquire(timeout=self.job_lock_wait_timeout_seconds)
        if not acquired:
            # Whoever's holding job_lock has held it far longer than any
            # legitimate handoff between jobs should ever take (see the
            # comment on job_lock_wait_timeout_seconds in __init__). Force
            # an unconditional reset to try to break whatever's wedged,
            # then fail this job outright rather than joining the queue
            # behind a lock that may never free.
            logger.error(
                "Timed out after %ss waiting for job_lock to run job %s; "
                "forcing a reset.", self.job_lock_wait_timeout_seconds, job_id)
            self.force_reset(reason=f"job_lock acquisition timed out (blocked job {job_id})")
            return {"error": "inference_error: runner was stuck and has been force-reset; please retry"}

        try:
            with self._state_lock:
                if job_id in self._cancelled_job_ids:
                    self._cancelled_job_ids.discard(job_id)
                    return {"error": "cancelled: job was cancelled before it started running here"}
                need_start = self.proc is None
            if need_start:
                try:
                    self.start()
                except Exception as e:
                    return {"error": f"model_load_failed: {e}"}

            with self._state_lock:
                # Re-check immediately before we commit to running this
                # job: a cancel for job_id may have arrived after the
                # first check above (line ~1201, before self.start()) but
                # before this point -- e.g. while self.start() was
                # spawning/loading the subprocess, or simply in the
                # scheduling gap between acquiring job_lock and getting
                # here. Until _current_job_id is actually set to job_id,
                # cancel_current_job() has no live subprocess to kill for
                # this job and can only record it in _cancelled_job_ids;
                # without this second check that recorded cancellation
                # would never be consulted again and the job would be
                # dispatched anyway. This closes that window.
                if job_id in self._cancelled_job_ids:
                    self._cancelled_job_ids.discard(job_id)
                    return {"error": "cancelled: job was cancelled before it started running here"}
                self._current_job_id = job_id
                self._kill_flag = False
                proc = self.proc

            if proc is None:
                return {"error": "model_load_failed: runner process unavailable"}

            try:
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(json.dumps({"type": "job", "job_id": job_id, "job": job_dict}) + "\n")
                proc.stdin.flush()
            except Exception as e:
                with self._state_lock:
                    was_killed = self._kill_flag
                if was_killed:
                    return {"error": "cancelled: job was cancelled (client disconnected or request timed out)"}
                return {"error": f"inference_error: failed to send job to runner: {e}"}

            result: Optional[dict] = None
            error: Optional[str] = None
            try:
                for raw_line in proc.stdout:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    event = json.loads(raw_line)
                    etype = event.get("type")
                    if etype == "token" and on_token is not None:
                        on_token(event.get("delta", "") or "", event.get("finish_reason"))
                    elif etype == "done":
                        result = event.get("result", {})
                        break
                    elif etype == "error":
                        error = event.get("error")
                        break
                else:
                    # EOF without a done/error event -- either we killed it
                    # (cancellation) or it crashed unexpectedly.
                    pass
            except Exception as e:
                error = f"lost connection to runner subprocess: {e}"
            finally:
                with self._state_lock:
                    self._current_job_id = None
                    was_killed = self._kill_flag

            if was_killed:
                return {"error": "cancelled: job was cancelled (client disconnected or request timed out)"}

            if result is not None:
                return {"result": result}
            if error is not None:
                return {"error": f"inference_error: {error}"}

            # Subprocess died without reporting anything and we didn't kill
            # it ourselves -- treat as a crash, tear down so the next job
            # gets a fresh (working) process instead of reusing a dead pipe.
            logger.error("Runner subprocess for job %s exited unexpectedly with no result.", job_id)
            self._teardown(reason="crashed")
            return {"error": "inference_error: runner process crashed unexpectedly"}
        finally:
            self.job_lock.release()


RUNNER = PersistentRunner()


def cancel_job(job_id: str) -> bool:
    """Entrypoint used by the WS/long-poll message handler when the proxy
    signals that a job's result is no longer wanted (client disconnected,
    or the proxy's request_timeout_seconds elapsed). Hard-kills the
    persistent runner subprocess immediately if it is currently working on
    job_id; otherwise (e.g. job_id is still queued behind job_lock waiting
    for an earlier job to finish tearing down) it's recorded so run_job()
    refuses to start it at all once its turn comes up. The subprocess is
    respawned (paying one model-load) lazily on the next job that actually
    runs."""
    found = RUNNER.cancel_current_job(job_id)
    if found:
        logger.info("Cancelled job %s on request from proxy (subprocess killed "
                     "if it was the one currently running).", job_id)
    return found



async def disk_monitor_loop():
    """Periodically evicts LRU cached models (other than the currently
    loaded one's backing file) if free space drops below the configured
    minimum."""
    disk_cfg = CONFIG.get("disk_management", {})
    if not disk_cfg.get("enabled", True):
        return
    interval = disk_cfg.get("monitor_interval_seconds", 30)
    min_free_bytes = int(disk_cfg.get("min_free_space_gb", 5) * (1024 ** 3))
    loop = asyncio.get_event_loop()

    def _check_and_evict() -> None:
        free = MODELS.disk_cache.free_space_bytes(MODELS.cache_dir)
        if free < min_free_bytes:
            protect = {MODELS.current_local_path} if MODELS.current_local_path else set()
            logger.info("Disk monitor: %.2f GB free < %.2f GB minimum; evicting LRU cached models.",
                        free / (1024 ** 3), min_free_bytes / (1024 ** 3))
            MODELS.disk_cache.ensure_space_for(min_free_bytes, protect_paths=protect)

    while True:
        await asyncio.sleep(interval)
        try:
            # free_space_bytes() alone is normally a cheap syscall, but
            # ensure_space_for() can synchronously os.remove() multi-GB
            # cached model files and rewrite cache metadata under a lock.
            # Both are pushed to the executor so a slow/networked disk
            # can never stall the event loop that also owns the worker's
            # WebSocket connection (a stall here reads as a missed
            # ping/pong and kills the connection with "keepalive ping
            # timeout", exactly like the earlier nvidia-smi issue).
            await loop.run_in_executor(None, _check_and_evict)
        except Exception as e:
            logger.warning("Disk monitor loop error: %s", e)

# --------------------------------------------------------------------------- #
# Step 7: inference execution
#
# Both run_inference() and run_inference_streaming() delegate to the single
# persistent RUNNER (a long-lived subprocess) instead of calling a Llama
# object in-process. This is what enables hard cancellation: cancel_job()
# (defined above, next to PersistentRunner) can kill that subprocess at any
# point during execution, immediately freeing the GPU instead of letting an
# orphaned generation run to completion and block every job queued behind
# it.
# --------------------------------------------------------------------------- #


def build_chat_prompt_messages(messages: list) -> list:
    # `content` is passed through as-is (string OR a list of OpenAI-style
    # content blocks, e.g. [{"type": "text", ...}, {"type": "image_url",
    # "image_url": {"url": "data:image/..."}}]). This function only
    # normalizes role/content presence; it does not need to understand
    # multimodal content itself -- llama-cpp-python's chat handler is
    # responsible for that, and only actually uses the image blocks if the
    # loaded model + chat_format support vision (e.g. llava/qwen-vl style
    # GGUFs with a --clip_model_path/mmproj file). A text-only model will
    # typically just ignore or error on image blocks, same as any other
    # OpenAI-compatible server given a model that doesn't support vision.
    return [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]


def build_gen_kwargs(payload: dict) -> dict:
    params = payload.get("params", {})
    defaults = CONFIG.get("default_generation", {})
    # max_tokens: None here means UNLIMITED, not "use some fallback
    # number". The proxy's normalize_generation_params() already resolved
    # whichever of max_tokens/max_completion_tokens/max_new_tokens/
    # max_output_tokens the client sent down to a single canonical
    # "max_tokens" (or None if the client wants unlimited generation), so
    # this worker just needs to forward it -- it must NOT invent its own
    # default of 512 here, or every "unlimited" request would silently get
    # capped again at the worker. default_generation.max_tokens in this
    # worker's own config file is only consulted as a last resort, for the
    # rare case a payload arrives with no "params" object at all (e.g.
    # driven directly rather than via the proxy); it also defaults to None
    # (unlimited) unless the operator sets it explicitly in
    # inference_server_config.json.
    gen_kwargs = dict(
        temperature=params.get("temperature", defaults.get("temperature", 0.8)),
        top_p=params.get("top_p", defaults.get("top_p", 0.95)),
        max_tokens=params.get("max_tokens", defaults.get("max_tokens")),
        stop=params.get("stop"),
        presence_penalty=params.get("presence_penalty", 0.0),
        frequency_penalty=params.get("frequency_penalty", 0.0),
    )
    if params.get("seed") is not None:
        gen_kwargs["seed"] = params["seed"]
    return gen_kwargs


def _build_runner_job_dict(payload: dict, gen_kwargs: dict) -> dict:
    """Flattens payload + gen_kwargs into the shape inference_runner_proc.py
    expects for its "job" field (see that script's docstring).

    Note: gen_kwargs (temperature/top_p/max_tokens/...) is only meaningful
    for chat/completion jobs -- it's still merged in unconditionally for
    embeddings jobs for simplicity, but inference_runner_proc.py's
    run_embeddings_job() ignores all of it and only reads job["input"].

    Tool-calling fields are forwarded as-is so the runner can pass them into
    llama-cpp-python and, if needed, parse native tool-call tags back into
    OpenAI-style tool_calls.
    """
    job = {
        "kind": payload["kind"],
        "stream": bool(payload.get("stream", False)),
        **gen_kwargs,
    }
    for key in ("tools", "tool_choice", "parallel_tool_calls"):
        if key in payload:
            job[key] = payload[key]
    if payload["kind"] == "chat":
        job["messages"] = build_chat_prompt_messages(payload["messages"])
    elif payload["kind"] == "embeddings":
        job["input"] = payload["input"]
    else:
        job["prompt"] = payload["prompt"]
    return job


def run_inference(payload: dict, job_id: str) -> dict:
    """Non-streaming path: blocking, called via run_in_executor. Runs on the
    persistent runner subprocess, which can be hard-killed by cancel_job()
    if a cancel arrives (client disconnected / job timed out on the proxy)
    before it finishes."""
    requested_model = str(payload.get("model", MODELS.target_model_id))
    allowed_models = {MODELS.target_model_id, MODELS.target_repo, MODELS.file_name}
    if requested_model not in allowed_models:
        return {"error": f"model_not_available: this worker only serves '{MODELS.target_model_id}' "
                          f"(requested '{requested_model}')"}

    if MODELS.current_local_path is None or MODELS.n_ctx is None:
        return {"error": "model_load_failed: model is not loaded"}

    gen_kwargs = build_gen_kwargs(payload)
    job_dict = _build_runner_job_dict(payload, gen_kwargs)

    MODELS.status = "busy"
    start = time.time()
    try:
        outcome = RUNNER.run_job(job_id, job_dict)
        if "error" in outcome:
            if not outcome["error"].startswith("cancelled:"):
                logger.error("Job %s failed: %s", job_id, outcome["error"])
            return outcome

        result = outcome["result"]
        elapsed = time.time() - start
        usage = result.get("usage", {})
        logger.info("Job %s for '%s' completed in %.2fs (completion_tokens=%s).",
                    job_id, requested_model, elapsed, usage.get("completion_tokens"))
        return {"result": result}
    except Exception as e:
        logger.error("Unexpected error running job %s: %s\n%s", job_id, e, traceback.format_exc())
        return {"error": f"inference_error: {e}"}
    finally:
        MODELS.status = "idle"


def run_inference_streaming(payload: dict, job_id: str, on_token):
    """Token-by-token streaming path (blocking; run in a background thread).
    Calls on_token(delta_text, finish_reason) as tokens are produced by the
    persistent runner subprocess. If the job is cancelled mid-stream, that
    subprocess is killed and this function raises so the caller can surface
    an error to the (still-connected, if any) SSE consumer; if nobody is
    listening anymore that's fine too -- the point of killing is to free
    the GPU immediately regardless of who's still watching."""
    if payload.get("kind") == "embeddings":
        # Defensive only -- the proxy never submits embeddings jobs with
        # stream=True (there's no token-by-token output for a single
        # embedding vector call), so this path should be unreachable in
        # practice. Guarding it explicitly turns a would-be confusing
        # KeyError (job_dict has no "prompt"/"messages") into a clear error.
        raise RuntimeError("embeddings jobs do not support streaming")
    requested_model = str(payload.get("model", MODELS.target_model_id))
    allowed_models = {MODELS.target_model_id, MODELS.target_repo, MODELS.file_name}
    if requested_model not in allowed_models:
        raise RuntimeError(f"model_not_available: this worker only serves '{MODELS.target_model_id}'")
    if MODELS.current_local_path is None or MODELS.n_ctx is None:
        raise RuntimeError("model_load_failed: model is not loaded")

    gen_kwargs = build_gen_kwargs(payload)
    job_dict = _build_runner_job_dict(payload, gen_kwargs)

    MODELS.status = "busy"
    start = time.time()
    token_count = 0

    def _on_token(delta: str, finish_reason: Optional[str]) -> None:
        nonlocal token_count
        if delta:
            token_count += 1
        on_token(delta, finish_reason)

    try:
        outcome = RUNNER.run_job(job_id, job_dict, on_token=_on_token)
        if "error" in outcome:
            raise RuntimeError(outcome["error"])

        logger.info("Streaming job %s for '%s' completed in %.2fs (~%d tokens).",
                    job_id, requested_model, time.time() - start, token_count)
    finally:
        MODELS.status = "idle"


# --------------------------------------------------------------------------- #
# Step 8: transport layer -- WebSocket primary, long-poll fallback
# --------------------------------------------------------------------------- #



class ProxyClient:
    def __init__(self, config: dict, target_model_id: str, target_repo: str, file_name: str):
        self.config = config
        self.target_model_id = target_model_id
        self.target_repo = target_repo
        self.file_name = file_name
        self.proxy_url = config["proxy_url"].rstrip("/")
        self.ws_url = self.proxy_url.replace("http://", "ws://").replace("https://", "wss://")             + config["transport"]["ws_path"]
        self.worker_id = config["worker_id"]
        self.secret = config["worker_shared_secret"]
        self.seen_job_ids: set = set()
        self.ws_failures = 0
        self.loop = asyncio.get_event_loop()
        # Dedicated single-thread executor for aggregate_gpu_stats(), kept
        # separate from the default run_in_executor(None, ...) pool used by
        # job execution (run_inference / process_job). Sharing one pool
        # meant a heartbeat's nvidia-smi call could queue up behind (or
        # alongside) job-execution work, delaying heartbeat frames right
        # when a job is busy -- the opposite of what a keepalive is for.
        self._gpu_stats_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gpu-stats")
        # job_id -> asyncio.Task for every job currently being handled on
        # the active WS connection (streaming or not). Tracked so that an
        # incoming "cancel" frame can also cancel the *task* awaiting the
        # result, not just the underlying subprocess -- and so a dropped
        # connection can clean up cleanly instead of leaking task refs
        # across reconnects.
        self.active_job_tasks: Dict[str, asyncio.Task] = {}

    async def _ws_send(self, ws, payload: dict) -> None:
        """The only way any code in this class should write to `ws`. Takes
        self._send_lock so that concurrent writers (the heartbeat task and
        whichever coroutine is handling the current job/stream) can never
        interleave frames on the wire -- see the comment where the lock is
        created in run_websocket_forever for why that matters."""
        async with self._send_lock:
            await ws.send(json.dumps(payload))

    def _sign(self) -> Dict[str, str]:
        ts = str(time.time())
        sig = hmac.new(self.secret.encode(), f"{self.worker_id}:{ts}".encode(),
                        hashlib.sha256).hexdigest()
        return {"worker_id": self.worker_id, "timestamp": ts, "signature": sig}

    def _worker_model_payload(self) -> dict:
        # The proxy routes purely by "is this model id a key in
        # worker.models" -- it doesn't currently read capability flags out
        # of this metadata dict for routing decisions. "supports_embeddings"
        # is included anyway so a dashboard/operator can see at a glance
        # which connected workers were started with ENABLE_EMBEDDINGS,
        # without having to know that out-of-band.
        return {
            "repo": self.target_repo,
            "file": self.file_name,
            "supports_embeddings": ENABLE_EMBEDDINGS,
        }

    async def _heartbeat_payload(self) -> dict:
        # aggregate_gpu_stats() shells out to nvidia-smi via a *blocking*
        # subprocess.run() call (with its own 10s timeout, and real-world
        # latency spikes under GPU load). Calling it directly from a
        # coroutine would stall the single-threaded event loop that also
        # owns this worker's WebSocket connection -- while stalled, the
        # `websockets` library can't process incoming pong frames / service
        # its own ping/pong bookkeeping, so a slow nvidia-smi call during a
        # busy generation can silently blow through ping_timeout and get
        # the connection killed with "keepalive ping timeout", right when
        # a job is in flight. Always run it in the executor instead --
        # specifically the dedicated single-thread self._gpu_stats_executor
        # (not the shared default pool), so this call can never queue up
        # behind job-execution work on the default executor (see the
        # comment where that executor is created in __init__).
        gpu_stats = await self.loop.run_in_executor(self._gpu_stats_executor, aggregate_gpu_stats)
        return {
            "type": "heartbeat",
            "worker_id": self.worker_id,
            "timestamp": str(time.time()),
            "gpu_stats": gpu_stats,
            "current_model": self.target_model_id,
            "status": MODELS.status,
            # Loaded context size, shown on the proxy dashboard. None until
            # load_startup_model() has resolved it.
            "n_ctx": MODELS.n_ctx,
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
        result = await self.loop.run_in_executor(None, run_inference, payload, job_id)
        return {"job_id": job_id, **result}

    async def _run_and_send_job(self, ws, job_id: str, payload: dict) -> None:
        """Wrapper used to run a non-streaming job as its own asyncio task
        (see run_websocket_forever) instead of being awaited inline in the
        WS receive loop. Sends the result itself once process_job resolves.
        Swallows send failures (e.g. the connection dropped while the job
        was running) since there's nothing useful left to do with a result
        nobody can receive -- the proxy's own watchdog/reconnect handling
        takes it from there."""
        try:
            result = await self.process_job(job_id, payload)
            await self._ws_send(ws, {"type": "result", **result})
        except Exception as e:
            logger.warning("Could not deliver result for job %s: %s", job_id, e)

    async def process_stream_job(self, ws, job_id: str, payload: dict) -> None:
        if self._check_and_mark_duplicate(job_id):
            await self._ws_send(ws, {
                "type": "stream_done", "job_id": job_id, "result": {"duplicate": True},
            })
            return

        queue: "asyncio.Queue" = asyncio.Queue()
        loop = self.loop
        stall_timeout = self.config["transport"].get("stream_token_stall_timeout_seconds", 120)

        def on_token(delta: str, finish_reason: Optional[str]) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"delta": delta, "finish_reason": finish_reason})

        def worker_thread() -> None:
            try:
                run_inference_streaming(payload, job_id, on_token)
                loop.call_soon_threadsafe(queue.put_nowait, {"__done__": True})
            except RuntimeError as e:
                if is_cuda_oom_error(e):
                    recover_from_oom(self.target_model_id, e)
                    err = "cuda_out_of_memory: the model ran out of GPU memory for this request"
                elif str(e).startswith("cancelled:"):
                    err = str(e)
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
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=stall_timeout)
                except asyncio.TimeoutError:
                    # No token, done, or error event for stall_timeout
                    # seconds -- the runner subprocess is wedged (or its
                    # pipe is). Don't wait forever on a queue nothing will
                    # ever fill: hard-kill the subprocess ourselves (same
                    # action a proxy-issued "cancel" would trigger) and
                    # report the failure, freeing this connection to serve
                    # the next job instead of sitting stuck indefinitely.
                    logger.warning(
                        "Streaming job %s produced no output for %ss; treating as stalled.",
                        job_id, stall_timeout)
                    await self.loop.run_in_executor(None, cancel_job, job_id)
                    await self._ws_send(ws, {
                        "type": "error", "job_id": job_id,
                        "error": f"inference_error: no output for {stall_timeout}s, job stalled",
                    })
                    return
                if item.get("__done__"):
                    await self._ws_send(ws, {
                        "type": "stream_done", "job_id": job_id,
                        "result": {"usage": {"completion_tokens": completion_tokens}},
                    })
                    return
                if "__error__" in item:
                    await self._ws_send(ws, {
                        "type": "error", "job_id": job_id, "error": item["__error__"],
                    })
                    return
                if item.get("delta"):
                    completion_tokens += 1
                await self._ws_send(ws, {
                    "type": "stream_chunk", "job_id": job_id,
                    "delta": item.get("delta", ""), "finish_reason": item.get("finish_reason"),
                })
        except Exception as e:
            logger.warning("Lost connection while streaming job %s: %s", job_id, e)

    async def run_websocket_forever(self):
        t_cfg = self.config["transport"]
        backoff = t_cfg["reconnect_initial_backoff_seconds"]
        max_backoff = t_cfg["reconnect_max_backoff_seconds"]

        while True:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=t_cfg.get("ws_ping_interval_seconds", 30),
                    ping_timeout=t_cfg.get("ws_ping_timeout_seconds", 45),
                ) as ws:
                    # Guards every ws.send() for the lifetime of this
                    # connection. The heartbeat loop (its own asyncio task)
                    # and this recv loop (which itself calls process_stream_job,
                    # sending one frame per generated token) both write to the
                    # SAME connection concurrently. websockets does not
                    # serialize concurrent send() calls for you -- two
                    # coroutines writing at once can interleave partial
                    # frames on the wire, corrupting the protocol stream.
                    # That corruption is what was actually causing the
                    # intermittent "keepalive ping timeout" disconnects during
                    # streaming (the heartbeat firing mid-stream), not real
                    # network jitter -- so every send must go through this
                    # lock instead of calling ws.send() directly.
                    self._send_lock = asyncio.Lock()

                    hello = {
                        "type": "hello",
                        "models": {self.target_model_id: self._worker_model_payload()},
                        **self._sign(),
                    }
                    await self._ws_send(ws, hello)
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
                            mtype = msg.get("type")
                            if mtype == "job":
                                job_id = msg["job_id"]
                                payload = msg["payload"]
                                await self._ws_send(ws, {"type": "ack", "job_id": job_id})
                                # Dispatched as its own task rather than
                                # awaited here. PersistentRunner.run_job()
                                # still serializes actual GPU work one job
                                # at a time (job_lock), so this doesn't
                                # change execution order -- it only keeps
                                # this receive loop free to keep reading
                                # incoming frames (a "cancel" for this same
                                # job, or anything else) and free for the
                                # `websockets` library to service its own
                                # ping/pong bookkeeping, instead of being
                                # blocked awaiting the job for however long
                                # it takes. That decoupling is what fixes
                                # the "keepalive ping timeout" disconnects:
                                # previously, if a job ever stalled, this
                                # loop could never read the proxy's cancel
                                # frame because it was stuck waiting on the
                                # very job it needed to be told to cancel.
                                if payload.get("stream"):
                                    task = asyncio.create_task(
                                        self.process_stream_job(ws, job_id, payload))
                                else:
                                    task = asyncio.create_task(
                                        self._run_and_send_job(ws, job_id, payload))
                                self.active_job_tasks[job_id] = task
                                task.add_done_callback(
                                    lambda t, jid=job_id: self.active_job_tasks.pop(jid, None))
                            elif mtype == "cancel":
                                # Client disconnected or the job timed out on
                                # the proxy side -- the eventual result is
                                # useless, so hard-kill the subprocess doing
                                # the work right now instead of letting it
                                # run to completion and block the next job.
                                job_id = msg.get("job_id")
                                if job_id:
                                    await self.loop.run_in_executor(None, cancel_job, job_id)
                                    # Also cancel the asyncio task itself, in
                                    # case it's awaiting something other than
                                    # the subprocess pipe (belt-and-suspenders
                                    # alongside cancel_job() above).
                                    task = self.active_job_tasks.get(job_id)
                                    if task is not None and not task.done():
                                        task.cancel()
                            elif mtype == "kill_all":
                                # Emergency reset from the proxy's /kill-all
                                # endpoint: unconditionally hard-kill and
                                # reset the local runner, regardless of
                                # which job_id it currently thinks it owns.
                                # This exists specifically for the case
                                # where the normal job_id-keyed cancel above
                                # has stopped working -- e.g. a wedged
                                # subprocess that never released job_lock,
                                # so later jobs just queue up behind it
                                # forever with no job_id match possible.
                                await self._handle_kill_all()
                            elif mtype == "heartbeat_ack":
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
                payload = await self._heartbeat_payload()
                await self._ws_send(ws, payload)
            except Exception:
                return

    async def run_long_poll_forever(self, stop_after_seconds: float):
        """Runs long-poll for a while (including its own periodic heartbeat
        posts), then returns so the caller can retry upgrading back to WS.

        Long-poll has no push channel for the proxy to deliver an
        out-of-band cancel while a job is running, so on this transport a
        cancel is instead piggybacked onto the heartbeat response (see
        _poll_heartbeat below) and checked between poll cycles. This means
        cancellation latency on long-poll is bounded by the heartbeat
        interval rather than being instant, which is an inherent limitation
        of a pure request/response fallback transport -- WS remains the
        transport to prefer whenever available for exactly this reason.
        """
        t_cfg = self.config["transport"]
        try:
            requests.post(f"{self.proxy_url}/worker/register", json={
                "models": {self.target_model_id: self._worker_model_payload()},
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
                if data.get("kill_all"):
                    await self._handle_kill_all()
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

    async def _handle_kill_all(self) -> None:
        """Shared kill_all handler for both transports: cancels every
        locally tracked job task and unconditionally force-resets the
        runner subprocess. See the "kill_all" WS message handler in
        run_websocket_forever for the full rationale."""
        logger.warning("Received kill_all from proxy: cancelling all local job tasks "
                        "and force-resetting the runner subprocess.")
        for jid, task in list(self.active_job_tasks.items()):
            if not task.done():
                task.cancel()
        await self.loop.run_in_executor(None, RUNNER.force_reset, "kill_all received from proxy")
        self.seen_job_ids.clear()

    async def _poll_heartbeat(self):
        try:
            payload = await self._heartbeat_payload()
            resp = await self.loop.run_in_executor(
                None,
                lambda: requests.post(f"{self.proxy_url}/worker/heartbeat",
                                       json=payload, timeout=15),
            )
            try:
                data = resp.json()
            except Exception:
                return
            cancel_job_id = data.get("cancel_job_id")
            if cancel_job_id:
                cancel_job(cancel_job_id)
            if data.get("kill_all"):
                await self._handle_kill_all()
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


def recover_from_oom(model_alias: str, e: Exception) -> None:
    """Free what CUDA cache we can in THIS (parent) process after a job's
    child process reported an OOM. The child process itself already exited
    (taking its own CUDA context with it), so there's no reload to do here
    -- just surface the error for that one request."""
    logger.error("CUDA OOM while running '%s': %s", model_alias, e)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


async def main():
    # Step 1/2: download + verify-load the single target model BEFORE
    # connecting to the proxy, so the worker never advertises itself as
    # ready until it actually knows a working (n_ctx, n_gpu_layers, ...)
    # configuration for per-job runner processes to use.
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, MODELS.load_startup_model)
    except Exception as e:
        logger.error("FATAL: could not load target model '%s': %s\n%s",
                     f"{TARGET_MODEL_REPO}/{FILE_NAME}", e, traceback.format_exc())
        sys.exit(1)

    client = ProxyClient(CONFIG, MODELS.target_model_id, TARGET_MODEL_REPO, FILE_NAME)
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
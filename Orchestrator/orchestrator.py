"""
orchestrator.py

Multi-account Kaggle deployment orchestrator: manages GPU quotas, deploys
inference servers, tracks deployment status via REST API + WebSocket.

AUTH DESIGN: kaggle==2.2.3 (kagglesdk-backed) authenticates via a single
KGAT_-prefixed token through the KAGGLE_API_TOKEN env var. Rather than
seeding a placeholder at import time, this file never imports `kaggle` at
module load. Each authenticated call sets KAGGLE_API_TOKEN to that
account's real token, then (re)imports the kaggle module tree fresh, so
kagglesdk's eager credential check always sees a valid token and no
account's credentials or cached client state can leak into another
account's call.

NOTEBOOK FORMAT: the generated notebook mirrors the known-working
`inferece.ipynb` two-cell shape exactly:
  1. a config cell that sets env vars,
  2. a single `!python <script>` cell that fetches and runs the worker.
Cell `source` is always stored as a list of lines, each ending in `\n`
(the standard nbformat convention) except the last line -- never as a
single flattened string -- since Kaggle/Jupyter do not insert missing
newlines between list elements themselves.

STATUS SERIALIZATION: kaggle's kernels_status() can return a
KernelWorkerStatus enum (not a plain string) on its `.status` attribute.
That enum is normalized to a plain string the moment it's read out of
the SDK response (see KaggleService.get_notebook_status), and
StateManager.save() also runs everything through a JSON-safe default
encoder as a second line of defense, so a future SDK change that
reintroduces a non-serializable status can never again crash the
deployment_status_loop / crash state persistence.
"""

from __future__ import annotations

import asyncio
import enum
import importlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import uvicorn

###############################################################################
# Constants
###############################################################################

CONFIG_FILE = "orchestrator_config.json"
STATE_FILE = "orchestrator_state.json"
KERNELS_WORKDIR = "/tmp/orchestrator_kernels"

# Valid Kaggle kernel-metadata "machine_shape" values. This is what
# actually selects the GPU type Kaggle attaches to the kernel (T4 x2,
# P100, etc). "enable_gpu": true alone just requests *some* accelerator;
# machine_shape pins the specific one.
VALID_MACHINE_SHAPES = {
    "NvidiaTeslaT4",
    "NvidiaTeslaP100",
    "NvidiaTpuVmV38",
    "None",
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###############################################################################
# JSON helpers
###############################################################################


def _json_safe_default(obj):
    """
    Fallback encoder for json.dump(..., default=_json_safe_default).

    Handles enum members (e.g. kagglesdk's KernelWorkerStatus) by
    reducing to their .value, and falls back to str() for anything else
    that isn't natively serializable, so a status field or similar can
    never again take down state persistence.
    """
    if isinstance(obj, enum.Enum):
        return obj.value
    return str(obj)


def _normalize_status(raw_status) -> str:
    """
    Coerces whatever kernels_status() handed back into a plain string.

    kagglesdk has, in some versions, returned a KernelWorkerStatus enum
    member directly on `.status` instead of a string. str(enum_member)
    would render as "KernelWorkerStatus.RUNNING", which is why we prefer
    `.value` / `.name` first and only fall back to raw str() for
    already-plain values.
    """
    if raw_status is None:
        return "error"
    if isinstance(raw_status, enum.Enum):
        # Prefer the enum's value (usually already the lowercase kaggle
        # status string); fall back to its name if value isn't a string.
        value = raw_status.value
        return value if isinstance(value, str) else raw_status.name
    return str(raw_status)


###############################################################################
# Dataclasses
###############################################################################


@dataclass
class DeploymentState:
    """Represents one deployment running on one Kaggle account."""

    deployment_id: str
    account_id: str
    model_name: str
    model_repo: str
    model_file: str
    notebook_id: str
    notebook_url: str
    notebook_status: str
    worker_id: str
    created_at: float
    started_at: Optional[float] = None
    last_status_check: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    quota_reserved_seconds: int = 0


@dataclass
class AccountState:
    """Represents one Kaggle account with quota information."""

    account_id: str
    username: str
    password: str
    kaggle_username: str
    # KGAT_-prefixed token from kaggle.com/settings/api -> "Generate New
    # Token". Authenticated via KAGGLE_API_TOKEN, set fresh per call.
    kaggle_api_token: str
    gpu_quota_total_seconds: int
    gpu_quota_used_seconds: int
    gpu_quota_remaining_seconds: int
    gpu_quota_refresh_time: str
    last_quota_update: float
    deployment: Optional[DeploymentState] = None


@dataclass
class OrchestratorConfig:
    """Loaded from orchestrator_config.json"""

    proxy_url: str
    proxy_shared_secret: str
    orchestrator_port: int
    accounts: List[dict]
    hf_token: Optional[str]
    notebook_template_url: str
    quota_refresh_interval_minutes: int
    deployment_status_check_interval_seconds: int
    # GPU shape requested on kernel push. Must be one of
    # VALID_MACHINE_SHAPES. Defaults to a T4 (the shape used by the
    # known-working reference config).
    machine_shape: str = "NvidiaTeslaT4"
    enable_tpu: bool = False


###############################################################################
# WebSocket Manager
###############################################################################


class WebSocketManager:
    """Tracks and broadcasts to connected WebSocket clients."""

    def __init__(self):
        self.clients: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.clients)}")

    async def disconnect(self, websocket: WebSocket):
        self.clients.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.clients)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = set()
        for client in self.clients:
            try:
                await client.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.add(client)
        for client in disconnected:
            self.clients.discard(client)


###############################################################################
# Notebook Generator
###############################################################################


def _lines_with_newlines(text: str) -> List[str]:
    """
    Splits `text` into the nbformat-conventional cell `source` shape:
    a list of lines, each ending in '\n' except the final one.

    This is the piece that was previously broken -- cells were built
    from `text.split('\n')`, which drops the newline characters
    entirely. Jupyter/Kaggle do NOT insert newlines between `source`
    list elements when there isn't one already present, so a plain
    `.split('\n')` collapses the whole cell onto one line when
    re-rendered (exactly what showed up in the uploaded
    orchestrated-inference-*.ipynb: "import osTARGET_MODEL_REPO=...").
    """
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


class NotebookGenerator:
    """Generates the two-cell Kaggle inference notebook.

    Mirrors the known-working `inferece.ipynb` shape exactly:
      cell 0: config / env-var setup
      cell 1: `!python <script>` -- fetches the worker script and runs
              it as a shell command in-cell (not via subprocess.Popen),
              matching the reference notebook so behavior in Kaggle
              matches what was already verified locally.
    """

    def __init__(self, worker_script_url: str):
        self.worker_script_url = worker_script_url

    def generate(
        self,
        deployment: "DeploymentState",
        config: "OrchestratorConfig",
    ) -> dict:
        """Generate notebook with placeholders replaced."""

        config_cell_text = (
            "import os\n"
            f"TARGET_MODEL_REPO = \"{deployment.model_repo}\"\n"
            f"FILE_NAME = \"{deployment.model_file}\"\n"
            f"PROXY_URL = \"{config.proxy_url}\"\n"
            f"WORKER_ID = \"{deployment.worker_id}\"\n"
            f"WORKER_SECRET = \"{config.proxy_shared_secret}\"\n"
            f"HF_TOKEN = \"{config.hf_token or ''}\"\n"
            "os.environ.update({\n"
            "    \"TARGET_MODEL_REPO\": TARGET_MODEL_REPO,\n"
            "    \"PROXY_URL\": PROXY_URL,\n"
            "    \"FILE_NAME\": FILE_NAME,\n"
            "    \"WORKER_ID\": WORKER_ID,\n"
            "    \"WORKER_SECRET\": WORKER_SECRET,\n"
            "    \"HUGGINGFACE_TOKEN\": HF_TOKEN,\n"
            "})"
        )

        # Matches the reference notebook's second cell exactly: download
        # the worker script with wget, then invoke it with `!python`
        # (an in-cell shell command) rather than subprocess.Popen, since
        # Popen backgrounds the process and returns immediately -- the
        # kernel would then be considered "complete" by Kaggle with the
        # worker never having actually run.
        run_cell_text = (
            f"!wget -q \"{self.worker_script_url}\" -O inference_server_kaggle.py"
            " && python inference_server_kaggle.py"
        )

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "config-cell",
                    "metadata": {},
                    "outputs": [],
                    "source": _lines_with_newlines(config_cell_text),
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "run-cell",
                    "metadata": {},
                    "outputs": [],
                    "source": _lines_with_newlines(run_cell_text),
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                },
                "kaggle": {
                    "accelerator": "gpu",
                    "dataSources": [],
                    "isInternetEnabled": True,
                    "language": "python",
                    "sourceType": "notebook",
                    "isGpuEnabled": True,
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        return notebook


###############################################################################
# HuggingFace Service
###############################################################################


class HuggingFaceService:
    """HuggingFace Hub integration with caching."""

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        self._cache: Dict[str, tuple] = {}
        self.cache_ttl = 3600

    async def list_repo_files(self, repo: str) -> List[str]:
        """List GGUF files in HF repo (cached)."""
        if repo in self._cache:
            files, timestamp = self._cache[repo]
            if time.time() - timestamp < self.cache_ttl:
                return files

        try:
            headers = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"

            async with httpx.AsyncClient() as client:
                url = f"https://huggingface.co/api/models/{repo}/tree/main"
                response = await client.get(url, headers=headers, timeout=30)
                response.raise_for_status()

                data = response.json()
                files = [
                    item.get("path", "")
                    for item in data.get("siblings", [])
                    if item.get("path", "").endswith(".gguf")
                ]

                self._cache[repo] = (files, time.time())
                return files

        except Exception as e:
            logger.error(f"Failed to list HF repo files for {repo}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to list repo: {e}")


###############################################################################
# Kaggle Service
###############################################################################


class KaggleService:
    """
    Kaggle API integration, lazy-loaded per call.

    `kaggle` is never imported at module load or in __init__. Each call
    sets KAGGLE_API_TOKEN to the target account's real token, then
    (re)imports kaggle/kagglesdk fresh via importlib, so kagglesdk's
    eager credential check always sees a valid token and no state from a
    previous account's call can leak into this one.

    Kernel push/status/delete go through api.kernels_push /
    api.kernels_status / api.kernels_delete(kernel, no_confirm=True).
    kernel-metadata `title` is always set to the tracked slug itself so
    Kaggle's title-derived slug can never diverge from what this
    orchestrator tracks. no_confirm=True is required on delete since the
    default path blocks on stdin, which doesn't exist in this executor
    thread.
    """

    # Modules to purge on each call so a fresh `import kaggle` re-runs its
    # credential resolution against the newly-set env var instead of
    # returning a cached client bound to a different account.
    _KAGGLE_MODULE_PREFIXES = ("kaggle", "kagglesdk")

    def __init__(self):
        # Guards the env-var + reimport window below; os.environ and
        # sys.modules are both process-global, and calls run on a shared
        # thread pool executor.
        self._kaggle_env_lock = threading.Lock()

    @staticmethod
    def _parse_seconds(s) -> int:
        """
        Normalizes a quota value to whole seconds.

        Two shapes have been observed from KaggleQuotaProvider's scrape:
          - "3600s"  -- a duration-string with a trailing "s", already
            in seconds (protobuf Duration JSON mapping style).
          - "18.896" -- a bare decimal with NO unit suffix, which is
            hours (matches what Kaggle's own quota UI shows, e.g.
            "18.9 / 30 hrs"). This is what triggered the crash: the old
            code assumed every string had a trailing "s" and called
            int() directly on "18.896", which isn't a valid int literal.

        Bare ints/floats (no trailing "s") are treated the same as the
        decimal-hours case, since that's what a non-string numeric
        timeUsed/totalTimeAllowed has been observed to represent.
        """
        if isinstance(s, str):
            stripped = s.strip()
            if stripped.endswith("s"):
                # "3600s" style -- already seconds.
                return int(float(stripped[:-1]))
            # "18.896" style -- decimal hours, no unit suffix.
            return int(round(stripped))
        # Non-string numeric: same decimal-hours assumption.
        return int(round(float(s) * 3600))

    @staticmethod
    def _log_http_error_detail(e: Exception, context: str):
        """Logs the response body, since requests' default HTTPError str hides it."""
        response = getattr(e, "response", None)
        if response is not None:
            try:
                logger.error(f"{context}: HTTP {response.status_code} body: {response.text[:2000]}")
            except Exception:
                logger.error(f"{context}: HTTP error but response body could not be read")
        else:
            logger.error(f"{context}: {e!r} (no .response attribute)")

    def _purge_kaggle_modules(self):
        """Drops cached kaggle/kagglesdk modules so the next import is fresh."""
        for name in list(sys.modules):
            if name.startswith(self._KAGGLE_MODULE_PREFIXES):
                del sys.modules[name]

    @contextmanager
    def _authenticated_api(self, account: "AccountState"):
        """
        Yields a KaggleApi authenticated as `account`. Sets KAGGLE_API_TOKEN,
        purges any cached kaggle/kagglesdk modules, imports fresh, builds
        and authenticates the client -- all inside the lock, and the
        caller's actual kernels_* call must happen inside this `with`
        block too, since kagglesdk resolves credentials lazily rather
        than caching them at authenticate() time. Env vars and modules
        are restored/purged again on exit so the next call starts clean.
        """
        if not account.kaggle_api_token:
            raise HTTPException(
                status_code=400,
                detail=f"Account {account.account_id} has no kaggle_api_token configured. "
                       f"Generate one at kaggle.com/settings/api -> 'Generate New Token'."
            )

        if not account.kaggle_api_token.startswith("KGAT_"):
            logger.warning(
                f"Account {account.account_id}'s kaggle_api_token does not start with "
                f"'KGAT_' -- may not be a valid new-style token."
            )

        with self._kaggle_env_lock:
            previous_token = os.environ.get("KAGGLE_API_TOKEN")
            previous_username = os.environ.get("KAGGLE_USERNAME")
            previous_key = os.environ.get("KAGGLE_KEY")

            os.environ["KAGGLE_API_TOKEN"] = account.kaggle_api_token
            os.environ.pop("KAGGLE_USERNAME", None)
            os.environ.pop("KAGGLE_KEY", None)

            self._purge_kaggle_modules()

            try:
                import kaggle  # noqa: F401
                from kaggle.api.kaggle_api_extended import KaggleApi

                api = KaggleApi()
                api.authenticate()
                yield api
            finally:
                def _restore(var, previous):
                    if previous is not None:
                        os.environ[var] = previous
                    else:
                        os.environ.pop(var, None)

                _restore("KAGGLE_API_TOKEN", previous_token)
                _restore("KAGGLE_USERNAME", previous_username)
                _restore("KAGGLE_KEY", previous_key)

                self._purge_kaggle_modules()

    async def refresh_quota(self, account: AccountState) -> tuple:
        try:
            try:
                from kaggle_quota_provider import KaggleQuotaProvider

                def _scrape():
                    with KaggleQuotaProvider() as scraper:
                        return scraper.login_and_scrape(account.username, account.password)

                quota = await asyncio.get_event_loop().run_in_executor(None, _scrape)

                total = self._parse_seconds(quota.gpu_quota["totalTimeAllowed"])
                used = self._parse_seconds(quota.gpu_quota["timeUsed"])
                remaining = total - used
                refresh_time = quota.quota_refresh_time

                return total, used, remaining, refresh_time
            except ImportError:
                logger.warning("KaggleQuotaProvider not available, using mock quota")
                return 108000, 0, 108000, datetime.now(timezone.utc).isoformat()

        except Exception as e:
            logger.error(f"Failed to refresh quota for {account.account_id}: {e}", exc_info=True)
            raise

    async def upload_notebook(
        self,
        account: AccountState,
        notebook_json: dict,
        deployment: DeploymentState,
        config: OrchestratorConfig,
    ) -> tuple:
        """Pushes a notebook (kernel) to Kaggle via kernels_push, returns (notebook_id, notebook_url)."""

        def _push():
            notebook_slug = f"orchestrated-inference-{deployment.deployment_id}"
            kernel_id = f"{account.kaggle_username}/{notebook_slug}"

            kernel_dir = os.path.join(KERNELS_WORKDIR, deployment.deployment_id)
            os.makedirs(kernel_dir, exist_ok=True)

            notebook_filename = f"{notebook_slug}.ipynb"
            notebook_path = os.path.join(kernel_dir, notebook_filename)
            with open(notebook_path, "w") as f:
                json.dump(notebook_json, f, indent=1)

            machine_shape = config.machine_shape or "NvidiaTeslaT4"
            if machine_shape not in VALID_MACHINE_SHAPES:
                logger.warning(
                    f"machine_shape {machine_shape!r} is not a recognized shape "
                    f"({sorted(VALID_MACHINE_SHAPES)}); sending it through as-is."
                )

            # title == notebook_slug so Kaggle's title-derived slug always
            # matches the id and what this orchestrator tracks.
            # machine_shape is what actually selects the GPU type (e.g.
            # T4 x2); enable_gpu alone only requests *some* accelerator.
            metadata = {
                "id": kernel_id,
                "title": notebook_slug,
                "code_file": notebook_filename,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": bool(config.enable_tpu),
                "enable_internet": True,
                "dataset_sources": [],
                "machine_shape": machine_shape,
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            metadata_path = os.path.join(kernel_dir, "kernel-metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            token_len = len(account.kaggle_api_token or "")
            logger.info(
                f"kernels_push: authenticating as kaggle_username="
                f"{account.kaggle_username!r} (token length={token_len}), "
                f"machine_shape={machine_shape!r} for deployment {deployment.deployment_id}"
            )

            try:
                with self._authenticated_api(account) as api:
                    push_result = api.kernels_push(kernel_dir)
                    logger.info(f"kernels_push result for {kernel_id}: {push_result!r}")

                    push_error = getattr(push_result, "error", None)
                    if push_error:
                        raise RuntimeError(f"kernels_push reported an error: {push_error}")
            except Exception as e:
                self._log_http_error_detail(e, f"kernels_push failed for {kernel_id}")
                raise

            notebook_url = f"https://www.kaggle.com/code/{account.kaggle_username}/{notebook_slug}"
            return notebook_slug, notebook_url

        try:
            notebook_id, notebook_url = await asyncio.get_event_loop().run_in_executor(None, _push)
            return notebook_id, notebook_url
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to upload notebook: {e}")
            raise HTTPException(status_code=500, detail=f"Notebook upload failed: {e}")

    async def get_notebook_status(self, deployment: DeploymentState, account: Optional[AccountState] = None) -> str:
        """Queries kernels_status; falls back to a timing-based mock if no account is available."""
        if not deployment.notebook_id or account is None:
            if deployment.started_at is None:
                return "idle"
            elapsed = time.time() - deployment.started_at
            return "running" if elapsed < 5 else "completed"

        def _status():
            kernel_ref = f"{account.kaggle_username}/{deployment.notebook_id}"
            with self._authenticated_api(account) as api:
                result = api.kernels_status(kernel_ref)
                raw_status = getattr(result, "status", None)
                if raw_status is None and isinstance(result, dict):
                    raw_status = result.get("status")
                # Normalize immediately -- kagglesdk can hand back a
                # KernelWorkerStatus enum member here rather than a
                # plain string, and that enum must never be allowed to
                # travel any further (it isn't JSON serializable and
                # str(enum_member) renders as "KernelWorkerStatus.X").
                return _normalize_status(raw_status)

        try:
            raw_status = await asyncio.get_event_loop().run_in_executor(None, _status)
        except Exception as e:
            self._log_http_error_detail(e, f"kernels_status failed for {deployment.notebook_id}")
            logger.error(f"Failed to get notebook status: {e}")
            return "error"

        mapping = {
            "complete": "completed",
            "cancelacknowledged": "stopped",
            "cancelAcknowledged": "stopped",
            "error": "error",
            "running": "running",
            "queued": "idle",
        }
        return mapping.get(raw_status, mapping.get(raw_status.lower(), raw_status))

    async def stop_notebook(self, deployment: DeploymentState, account: Optional[AccountState] = None):
        """Deletes the kernel (Kaggle has no separate stop endpoint), falls back to local-only bookkeeping if unavailable."""
        if not deployment.notebook_id:
            logger.info(
                f"stop_notebook: deployment {deployment.deployment_id} has no notebook_id; nothing to delete remotely."
            )
            return

        if account is None:
            logger.warning(
                f"stop_notebook: no account credentials for deployment {deployment.deployment_id}; "
                f"marking stopped locally only."
            )
            return

        kernel_ref = f"{account.kaggle_username}/{deployment.notebook_id}"

        def _delete():
            with self._authenticated_api(account) as api:
                api.kernels_delete(kernel_ref, no_confirm=True)

        try:
            await asyncio.get_event_loop().run_in_executor(None, _delete)
            logger.info(f"Deleted kernel {kernel_ref} for deployment {deployment.deployment_id}")
        except HTTPException:
            raise
        except Exception as e:
            self._log_http_error_detail(e, f"kernels_delete failed for {kernel_ref}")
            logger.error(
                f"Failed to delete kernel {kernel_ref} for deployment {deployment.deployment_id}: {e}. "
                f"May still be running on Kaggle; delete manually if needed."
            )


###############################################################################
# State Manager
###############################################################################


class StateManager:
    """Manages orchestrator_state.json persistence."""

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = Path(state_file)

    def load(self) -> tuple:
        """Loads state; quota fields are always reset, never trusted from disk."""
        if not self.state_file.exists():
            return {}, {}

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            accounts = {}
            deployments = {}

            for account_id, account_data in data.get("accounts", {}).items():
                deployment_data = account_data.pop("deployment", None)

                account_data["gpu_quota_total_seconds"] = 0
                account_data["gpu_quota_used_seconds"] = 0
                account_data["gpu_quota_remaining_seconds"] = 0
                account_data["gpu_quota_refresh_time"] = ""
                account_data["last_quota_update"] = 0
                account_data.setdefault("kaggle_api_token", "")

                account = AccountState(**account_data)

                if deployment_data:
                    # notebook_status may have been written by an older,
                    # buggy build as the str() of an enum
                    # ("KernelWorkerStatus.RUNNING"); normalize on load
                    # too so old state files don't keep resurfacing it.
                    if "notebook_status" in deployment_data:
                        deployment_data["notebook_status"] = _normalize_status(
                            deployment_data["notebook_status"]
                        )
                    account.deployment = DeploymentState(**deployment_data)

                accounts[account_id] = account

            for dep_id, dep_data in data.get("deployments", {}).items():
                if "notebook_status" in dep_data:
                    dep_data["notebook_status"] = _normalize_status(dep_data["notebook_status"])
                deployments[dep_id] = DeploymentState(**dep_data)

            logger.info(f"Loaded state: {len(accounts)} accounts, {len(deployments)} deployments")
            return accounts, deployments

        except Exception as e:
            logger.warning(f"Failed to load state: {e}. Starting fresh.")
            return {}, {}

    def save(self, accounts: Dict[str, AccountState], deployments: Dict[str, DeploymentState]):
        """Save state to file."""
        try:
            data = {
                "last_updated": time.time(),
                "accounts": {},
                "deployments": {}
            }

            for account_id, account in accounts.items():
                account_dict = asdict(account)
                if account.deployment:
                    account_dict["deployment"] = asdict(account.deployment)
                data["accounts"][account_id] = account_dict

            for dep_id, deployment in deployments.items():
                data["deployments"][dep_id] = asdict(deployment)

            with open(self.state_file, 'w') as f:
                # `default=_json_safe_default` is the safety net: even if
                # a non-serializable value (e.g. a status enum) makes it
                # this far, it's coerced to a plain string instead of
                # crashing the whole save and silently dropping state.
                json.dump(data, f, indent=2, default=_json_safe_default)

            logger.debug(f"State saved to {self.state_file}")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")


###############################################################################
# Orchestrator
###############################################################################


class Orchestrator:
    """Main orchestration service."""

    def __init__(self):
        self.config: Optional[OrchestratorConfig] = None
        self.accounts: Dict[str, AccountState] = {}
        self.deployments: Dict[str, DeploymentState] = {}
        self.websocket = WebSocketManager()
        self.state = StateManager()
        self.kaggle = KaggleService()
        self.hf = HuggingFaceService()
        self.generator: Optional[NotebookGenerator] = None
        self.background_tasks: List[asyncio.Task] = []

    async def startup(self):
        """Initialize orchestrator on startup."""
        try:
            logger.info("Loading config...")
            self.config = self._load_config()
            self.generator = NotebookGenerator(self.config.notebook_template_url)
            self.hf = HuggingFaceService(self.config.hf_token)

            logger.info("Loading saved state...")
            accounts, deployments = self.state.load()
            self.accounts = accounts
            self.deployments = deployments

            logger.info("Initializing accounts...")
            for account_config in self.config.accounts:
                account_id = account_config.get("kaggle_username")

                if account_id not in self.accounts:
                    self.accounts[account_id] = AccountState(
                        account_id=account_id,
                        username=account_config.get("username"),
                        password=account_config.get("password"),
                        kaggle_username=account_config.get("kaggle_username"),
                        kaggle_api_token=account_config.get("kaggle_api_token", ""),
                        gpu_quota_total_seconds=0,
                        gpu_quota_used_seconds=0,
                        gpu_quota_remaining_seconds=0,
                        gpu_quota_refresh_time="",
                        last_quota_update=0,
                    )
                else:
                    existing = self.accounts[account_id]
                    existing.username = account_config.get("username", existing.username)
                    existing.password = account_config.get("password", existing.password)
                    existing.kaggle_username = account_config.get("kaggle_username", existing.kaggle_username)
                    existing.kaggle_api_token = account_config.get("kaggle_api_token", existing.kaggle_api_token)

                token = self.accounts[account_id].kaggle_api_token
                if not token:
                    logger.warning(
                        f"Account {account_id} has no kaggle_api_token set in {CONFIG_FILE}. "
                        f"Kernel push/status calls will fail until it's added."
                    )
                elif not token.startswith("KGAT_"):
                    logger.warning(
                        f"Account {account_id}'s kaggle_api_token does not start with 'KGAT_' "
                        f"and may not authenticate correctly."
                    )

            logger.info("Refreshing quotas (live, not from cached state)...")
            await self.refresh_quotas()

            logger.info("Restoring deployment status...")
            for deployment in self.deployments.values():
                if deployment.notebook_status not in ["completed", "error", "stopped"]:
                    account = self.accounts.get(deployment.account_id)
                    deployment.notebook_status = await self.kaggle.get_notebook_status(deployment, account)

            self.state.save(self.accounts, self.deployments)

            logger.info("Starting background tasks...")
            task1 = asyncio.create_task(self.quota_refresh_loop())
            task2 = asyncio.create_task(self.deployment_status_loop())
            self.background_tasks = [task1, task2]

            logger.info(f"Orchestrator started: {len(self.accounts)} accounts, "
                       f"{len(self.deployments)} deployments")

        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise

    def _load_config(self) -> OrchestratorConfig:
        """Loads orchestrator_config.json. kaggle_api_token must be a KGAT_ token per account."""
        try:
            with open(CONFIG_FILE, 'r') as f:
                config_dict = json.load(f)

            machine_shape = config_dict.get("machine_shape", "NvidiaTeslaT4")
            if machine_shape not in VALID_MACHINE_SHAPES:
                logger.warning(
                    f"{CONFIG_FILE} machine_shape={machine_shape!r} is not one of "
                    f"{sorted(VALID_MACHINE_SHAPES)}; using it as-is, but double-check "
                    f"it matches what Kaggle expects."
                )

            return OrchestratorConfig(
                proxy_url=config_dict.get("proxy_url", ""),
                proxy_shared_secret=config_dict.get("proxy_shared_secret", ""),
                orchestrator_port=config_dict.get("orchestrator_port", 5000),
                accounts=config_dict.get("accounts", []),
                hf_token=config_dict.get("hf_token"),
                notebook_template_url=config_dict.get("notebook_template_url", ""),
                quota_refresh_interval_minutes=config_dict.get("quota_refresh_interval_minutes", 10),
                deployment_status_check_interval_seconds=config_dict.get("deployment_status_check_interval_seconds", 30),
                machine_shape=machine_shape,
                enable_tpu=config_dict.get("enable_tpu", False),
            )
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise HTTPException(status_code=500, detail="Config load failed")

    async def refresh_quotas(self):
        """Refresh GPU quotas for all accounts."""
        for account in self.accounts.values():
            try:
                total, used, remaining, refresh_time = await self.kaggle.refresh_quota(account)
                account.gpu_quota_total_seconds = total
                account.gpu_quota_used_seconds = used
                account.gpu_quota_remaining_seconds = remaining
                account.gpu_quota_refresh_time = refresh_time
                account.last_quota_update = time.time()

                logger.info(f"Updated quota for {account.account_id}: {remaining}s remaining")

                await self.websocket.broadcast({
                    "event": "quota_update",
                    "account_id": account.account_id,
                    "gpu_quota_remaining_seconds": remaining,
                    "gpu_quota_total_seconds": total,
                    "gpu_quota_used_seconds": used,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            except Exception as e:
                logger.warning(f"Failed to refresh quota for {account.account_id}: {e}")

    async def quota_refresh_loop(self):
        """Background task: refresh quotas periodically."""
        interval = (self.config.quota_refresh_interval_minutes * 60) if self.config else 600

        while True:
            try:
                await asyncio.sleep(interval)
                await self.refresh_quotas()
                self.state.save(self.accounts, self.deployments)
            except Exception as e:
                logger.error(f"Quota refresh loop error: {e}")

    async def deployment_status_loop(self):
        """Background task: poll deployment status periodically."""
        interval = self.config.deployment_status_check_interval_seconds if self.config else 30

        while True:
            try:
                await asyncio.sleep(interval)

                for deployment in list(self.deployments.values()):
                    if deployment.notebook_status in ["completed", "error", "stopped"]:
                        continue

                    old_status = deployment.notebook_status
                    account = self.accounts.get(deployment.account_id)
                    # get_notebook_status already normalizes any enum
                    # into a plain string before returning it, so this
                    # assignment is always JSON-safe.
                    deployment.notebook_status = await self.kaggle.get_notebook_status(deployment, account)
                    deployment.last_status_check = time.time()

                    if old_status != deployment.notebook_status:
                        logger.info(f"Deployment {deployment.deployment_id} status changed: "
                                   f"{old_status} -> {deployment.notebook_status}")

                        if deployment.account_id in self.accounts:
                            self.accounts[deployment.account_id].deployment = deployment

                        await self.websocket.broadcast({
                            "event": "deployment_status_changed",
                            "deployment_id": deployment.deployment_id,
                            "status": deployment.notebook_status,
                            "notebook_url": deployment.notebook_url,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                        if deployment.notebook_status == "error":
                            await self.websocket.broadcast({
                                "event": "deployment_error",
                                "deployment_id": deployment.deployment_id,
                                "error": deployment.error_message or "Unknown error",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })

                self.state.save(self.accounts, self.deployments)

            except Exception as e:
                logger.error(f"Deployment status loop error: {e}")

    async def create_deployment(
        self,
        account_id: str,
        model_repo: str,
        model_file: str,
        model_name: str,
        estimated_hours: int,
    ) -> DeploymentState:
        """Create and deploy a new inference server."""

        if account_id not in self.accounts:
            raise HTTPException(status_code=404, detail="Account not found")

        account = self.accounts[account_id]

        if not account.kaggle_api_token:
            raise HTTPException(
                status_code=400,
                detail=f"Account {account_id} has no kaggle_api_token configured in {CONFIG_FILE}."
            )

        quota_needed = estimated_hours * 3600
        if account.gpu_quota_remaining_seconds < quota_needed:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient quota. Need {quota_needed}s, have {account.gpu_quota_remaining_seconds}s"
            )

        if account.deployment:
            raise HTTPException(
                status_code=400,
                detail=f"Account already has active deployment: {account.deployment.deployment_id}"
            )

        try:
            deployment_id = f"dep-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

            deployment = DeploymentState(
                deployment_id=deployment_id,
                account_id=account_id,
                model_name=model_name,
                model_repo=model_repo,
                model_file=model_file,
                notebook_id="",
                notebook_url="",
                notebook_status="creating",
                worker_id=account_id,
                created_at=time.time(),
                quota_reserved_seconds=quota_needed,
            )

            logger.info(f"Generating notebook for {deployment_id}...")
            notebook_json = self.generator.generate(deployment, self.config)

            logger.info(f"Uploading notebook for {deployment_id}...")
            notebook_id, notebook_url = await self.kaggle.upload_notebook(
                account, notebook_json, deployment, self.config
            )

            deployment.notebook_id = notebook_id
            deployment.notebook_url = notebook_url
            deployment.notebook_status = "created"
            deployment.started_at = time.time()

            self.deployments[deployment_id] = deployment
            account.deployment = deployment

            self.state.save(self.accounts, self.deployments)

            logger.info(f"Created deployment {deployment_id}: {notebook_url}")

            await self.websocket.broadcast({
                "event": "deployment_created",
                "deployment_id": deployment_id,
                "account_id": account_id,
                "notebook_url": notebook_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return deployment

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Deployment creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Deployment failed: {e}")

    async def delete_deployment(self, deployment_id: str):
        """Stop (delete the underlying Kaggle kernel) and remove a deployment from local tracking."""
        if deployment_id not in self.deployments:
            raise HTTPException(status_code=404, detail="Deployment not found")

        deployment = self.deployments[deployment_id]

        try:
            account = self.accounts.get(deployment.account_id)
            await self.kaggle.stop_notebook(deployment, account)

            if deployment.account_id in self.accounts:
                self.accounts[deployment.account_id].deployment = None

            deployment.notebook_status = "stopped"

            self.state.save(self.accounts, self.deployments)

            logger.info(f"Deleted deployment {deployment_id}")

            await self.websocket.broadcast({
                "event": "deployment_stopped",
                "deployment_id": deployment_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

    async def refresh_deployment(self, deployment_id: str):
        """Force a status check for a deployment."""
        if deployment_id not in self.deployments:
            raise HTTPException(status_code=404, detail="Deployment not found")

        deployment = self.deployments[deployment_id]
        old_status = deployment.notebook_status

        try:
            account = self.accounts.get(deployment.account_id)
            deployment.notebook_status = await self.kaggle.get_notebook_status(deployment, account)
            deployment.last_status_check = time.time()

            if old_status != deployment.notebook_status:
                await self.websocket.broadcast({
                    "event": "deployment_status_changed",
                    "deployment_id": deployment_id,
                    "status": deployment.notebook_status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            self.state.save(self.accounts, self.deployments)

        except Exception as e:
            logger.error(f"Failed to refresh deployment status: {e}")
            raise HTTPException(status_code=500, detail=f"Refresh failed: {e}")


###############################################################################
# FastAPI App
###############################################################################

app = FastAPI(title="Kaggle Orchestrator")
orch = Orchestrator()


@app.on_event("startup")
async def startup():
    await orch.startup()


###############################################################################
# Accounts Endpoints
###############################################################################


@app.get("/api/accounts")
async def list_accounts():
    """Return all accounts sorted by remaining quota (ascending)."""
    accounts_list = sorted(
        orch.accounts.values(),
        key=lambda a: a.gpu_quota_remaining_seconds
    )

    return {
        "accounts": [
            {
                "account_id": a.account_id,
                "username": a.username,
                "kaggle_username": a.kaggle_username,
                "has_api_token": bool(a.kaggle_api_token),
                "gpu_quota_total_seconds": a.gpu_quota_total_seconds,
                "gpu_quota_used_seconds": a.gpu_quota_used_seconds,
                "gpu_quota_remaining_seconds": a.gpu_quota_remaining_seconds,
                "gpu_quota_refresh_time": a.gpu_quota_refresh_time,
                "last_quota_update": a.last_quota_update,
                "has_deployment": a.deployment is not None,
                "deployment_id": a.deployment.deployment_id if a.deployment else None,
            }
            for a in accounts_list
        ]
    }


###############################################################################
# Deployments Endpoints
###############################################################################


@app.get("/api/deployments")
async def list_deployments():
    """Return all deployments."""
    return {
        "deployments": [
            {
                "deployment_id": d.deployment_id,
                "account_id": d.account_id,
                "model_name": d.model_name,
                "model_repo": d.model_repo,
                "model_file": d.model_file,
                "notebook_id": d.notebook_id,
                "notebook_url": d.notebook_url,
                "notebook_status": d.notebook_status,
                "worker_id": d.worker_id,
                "created_at": d.created_at,
                "started_at": d.started_at,
                "last_status_check": d.last_status_check,
                "error_message": d.error_message,
                "quota_reserved_seconds": d.quota_reserved_seconds,
            }
            for d in orch.deployments.values()
        ]
    }


@app.get("/api/deployments/{deployment_id}")
async def get_deployment(deployment_id: str):
    """Get single deployment status."""
    if deployment_id not in orch.deployments:
        raise HTTPException(status_code=404, detail="Deployment not found")

    d = orch.deployments[deployment_id]
    return {
        "deployment_id": d.deployment_id,
        "account_id": d.account_id,
        "model_name": d.model_name,
        "model_repo": d.model_repo,
        "model_file": d.model_file,
        "notebook_id": d.notebook_id,
        "notebook_url": d.notebook_url,
        "notebook_status": d.notebook_status,
        "worker_id": d.worker_id,
        "created_at": d.created_at,
        "started_at": d.started_at,
        "last_status_check": d.last_status_check,
        "error_message": d.error_message,
        "quota_reserved_seconds": d.quota_reserved_seconds,
    }


@app.post("/api/deployments")
async def deploy(body: dict):
    """Create new deployment."""
    try:
        account_id = body.get("account_id")
        model_repo = body.get("model_repo")
        model_file = body.get("model_file")
        model_name = body.get("model_name")
        estimated_hours = body.get("estimated_quota_hours", 1)

        if not all([account_id, model_repo, model_file, model_name]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: account_id, model_repo, model_file, model_name"
            )

        deployment = await orch.create_deployment(
            account_id=account_id,
            model_repo=model_repo,
            model_file=model_file,
            model_name=model_name,
            estimated_hours=estimated_hours,
        )

        return {
            "deployment_id": deployment.deployment_id,
            "account_id": deployment.account_id,
            "notebook_url": deployment.notebook_url,
            "status": deployment.notebook_status,
            "worker_id": deployment.worker_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deploy endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/deployments/{deployment_id}")
async def undeploy(deployment_id: str):
    """Stop deployment (deletes the underlying Kaggle kernel)."""
    await orch.delete_deployment(deployment_id)
    return {"status": "stopped"}


@app.post("/api/deployments/{deployment_id}/refresh-status")
async def refresh_status(deployment_id: str):
    """Force status refresh."""
    await orch.refresh_deployment(deployment_id)
    return {"status": "refreshed"}


###############################################################################
# Models Endpoints
###############################################################################


@app.get("/api/models/list-files")
async def list_model_files(repo: str):
    """List GGUF files in HuggingFace repo."""
    try:
        files = await orch.hf.list_repo_files(repo)
        return {"repo": repo, "files": files}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list model files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


###############################################################################
# Health Endpoint
###############################################################################


@app.get("/api/health")
async def health():
    """Orchestrator health status."""
    deployments_running = sum(
        1 for d in orch.deployments.values()
        if d.notebook_status == "running"
    )
    deployments_idle = sum(
        1 for d in orch.deployments.values()
        if d.notebook_status == "idle"
    )

    return {
        "status": "ok",
        "accounts_online": len(orch.accounts),
        "deployments_running": deployments_running,
        "deployments_idle": deployments_idle,
        "total_deployments": len(orch.deployments),
    }


###############################################################################
# WebSocket Endpoint
###############################################################################


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await orch.websocket.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"WebSocket received: {data}")
    except WebSocketDisconnect:
        await orch.websocket.disconnect(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await orch.websocket.disconnect(websocket)


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    try:
        with open(CONFIG_FILE, 'r') as f:
            config_dict = json.load(f)
        port = config_dict.get("orchestrator_port", 5000)
    except Exception as e:
        print(f"Warning: Could not load config, using default port 5000: {e}")
        port = 5000

    print(f"Starting Kaggle Orchestrator on port {port}...")
    uvicorn.run(
        "orchestrator:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
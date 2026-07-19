"""
orchestrator.py

Multi-account Kaggle deployment orchestrator: manages GPU quotas, deploys
inference servers, tracks deployment status via REST API + WebSocket.

AUTH DESIGN (Kaggle, kernel push/status/delete): kaggle==2.2.3
(kagglesdk-backed) authenticates via a single KGAT_-prefixed token
through the KAGGLE_API_TOKEN env var. Rather than seeding a placeholder
at import time, this file never imports `kaggle` at module load. Each
authenticated call sets KAGGLE_API_TOKEN to that account's real token,
then (re)imports the kaggle module tree fresh, so kagglesdk's eager
credential check always sees a valid token and no account's credentials
or cached client state can leak into another account's call.

AUTH DESIGN (Kaggle, GPU quota scraping): quota is scraped from
Kaggle's web UI (there is no quota field in the official kaggle API),
which used to mean a fresh Playwright browser login on every single
quota refresh cycle. That is no longer how this works. See
`KaggleSessionManager` below -- the orchestrator instead loads
pre-generated Playwright `storage_state` blobs (one per account) from
the KAGGLE_SESSIONS_JSON env var once at startup, keeps them in memory,
and opens an already-authenticated browser context per quota check.
Playwright login (username/password) is now a rare fallback path, only
ever attempted when an account's own randomized reauth window has
elapsed *and* the stored session has actually stopped working -- never
on a fixed schedule, never on every refresh.

AUTH DESIGN (Orchestrator API): every REST endpoint under /api and the
/ws WebSocket endpoint require the orchestrator's own shared secret
(`orchestrator_api_shared_secret` in config). REST calls send it as a
standard `Authorization: Bearer <secret>` header, checked by the
`require_shared_secret` FastAPI dependency using a constant-time
comparison. Native WebSocket clients can't set arbitrary headers during
the handshake in a browser, so the same secret is instead accepted as a
`?secret=` (or `?token=`) query parameter on the `/ws` connection URL;
the connection is accepted only if it matches, and closed immediately
(policy-violation close code) otherwise.

CONFIG LOADING: orchestrator_config.json is no longer read from disk.
The full config JSON is instead read from the `ORCHESTRATOR_CONFIG`
environment variable, base64-encoded, and decoded once at startup (see
`_load_config`). This keeps secrets (Kaggle tokens, the shared secret
itself, HF token, proxy shared secret, account passwords) out of any
file on disk.

SESSIONS LOADING (new): a second env var, KAGGLE_SESSIONS_JSON, holds a
base64-encoded JSON blob of pre-generated Playwright storage states,
one per Kaggle account, produced offline by the sibling script
`generate_kaggle_sessions_env.py`. This is decoded exactly once at
startup (see `KaggleSessionManager.load_from_env`) and kept in memory
for the lifetime of the process; it is never re-read from the env var
after that. Startup fails fast if this env var is missing or malformed,
mirroring how ORCHESTRATOR_CONFIG is handled.

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

STARTUP RECONCILIATION: orchestrator_state.json is treated as a cache,
not a source of truth. On startup, after quotas are refreshed, the
orchestrator queries every configured Kaggle account directly (via
KaggleService.list_notebooks) and rebuilds self.deployments /
account.deployment from whatever `orchestrated-inference-<id>` kernels
actually exist on Kaggle right now (see
Orchestrator.discover_existing_deployments). Local state that no longer
matches a live Kaggle kernel is dropped; local state that matches is
refreshed with live status/url/id but keeps its saved model metadata.
This is the single per-account defense against split-brain state (e.g.
the orchestrator process restarting after a crash mid-deployment, or
being redeployed against a fresh/empty orchestrator_state.json while
Kaggle notebooks are still running).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import enum
import hmac
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.utils import get_authorization_scheme_param
import uvicorn

###############################################################################
# Constants
###############################################################################

# The config used to be read from this file on disk; it is now read from
# the ORCHESTRATOR_CONFIG env var (base64-encoded JSON) instead. The
# constant is kept only as a label for error messages.
CONFIG_ENV_VAR = "ORCHESTRATOR_CONFIG"

# Base64-encoded JSON blob of pre-generated Playwright storage states,
# one per Kaggle account. Produced offline by
# generate_kaggle_sessions_env.py. Decoded exactly once at startup by
# KaggleSessionManager.load_from_env and kept in memory -- never
# re-read from the env var afterwards. See module docstring
# "SESSIONS LOADING" section above.
SESSIONS_ENV_VAR = "KAGGLE_SESSIONS_JSON"

STATE_FILE = "orchestrator_state.json"
KERNELS_WORKDIR = "/tmp/orchestrator_kernels"

# Prefix used for every notebook/kernel slug this orchestrator creates.
# Startup reconciliation uses this prefix to recognize which of an
# account's Kaggle kernels belong to this orchestrator (see
# Orchestrator.discover_existing_deployments) -- any kernel whose slug
# doesn't start with this is left alone and never touched.
NOTEBOOK_SLUG_PREFIX = "orchestrated-inference-"

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

# WebSocket close code for auth failures (policy violation).
WS_CLOSE_POLICY_VIOLATION = 1008

# Reauthentication window: each account is given its own random interval
# in this range (in days) for how long its stored Playwright session is
# trusted before the orchestrator is even willing to consider a fresh
# Playwright login. This is intentionally randomized per-account (see
# KaggleSessionManager._random_reauth_interval) so accounts don't all
# attempt reauthentication in lockstep.
REAUTH_MIN_DAYS = 2.0
REAUTH_MAX_DAYS = 5.0

# Kaggle homepage used purely as an "are we actually logged in" probe
# after loading a storage_state into a fresh browser context, and to
# get a live page context to read the XSRF cookie from, before calling
# the real quota endpoint.
KAGGLE_HOMEPAGE_URL = "https://www.kaggle.com/"

# Kaggle's actual internal quota RPC (there is no public REST quota
# endpoint). Matches the original KaggleQuotaProvider exactly: POST
# with a JSON body of "{}" and an x-xsrf-token header sourced from the
# session's own XSRF/CSRF cookie.
KAGGLE_BASE = "https://www.kaggle.com"
QUOTA_ENDPOINT = "/api/i/kernels.KernelsService/GetAcceleratorQuotaStatistics"

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


def _deployment_id_from_slug(slug: str) -> Optional[str]:
    """
    Extracts the deployment id from a notebook slug of the form
    `orchestrated-inference-<deployment_id>`, e.g.
    "orchestrated-inference-dep-20260718-123456-abcd12" ->
    "dep-20260718-123456-abcd12".

    Returns None if the slug doesn't carry the expected prefix (callers
    should already have filtered on this, but it's re-checked here so
    this helper is safe to call standalone).
    """
    if not slug or not slug.startswith(NOTEBOOK_SLUG_PREFIX):
        return None
    deployment_id = slug[len(NOTEBOOK_SLUG_PREFIX):]
    return deployment_id or None


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    """Best-effort ISO-8601 -> aware datetime parse. Returns None on any failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


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
    # (Used only for the official kaggle API -- kernel push/status/
    # delete -- never for quota scraping.)
    kaggle_api_token: str
    gpu_quota_total_seconds: int
    gpu_quota_used_seconds: int
    gpu_quota_remaining_seconds: int
    gpu_quota_refresh_time: str
    last_quota_update: float
    deployment: Optional[DeploymentState] = None
    # Playwright-session-based quota auth bookkeeping. None of this is
    # persisted to orchestrator_state.json's schema in a way that's load
    # bearing -- KaggleSessionManager is the source of truth for
    # storage_state/next_reauth_after in memory; these fields just mirror
    # the account's current session status for API visibility
    # (/api/accounts) and logging.
    session_status: str = "unknown"  # "ok" | "SESSION_EXPIRED" | "unknown"
    session_last_verified: Optional[str] = None
    session_next_reauth_after: Optional[str] = None
    session_last_auth_failure: Optional[str] = None


@dataclass
class OrchestratorConfig:
    """Loaded from the base64-encoded JSON in the ORCHESTRATOR_CONFIG env var."""

    proxy_url: str
    proxy_shared_secret: str
    orchestrator_port: int
    accounts: List[dict]
    hf_token: Optional[str]
    notebook_template_url: str
    quota_refresh_interval_minutes: int
    deployment_status_check_interval_seconds: int
    # Shared secret required to call any /api/* endpoint (as
    # `Authorization: Bearer <secret>`) or open the /ws WebSocket (as
    # `?secret=<secret>`, since browsers can't set custom headers on
    # the WebSocket handshake). Required -- the orchestrator refuses to
    # start without it, since running the API unauthenticated would
    # expose Kaggle tokens, HF token, and the proxy shared secret.
    orchestrator_api_shared_secret: str = ""
    # GPU shape requested on kernel push. Must be one of
    # VALID_MACHINE_SHAPES. Defaults to a T4 (the shape used by the
    # known-working reference config).
    machine_shape: str = "NvidiaTeslaT4"
    enable_tpu: bool = False


###############################################################################
# Auth helpers (Orchestrator REST/WS API)
###############################################################################


def _constant_time_eq(a: str, b: str) -> bool:
    """Constant-time string comparison to avoid timing side-channels on secret checks."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def require_shared_secret(authorization: Optional[str] = Header(default=None)) -> None:
    """
    FastAPI dependency enforcing `Authorization: Bearer <orchestrator_api_shared_secret>`
    on every /api/* endpoint. Raises 401 on any mismatch, missing header,
    wrong scheme, or (defensively) if the orchestrator's own secret isn't
    configured -- an unconfigured secret must never be treated as "auth
    disabled".
    """
    configured_secret = orch.config.orchestrator_api_shared_secret if orch.config else ""

    if not configured_secret:
        # Defense in depth: startup() already refuses to boot without a
        # configured secret, but if this is ever reached anyway, fail
        # closed rather than silently allowing every request through.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    scheme, credentials = get_authorization_scheme_param(authorization or "")
    if not authorization or scheme.lower() != "bearer" or not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header; expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not _constant_time_eq(credentials, configured_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _websocket_secret_is_valid(websocket: WebSocket) -> bool:
    """
    Checks the `secret` (or `token`) query parameter on a WebSocket
    connection against orchestrator_api_shared_secret. Browsers cannot
    set arbitrary headers during the WS handshake, so the secret travels
    as a query param here instead of an Authorization header.
    """
    configured_secret = orch.config.orchestrator_api_shared_secret if orch.config else ""
    if not configured_secret:
        logger.error("_websocket_secret_is_valid: no orchestrator_api_shared_secret configured; denying connection.")
        return False

    provided = websocket.query_params.get("secret") or websocket.query_params.get("token")
    if not provided:
        return False

    return _constant_time_eq(provided, configured_secret)


###############################################################################
# Kaggle Session Manager (Playwright storage-state based quota auth)
###############################################################################


class KaggleSessionError(Exception):
    """Raised internally for session-auth problems; always caught and handled."""


class KaggleSessionManager:
    """
    Owns every account's Playwright `storage_state` in memory, loaded
    once at startup from KAGGLE_SESSIONS_JSON (base64-encoded JSON, see
    module docstring). This class is the entire replacement for
    "log into Kaggle with Playwright on every quota refresh":

        Render startup -> decode KAGGLE_SESSIONS_JSON once -> keep in
        memory -> per quota refresh, open a browser context from the
        stored storage_state (no login page, no password field, no
        CAPTCHA) -> call the quota endpoint -> done.

    Reauthentication (an actual Playwright login with username/
    password) is a rare fallback, gated by BOTH of:
      1. wall-clock time has passed this account's own randomized
         `next_reauth_after` (2-5 days out, different per account -- see
         `_random_reauth_interval`), AND
      2. the stored session has been tried and has actually failed.

    Neither condition alone triggers a login. An expired-but-still-
    working session is left alone; a failing-but-not-yet-expired
    session is marked SESSION_EXPIRED and simply skipped (quota refresh
    for that account is skipped) until its window opens -- Kaggle is
    never hammered with repeated login attempts.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # account_id (kaggle_username) -> session dict:
        #   {
        #     "storage_state": {...},
        #     "generated_at": "...",
        #     "last_verified": "...",
        #     "next_reauth_after": "...",  (ISO-8601)
        #     "last_auth_failure": None | "...",
        #     "status": "ok" | "SESSION_EXPIRED",
        #   }
        self._sessions: Dict[str, dict] = {}
        self._loaded = False
        self._playwright = None  # lazily started, shared across calls

        # CRITICAL: Playwright's sync API is greenlet-based and binds its
        # internal dispatch loop to whichever OS thread called
        # sync_playwright().start(). asyncio's default executor
        # (run_in_executor(None, ...)) is a ThreadPoolExecutor that can
        # (and, under any concurrent load, will) run different calls on
        # different worker threads -- the moment a *second* Playwright
        # call lands on a thread other than the one that started the
        # driver, you get
        #   greenlet.error: Cannot switch to a different thread
        # which looks exactly like an auth failure from the caller's
        # point of view (get_quota's except-Exception catches it) and
        # was incorrectly marking healthy sessions as SESSION_EXPIRED.
        #
        # Fix: every single Playwright call in this class -- driver
        # start, browser launch, quota fetch, login -- is funneled
        # through this dedicated ONE-worker executor, so it's always
        # literally the same thread. Never pass this thread pool's
        # worker count as anything but 1.
        import concurrent.futures
        self._pw_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kaggle-playwright"
        )

    def _run_on_playwright_thread(self, fn, *args, **kwargs):
        """
        Runs `fn(*args, **kwargs)` on this manager's single dedicated
        Playwright thread and blocks (from the calling thread) until it
        completes. This is a plain concurrent.futures call, not asyncio
        -- callers awaiting this from async code should wrap it in
        `await loop.run_in_executor(None, self._run_on_playwright_thread, fn, ...)`
        is WRONG (that reintroduces the multi-thread problem); instead
        use `await asyncio.wrap_future(self._pw_executor.submit(fn, *args, **kwargs))`,
        which is exactly what get_quota does below.
        """
        future = self._pw_executor.submit(fn, *args, **kwargs)
        return future.result()

    # -- Loading -------------------------------------------------------

    def load_from_env(self) -> None:
        """
        Decodes KAGGLE_SESSIONS_JSON exactly once and populates
        self._sessions. Raises RuntimeError with a clear message on any
        problem (missing env var, bad base64, bad JSON, bad schema) --
        callers (Orchestrator.startup) should let this abort startup,
        the same way a missing ORCHESTRATOR_CONFIG does.
        """
        raw_env_value = os.environ.get(SESSIONS_ENV_VAR)
        if not raw_env_value:
            raise RuntimeError(
                f"{SESSIONS_ENV_VAR} env var is not set. Generate it locally with "
                f"generate_kaggle_sessions_env.py and set it before starting the "
                f"orchestrator -- the server no longer logs into Kaggle on its own "
                f"for quota refreshes."
            )

        try:
            decoded_bytes = base64.b64decode(raw_env_value, validate=True)
        except (binascii.Error, ValueError) as e:
            raise RuntimeError(f"{SESSIONS_ENV_VAR} is not valid base64: {e}")

        try:
            payload = json.loads(decoded_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise RuntimeError(f"{SESSIONS_ENV_VAR} did not decode to valid JSON: {e}")

        if not isinstance(payload, dict):
            raise RuntimeError(f"{SESSIONS_ENV_VAR} must decode to a JSON object.")

        accounts = payload.get("accounts")
        if not isinstance(accounts, dict) or not accounts:
            raise RuntimeError(
                f"{SESSIONS_ENV_VAR} is missing a non-empty 'accounts' object."
            )

        sessions: Dict[str, dict] = {}
        for account_id, entry in accounts.items():
            if not isinstance(entry, dict) or "storage_state" not in entry:
                raise RuntimeError(
                    f"{SESSIONS_ENV_VAR}.accounts[{account_id!r}] is missing "
                    f"required field 'storage_state'."
                )
            storage_state = entry["storage_state"]
            if not isinstance(storage_state, dict):
                raise RuntimeError(
                    f"{SESSIONS_ENV_VAR}.accounts[{account_id!r}].storage_state "
                    f"must be a JSON object (a Playwright storage_state)."
                )

            next_reauth_after = entry.get("next_reauth_after") or self._format_iso(
                self._random_reauth_deadline()
            )

            sessions[account_id] = {
                "storage_state": storage_state,
                "generated_at": entry.get("generated_at"),
                "last_verified": entry.get("last_verified"),
                "next_reauth_after": next_reauth_after,
                "last_auth_failure": entry.get("last_auth_failure"),
                "status": "ok",
            }

        with self._lock:
            self._sessions = sessions
            self._loaded = True

        logger.info(
            f"KaggleSessionManager: loaded {len(sessions)} account session(s) from "
            f"{SESSIONS_ENV_VAR} (decoded once; kept in memory only)."
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    def has_session(self, account_id: str) -> bool:
        with self._lock:
            return account_id in self._sessions

    def get_status_snapshot(self, account_id: str) -> dict:
        """Read-only snapshot for API/logging surfaces; never mutated by the caller."""
        with self._lock:
            entry = self._sessions.get(account_id)
            if not entry:
                return {
                    "status": "unknown",
                    "last_verified": None,
                    "next_reauth_after": None,
                    "last_auth_failure": None,
                }
            return {
                "status": entry.get("status", "unknown"),
                "last_verified": entry.get("last_verified"),
                "next_reauth_after": entry.get("next_reauth_after"),
                "last_auth_failure": entry.get("last_auth_failure"),
            }

    # -- Random reauth scheduling ---------------------------------------

    @staticmethod
    def _random_reauth_deadline(now: Optional[datetime] = None) -> datetime:
        """
        Picks a fresh random 2-5 day deadline from `now` (default: now),
        independently per call -- this is what gives each account its
        own different interval (e.g. 2.3 days, 4.8 days, 3.1 days)
        rather than every account expiring in lockstep.
        """
        now = now or datetime.now(timezone.utc)
        days = random.uniform(REAUTH_MIN_DAYS, REAUTH_MAX_DAYS)
        return now + timedelta(days=days)

    @staticmethod
    def _format_iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat()

    def _reauth_window_open(self, entry: dict) -> bool:
        """True only if wall-clock time is past this account's next_reauth_after."""
        deadline = _parse_iso8601(entry.get("next_reauth_after"))
        if deadline is None:
            # No parseable deadline -- treat conservatively as "not yet
            # open" rather than reauthenticating immediately/erroneously.
            return False
        return datetime.now(timezone.utc) > deadline

    # -- Playwright plumbing ---------------------------------------------

    def _ensure_playwright(self):
        """
        Lazily starts a single shared Playwright driver, reused across
        calls. MUST only ever be called from within a function submitted
        to self._pw_executor (the one dedicated Playwright thread) --
        never directly from an asyncio task or a general-purpose
        executor thread, or you'll hit the greenlet cross-thread error
        this class exists to avoid.
        """
        if self._playwright is None:
            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
        return self._playwright

    def close(self):
        """
        Cleanup on process shutdown. Stopping Playwright must happen on
        the same dedicated thread it was started on (same greenlet
        constraint as everything else in this class), so this is
        submitted to _pw_executor rather than calling
        self._playwright.stop() directly from whatever thread close()
        happens to be called from.
        """
        if self._playwright is not None:
            def _stop():
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

            try:
                self._pw_executor.submit(_stop).result(timeout=10)
            except Exception:
                pass

        self._pw_executor.shutdown(wait=False, cancel_futures=True)

    @contextmanager
    def _browser_context(self, storage_state: Optional[dict]):
        """
        Yields a Playwright browser context, optionally pre-loaded with
        `storage_state` (skips the login page entirely when provided).
        Guarantees the browser and context are always closed, even on
        exceptions raised inside the `with` block.
        """
        pw = self._ensure_playwright()
        browser = pw.chromium.launch(headless=True)
        try:
            context = browser.new_context(storage_state=storage_state) if storage_state else browser.new_context()
            try:
                yield context
            finally:
                context.close()
        finally:
            browser.close()

    # -- Quota scraping using a stored session ---------------------------

    @staticmethod
    def _get_xsrf_token(page) -> str:
        """
        Reads the XSRF/CSRF token out of the page's own cookies via a
        page.evaluate call (i.e. from inside the authenticated browser
        context), exactly the way the original KaggleQuotaProvider did
        it -- this is required by GetAcceleratorQuotaStatistics, which
        401s without a matching x-xsrf-token header even with valid
        session cookies present.
        """
        token = page.evaluate(
            """
            () => {
            const cookies = document.cookie.split(';').map(c => c.trim());
            const pick = (name) => {
                const hit = cookies.find(c => c.startsWith(name + '='));
                return hit ? hit.slice(name.length + 1) : '';
            };
            return pick('XSRF-TOKEN') || pick('CSRF-TOKEN') || pick('__Host-XSRF-TOKEN');
            }
            """
        )
        if not token:
            return ""
        try:
            return unquote(token)
        except Exception:
            return token

    def _try_quota_with_storage_state(self, storage_state: dict) -> dict:
        """
        Opens an authenticated context from `storage_state`, visits the
        Kaggle homepage (to obtain a live page whose cookies we can read
        the XSRF token from, and as a liveness probe for the session
        itself), then calls Kaggle's real internal quota RPC --
        POST {KAGGLE_BASE}{QUOTA_ENDPOINT}
        ("/api/i/kernels.KernelsService/GetAcceleratorQuotaStatistics")
        -- the same endpoint and calling convention
        (fetch() from page.evaluate, credentials: 'include', JSON body
        "{}", x-xsrf-token header) used by the original
        KaggleQuotaProvider.scrape_usage. There is no public REST quota
        endpoint; this internal RPC is the only source for it.

        Raises KaggleSessionError if anything indicates the session is
        no longer authenticated (redirected to login, non-2xx from the
        quota RPC, or an unparseable response).

        Returns the raw quota JSON (matching KaggleQuotas.from_api_response's
        expected shape: quotaRefreshTime / tpuQuota / gpuQuota) on success.
        """
        with self._browser_context(storage_state) as context:
            page = context.new_page()
            try:
                response = page.goto(KAGGLE_HOMEPAGE_URL, wait_until="domcontentloaded", timeout=30_000)
                if response is None:
                    raise KaggleSessionError("No response loading Kaggle homepage.")

                final_url = page.url or ""
                if "/account/login" in final_url or "/account/sign-in" in final_url:
                    raise KaggleSessionError(
                        "Redirected to Kaggle login page -- stored session is no longer authenticated."
                    )

                xsrf = self._get_xsrf_token(page)
                if not xsrf:
                    logger.warning(
                        "No XSRF token found in cookies for quota check; proceeding without "
                        "it (endpoint may reject the request)."
                    )

                try:
                    quota_json = page.evaluate(
                        """
                        async ({ endpoint, xsrf }) => {
                        const res = await fetch(endpoint, {
                            method: 'POST',
                            headers: {
                            'content-type': 'application/json',
                            ...(xsrf ? { 'x-xsrf-token': xsrf } : {})
                            },
                            credentials: 'include',
                            body: '{}'
                        });

                        const text = await res.text();
                        if (!res.ok) {
                            throw new Error(`HTTP ${res.status}: ${text}`);
                        }

                        try {
                            return JSON.parse(text);
                        } catch (e) {
                            throw new Error(`Failed to parse JSON: ${text}`);
                        }
                        }
                        """,
                        {"endpoint": f"{KAGGLE_BASE}{QUOTA_ENDPOINT}", "xsrf": xsrf},
                    )
                except Exception as e:
                    message = str(e)
                    if "HTTP 401" in message or "HTTP 403" in message:
                        raise KaggleSessionError(
                            f"Quota endpoint rejected the request -- session no longer "
                            f"authenticated: {message}"
                        )
                    raise KaggleSessionError(f"Quota endpoint call failed: {message}")

                if not isinstance(quota_json, dict):
                    raise KaggleSessionError(
                        f"Quota endpoint returned unexpected shape (not a JSON object): {quota_json!r}"
                    )

                return quota_json
            finally:
                page.close()

    def get_quota(self, account: "AccountState") -> dict:
        """
        Main entry point used by KaggleService.refresh_quota. Returns
        the raw quota JSON dict on success.

        Flow:
          1. Load the stored storage_state for this account.
          2. Try it (no login page visited).
          3. On auth failure, retry once with a brand-new browser
             context using the same storage_state (covers transient
             issues, not stale credentials).
          4. If it still fails:
               - If the account's reauth window is open (time expired
                 AND this failure), attempt exactly one Playwright
                 login using username/password, update the in-memory
                 session + a fresh random next_reauth_after on success.
               - Otherwise, mark the account SESSION_EXPIRED and skip
                 (raise KaggleSessionError so the caller skips this
                 refresh cycle) -- never hammer Kaggle with repeated
                 logins outside the account's own window.

        Passwords are only ever touched inside the reauthentication
        branch below -- never during the normal storage_state path.
        """
        with self._lock:
            entry = self._sessions.get(account.account_id)

        if entry is None:
            raise KaggleSessionError(
                f"No stored Playwright session for account {account.account_id!r} in "
                f"{SESSIONS_ENV_VAR}. Run generate_kaggle_sessions_env.py to add it."
            )

        storage_state = entry["storage_state"]

        # Attempt 1, then one retry with a fresh context on the same
        # storage_state (transient-failure tolerant; still never touches
        # the login page or a password).
        last_error: Optional[Exception] = None
        for attempt in (1, 2):
            try:
                quota_json = self._try_quota_with_storage_state(storage_state)
                self._mark_verified(account.account_id)
                return quota_json
            except Exception as e:
                last_error = e
                logger.warning(
                    f"KaggleSessionManager: quota check attempt {attempt}/2 failed for "
                    f"{account.account_id}: {e}"
                )

        # Both attempts failed. Only now do we even consider a login,
        # and only if this account's randomized window has elapsed.
        reauth_open = self._reauth_window_open(entry)

        if not reauth_open:
            self._mark_expired(account.account_id, str(last_error))
            raise KaggleSessionError(
                f"Stored session for {account.account_id} failed and reauth window is "
                f"not yet open (next_reauth_after={entry.get('next_reauth_after')}); "
                f"skipping quota refresh for this account without logging in."
            )

        logger.info(
            f"KaggleSessionManager: reauth window open for {account.account_id} and "
            f"stored session failed -- attempting one Playwright login."
        )
        try:
            new_storage_state = self._login_and_capture_storage_state(account)
        except Exception as e:
            self._mark_expired(account.account_id, str(e))
            raise KaggleSessionError(
                f"Reauthentication failed for {account.account_id}: {e}"
            )

        # Reauth succeeded: persist updated storage_state, generated_at,
        # and a fresh randomized next_reauth_after into memory. No env
        # var is rewritten -- this is in-memory only, as specified.
        now_iso = self._format_iso(datetime.now(timezone.utc))
        new_deadline = self._format_iso(self._random_reauth_deadline())
        with self._lock:
            self._sessions[account.account_id] = {
                "storage_state": new_storage_state,
                "generated_at": now_iso,
                "last_verified": now_iso,
                "next_reauth_after": new_deadline,
                "last_auth_failure": None,
                "status": "ok",
            }
        logger.info(
            f"KaggleSessionManager: session refreshed via Playwright login for "
            f"{account.account_id}; next_reauth_after={new_deadline}"
        )

        # Use the freshly-logged-in session to actually answer this
        # quota request too, rather than making the caller wait another
        # cycle.
        try:
            quota_json = self._try_quota_with_storage_state(new_storage_state)
            self._mark_verified(account.account_id)
            return quota_json
        except Exception as e:
            # Reauth "succeeded" but the very next call still failed --
            # be conservative and mark expired rather than looping.
            self._mark_expired(account.account_id, str(e))
            raise KaggleSessionError(
                f"Quota check failed even immediately after a successful reauth for "
                f"{account.account_id}: {e}"
            )

    def _login_and_capture_storage_state(self, account: "AccountState") -> dict:
        """
        The one and only place password material is used at server
        runtime. Performs an interactive Playwright login using
        account.username / account.password, waits for confirmation of
        an authenticated state, and returns the resulting storage_state
        dict. Headless -- this runtime path assumes credentials alone
        are sufficient (no CAPTCHA-solving here); if Kaggle presents a
        CAPTCHA at runtime, this will fail and the account is correctly
        marked SESSION_EXPIRED rather than hanging the server waiting
        for manual input. (Manual CAPTCHA-assisted login is supported
        only in the offline generate_kaggle_sessions_env.py helper,
        which runs headed/interactively for exactly this reason.)
        """
        if not account.username or not account.password:
            raise KaggleSessionError(
                f"Account {account.account_id} has no username/password configured; "
                f"cannot reauthenticate."
            )

        pw = self._ensure_playwright()
        browser = pw.chromium.launch(headless=True)
        try:
            context = browser.new_context()
            try:
                page = context.new_page()
                page.goto("https://www.kaggle.com/account/login", wait_until="domcontentloaded", timeout=30_000)

                page.fill('input[name="email"]', account.username, timeout=15_000)
                page.fill('input[name="password"]', account.password, timeout=15_000)
                page.click('button[type="submit"]', timeout=15_000)

                page.wait_for_url(lambda url: "/account/login" not in url, timeout=30_000)

                final_url = page.url or ""
                if "/account/login" in final_url:
                    raise KaggleSessionError("Still on login page after submit -- credentials rejected or CAPTCHA required.")

                return context.storage_state()
            finally:
                context.close()
        finally:
            browser.close()

    def _mark_verified(self, account_id: str) -> None:
        with self._lock:
            entry = self._sessions.get(account_id)
            if entry is not None:
                entry["status"] = "ok"
                entry["last_verified"] = self._format_iso(datetime.now(timezone.utc))
                entry["last_auth_failure"] = None

    def _mark_expired(self, account_id: str, error_detail: str) -> None:
        with self._lock:
            entry = self._sessions.get(account_id)
            if entry is not None:
                entry["status"] = "SESSION_EXPIRED"
                entry["last_auth_failure"] = self._format_iso(datetime.now(timezone.utc))
        logger.error(f"KaggleSessionManager: marking {account_id} SESSION_EXPIRED: {error_detail}")


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

    kernels_list is used by startup reconciliation
    (Orchestrator.discover_existing_deployments) to enumerate an
    account's existing kernels so orchestrator-created notebooks can be
    recognized by slug prefix even if orchestrator_state.json is stale,
    missing, or was wiped.

    GPU QUOTA: unlike kernel push/status/delete (which use the official
    KAGGLE_API_TOKEN-based kaggle API), quota has no such endpoint and
    is fetched via KaggleSessionManager, which uses a pre-generated
    Playwright storage_state instead of logging in on every call. See
    `refresh_quota` below and the KaggleSessionManager docstring.
    """

    # Modules to purge on each call so a fresh `import kaggle` re-runs its
    # credential resolution against the newly-set env var instead of
    # returning a cached client bound to a different account.
    _KAGGLE_MODULE_PREFIXES = ("kaggle", "kagglesdk")

    def __init__(self, session_manager: Optional["KaggleSessionManager"] = None):
        # Guards the env-var + reimport window below; os.environ and
        # sys.modules are both process-global, and calls run on a shared
        # thread pool executor.
        self._kaggle_env_lock = threading.Lock()
        # Owns Playwright storage-state-based quota auth. Injected by
        # Orchestrator.startup after KaggleSessionManager.load_from_env
        # succeeds; refresh_quota requires this to be set (there is no
        # more username/password-per-refresh fallback).
        self.session_manager = session_manager

    @staticmethod
    def _parse_seconds(s) -> int:
        """
        Normalizes a quota value to whole seconds.

        Two shapes have been observed from Kaggle's quota JSON:
          - "3600s"  -- a duration-string with a trailing "s", already
            in seconds (protobuf Duration JSON mapping style).
          - "18.896" -- a bare decimal with NO unit suffix, which is
            hours (matches what Kaggle's own quota UI shows, e.g.
            "18.9 / 30 hrs"). This is what triggered a crash historically:
            code that assumed every string had a trailing "s" and called
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
            return int(round(float(stripped) * 3600))
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

        (This is the official-kaggle-API path for kernel push/status/
        delete -- unrelated to quota scraping, which never touches this
        method.)
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

    async def refresh_quota(self, account: AccountState) -> Optional[tuple]:
        """
        Fetches GPU quota for `account` using the pre-generated
        Playwright storage_state via KaggleSessionManager -- no
        per-refresh Kaggle login. Returns (total, used, remaining,
        refresh_time) on success, or None if this account's session is
        unavailable/expired and should simply be skipped this cycle
        (the caller -- Orchestrator.refresh_quotas -- treats None as
        "leave existing quota fields as-is, log, move on").

        Passwords are never read or used here; they are only ever
        touched inside KaggleSessionManager's rare reauthentication
        fallback, gated by that account's own randomized reauth window.
        """
        if self.session_manager is None or not self.session_manager.loaded:
            logger.error(
                f"refresh_quota: no KaggleSessionManager loaded; cannot fetch quota "
                f"for {account.account_id} without KAGGLE_SESSIONS_JSON."
            )
            return None

        # IMPORTANT: this must run on KaggleSessionManager's own
        # dedicated single-worker executor (_pw_executor), NOT on
        # asyncio's default run_in_executor(None, ...) pool. The
        # default pool can (and under concurrent quota refreshes,
        # eventually will) run this call on a different OS thread than
        # the one that originally started the shared Playwright driver,
        # which raises `greenlet.error: Cannot switch to a different
        # thread` -- a failure that looked identical to an auth failure
        # from here, incorrectly flipping healthy accounts to
        # SESSION_EXPIRED. Every Playwright call for every account is
        # serialized through that one thread; this is deliberately not
        # parallel across accounts, which is fine given the refresh
        # interval is minutes, not seconds, and this environment's tiny
        # CPU/RAM budget (0.1 vCPU / 512MB) can't run concurrent
        # browsers anyway.
        future = self.session_manager._pw_executor.submit(self.session_manager.get_quota, account)

        try:
            quota = await asyncio.wrap_future(future)
        except KaggleSessionError as e:
            logger.warning(f"refresh_quota: skipping {account.account_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"refresh_quota: unexpected error for {account.account_id}: {e}", exc_info=True)
            return None

        try:
            # Real Kaggle response shape (matches KaggleQuotas.from_api_response
            # in the reference KaggleQuotaProvider): the GPU quota fields
            # are nested under a top-level "gpuQuota" object, not at the
            # response's top level.
            gpu_quota = quota["gpuQuota"]
            total = self._parse_seconds(gpu_quota["totalTimeAllowed"])
            used = self._parse_seconds(gpu_quota["timeUsed"])
            remaining = total - used
            refresh_time = quota.get("quotaRefreshTime") or ""
            return total, used, remaining, refresh_time
        except Exception as e:
            logger.error(
                f"refresh_quota: quota JSON for {account.account_id} had unexpected "
                f"shape: {e}. Raw: {quota!r}"
            )
            return None

    async def upload_notebook(
        self,
        account: AccountState,
        notebook_json: dict,
        deployment: DeploymentState,
        config: OrchestratorConfig,
    ) -> tuple:
        """Pushes a notebook (kernel) to Kaggle via kernels_push, returns (notebook_id, notebook_url)."""

        def _push():
            notebook_slug = f"{NOTEBOOK_SLUG_PREFIX}{deployment.deployment_id}"
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

    async def list_notebooks(self, account: AccountState) -> List[dict]:
        """
        Lists kernels/notebooks owned by `account` via kernels_list, and
        normalizes each into a plain dict with the fields startup
        reconciliation needs:

            {"ref": "<owner>/<slug>", "slug": "<slug>",
             "title": "<title>", "last_run_time": <float epoch or None>}

        kagglesdk's kernels_list can return either plain dicts or SDK
        model objects depending on version, so field access below tries
        attribute access first and falls back to dict-style access.
        Kernels this orchestrator doesn't recognize (i.e. whose slug
        doesn't start with NOTEBOOK_SLUG_PREFIX) are still included here
        -- filtering by prefix is the caller's job
        (Orchestrator.discover_existing_deployments) -- this method's
        only job is faithfully listing what Kaggle has.

        Returns [] (never raises) if kernels_list itself fails or isn't
        available on this kagglesdk version, so one account's listing
        failure can never abort reconciliation for the rest.
        """

        def _get(obj, *names, default=None):
            for name in names:
                if isinstance(obj, dict):
                    if name in obj:
                        return obj[name]
                else:
                    if hasattr(obj, name):
                        return getattr(obj, name)
            return default

        def _list():
            with self._authenticated_api(account) as api:
                # mine=True restricts to kernels owned by the
                # authenticated account, which is exactly the set we
                # need -- avoids pulling in public/other users' kernels.
                raw_kernels = api.kernels_list(mine=True, page_size=200)

            notebooks = []
            for k in raw_kernels or []:
                ref = _get(k, "ref", "kernelSlug", default="") or ""
                slug = ref.split("/")[-1] if "/" in ref else ref
                if not slug:
                    slug = _get(k, "slug", default="") or ""

                title = _get(k, "title", default=slug) or slug

                last_run = _get(k, "lastRunTime", "last_run_time", "lastRunAt")
                last_run_epoch = None
                if last_run:
                    try:
                        if isinstance(last_run, (int, float)):
                            last_run_epoch = float(last_run)
                        else:
                            parsed = datetime.fromisoformat(str(last_run).replace("Z", "+00:00"))
                            last_run_epoch = parsed.timestamp()
                    except Exception:
                        last_run_epoch = None

                notebooks.append({
                    "ref": ref or f"{account.kaggle_username}/{slug}",
                    "slug": slug,
                    "title": title,
                    "last_run_time": last_run_epoch,
                })

            return notebooks

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _list)
        except Exception as e:
            self._log_http_error_detail(e, f"kernels_list failed for account {account.account_id}")
            logger.error(f"Failed to list notebooks for account {account.account_id}: {e}")
            return []


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
                # Session-status fields are informational mirrors of
                # KaggleSessionManager's in-memory state; never trusted
                # from disk either -- reset and let the session manager
                # (re)populate them as quota checks happen.
                account_data["session_status"] = "unknown"
                account_data["session_last_verified"] = None
                account_data["session_next_reauth_after"] = None
                account_data["session_last_auth_failure"] = None

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
        self.sessions = KaggleSessionManager()
        self.kaggle = KaggleService(session_manager=self.sessions)
        self.hf = HuggingFaceService()
        self.generator: Optional[NotebookGenerator] = None
        self.background_tasks: List[asyncio.Task] = []

    async def startup(self):
        """Initialize orchestrator on startup."""
        try:
            logger.info(f"Loading config from {CONFIG_ENV_VAR} env var...")
            self.config = self._load_config()

            if not self.config.orchestrator_api_shared_secret:
                # Fail closed: refuse to serve an unauthenticated API
                # rather than silently starting with auth disabled.
                raise RuntimeError(
                    "orchestrator_api_shared_secret is not set in the ORCHESTRATOR_CONFIG "
                    "payload. Refusing to start with the API unauthenticated."
                )

            logger.info(f"Loading Kaggle sessions from {SESSIONS_ENV_VAR} env var...")
            # Decoded exactly once here and kept in memory for the life
            # of the process -- see module docstring "SESSIONS LOADING".
            # Startup intentionally fails hard if this is missing or
            # malformed, the same way a missing ORCHESTRATOR_CONFIG does,
            # since there is no more automatic-login fallback.
            self.sessions.load_from_env()

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
                        f"Account {account_id} has no kaggle_api_token set in {CONFIG_ENV_VAR}. "
                        f"Kernel push/status calls will fail until it's added."
                    )
                elif not token.startswith("KGAT_"):
                    logger.warning(
                        f"Account {account_id}'s kaggle_api_token does not start with 'KGAT_' "
                        f"and may not authenticate correctly."
                    )

                if not self.sessions.has_session(account_id):
                    logger.warning(
                        f"Account {account_id} has no entry in {SESSIONS_ENV_VAR}; quota "
                        f"refresh will be skipped for this account until a session is "
                        f"generated for it via generate_kaggle_sessions_env.py."
                    )

            logger.info("Refreshing quotas (using stored Playwright sessions, not live login)...")
            await self.refresh_quotas()

            logger.info("Reconciling deployments against live Kaggle state...")
            await self.discover_existing_deployments()

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

    async def shutdown(self):
        """Cleanup on process shutdown (closes the shared Playwright driver, if started)."""
        try:
            self.sessions.close()
        except Exception as e:
            logger.warning(f"Error during KaggleSessionManager shutdown: {e}")

    async def discover_existing_deployments(self):
        """
        Startup reconciliation: rebuilds self.deployments / account.deployment
        from what's actually running on Kaggle, using orchestrator_state.json
        only as a source of model metadata (model_name/model_repo/model_file)
        to merge in where possible -- never as the source of truth for
        whether a deployment exists or what its live status/url/id are.

        For each configured account this:
          1. Lists the account's Kaggle kernels (KaggleService.list_notebooks).
          2. Filters to slugs starting with NOTEBOOK_SLUG_PREFIX.
          3. Picks the most-recently-updated match if more than one exists
             (an account should only ever have one orchestrator deployment,
             but Kaggle is the source of truth even if that invariant was
             violated out-of-band).
          4. Queries live status via the existing get_notebook_status path.
          5. Builds/updates a DeploymentState, preferring live Kaggle
             fields (status/url/id) and preserving saved model metadata
             when a matching local record existed.

        Any deployment_id present locally (in self.deployments /
        account.deployment) that has no corresponding live Kaggle kernel
        for that account is then dropped -- Kaggle is the source of
        truth, so state that only exists locally is considered stale.

        A failure listing or querying one account is logged and skipped;
        it never aborts startup or affects other accounts.
        """
        restored_count = 0
        stale_removed_count = 0

        # deployment_ids we positively confirmed are still live on Kaggle
        # this run, across all accounts -- anything in self.deployments
        # NOT in this set afterwards gets dropped as stale.
        confirmed_deployment_ids: Set[str] = set()

        for account in self.accounts.values():
            if not account.kaggle_api_token:
                logger.info(
                    f"Skipping reconciliation for account {account.account_id}: "
                    f"no kaggle_api_token configured."
                )
                continue

            try:
                notebooks = await self.kaggle.list_notebooks(account)
            except Exception as e:
                # list_notebooks already catches internally and returns
                # [], but guard here too so a truly unexpected exception
                # (e.g. a bug in list_notebooks itself) still can't take
                # down the rest of reconciliation.
                logger.error(
                    f"Reconciliation: failed to list notebooks for account "
                    f"{account.account_id}: {e}. Skipping this account."
                )
                continue

            matches = []
            for nb in notebooks:
                slug = nb.get("slug", "")
                if not slug.startswith(NOTEBOOK_SLUG_PREFIX):
                    continue
                deployment_id = _deployment_id_from_slug(slug)
                if not deployment_id:
                    continue
                logger.info(f"Found orchestrator notebook: {slug}")
                matches.append((deployment_id, slug, nb))

            if not matches:
                logger.info(f"Account {account.account_id} has no active deployments.")
                continue

            if len(matches) > 1:
                logger.warning(
                    f"Account {account.account_id} has {len(matches)} orchestrator "
                    f"notebooks ({[m[1] for m in matches]}); an account should have "
                    f"at most one. Choosing the most recently updated and ignoring "
                    f"the others."
                )

            # Most-recently-updated wins; treat missing last_run_time as
            # oldest (0) rather than erroring, so accounts/kernels without
            # that field don't crash the sort.
            deployment_id, slug, nb = max(
                matches, key=lambda m: (m[2].get("last_run_time") or 0)
            )

            try:
                notebook_url = f"https://www.kaggle.com/code/{account.kaggle_username}/{slug}"

                # Build a throwaway DeploymentState just to drive the
                # existing get_notebook_status() call, which only needs
                # notebook_id + started_at off the object it's given.
                probe = DeploymentState(
                    deployment_id=deployment_id,
                    account_id=account.account_id,
                    model_name="",
                    model_repo="",
                    model_file="",
                    notebook_id=slug,
                    notebook_url=notebook_url,
                    notebook_status="unknown",
                    worker_id=account.account_id,
                    created_at=time.time(),
                    started_at=time.time(),
                )
                live_status = await self.kaggle.get_notebook_status(probe, account)

                # Merge with whatever we already know locally about this
                # deployment_id, preferring live Kaggle fields but keeping
                # saved model metadata when present.
                existing = self.deployments.get(deployment_id) or (
                    account.deployment if account.deployment and account.deployment.deployment_id == deployment_id
                    else None
                )

                if existing:
                    existing.notebook_id = slug
                    existing.notebook_url = notebook_url
                    existing.notebook_status = live_status
                    existing.account_id = account.account_id
                    existing.worker_id = existing.worker_id or account.account_id
                    existing.last_status_check = time.time()
                    deployment = existing
                else:
                    deployment = DeploymentState(
                        deployment_id=deployment_id,
                        account_id=account.account_id,
                        model_name="unknown",
                        model_repo="unknown",
                        model_file="unknown",
                        notebook_id=slug,
                        notebook_url=notebook_url,
                        notebook_status=live_status,
                        worker_id=account.account_id,
                        created_at=time.time(),
                        started_at=None,
                        error_message=None,
                        quota_reserved_seconds=0,
                    )

                self.deployments[deployment_id] = deployment
                account.deployment = deployment
                confirmed_deployment_ids.add(deployment_id)
                restored_count += 1

                logger.info(f"Recovered deployment: {deployment_id} (status={live_status})")

            except Exception as e:
                logger.error(
                    f"Reconciliation: failed to recover deployment for notebook "
                    f"{slug!r} on account {account.account_id}: {e}. Skipping."
                )
                continue

        # Drop anything local that we didn't just confirm is still live.
        stale_ids = [
            dep_id for dep_id in list(self.deployments.keys())
            if dep_id not in confirmed_deployment_ids
        ]
        for dep_id in stale_ids:
            stale = self.deployments.pop(dep_id, None)
            if stale is not None:
                account = self.accounts.get(stale.account_id)
                if account is not None and account.deployment and account.deployment.deployment_id == dep_id:
                    account.deployment = None
                logger.info(f"Removed stale deployment: {dep_id}")
                stale_removed_count += 1

        # Also clear any account.deployment reference that points at a
        # deployment_id no longer present at all (defensive -- covers the
        # case where account.deployment was set but self.deployments
        # never had a matching entry to begin with).
        for account in self.accounts.values():
            if account.deployment and account.deployment.deployment_id not in confirmed_deployment_ids:
                account.deployment = None

        logger.info(
            f"Startup reconciliation complete: {restored_count} deployments restored, "
            f"{stale_removed_count} stale deployments removed"
        )

    def _load_config(self) -> OrchestratorConfig:
        """
        Loads config from the ORCHESTRATOR_CONFIG env var: base64-encoded
        JSON, decoded once here. orchestrator_api_shared_secret must be
        set; kaggle_api_token must be a KGAT_ token per account.
        """
        raw_env_value = os.environ.get(CONFIG_ENV_VAR)
        if not raw_env_value:
            logger.error(f"{CONFIG_ENV_VAR} env var is not set.")
            raise HTTPException(status_code=500, detail=f"{CONFIG_ENV_VAR} env var is not set")

        try:
            decoded_bytes = base64.b64decode(raw_env_value, validate=True)
        except (binascii.Error, ValueError) as e:
            logger.error(f"{CONFIG_ENV_VAR} is not valid base64: {e}")
            raise HTTPException(status_code=500, detail=f"{CONFIG_ENV_VAR} is not valid base64")

        try:
            config_dict = json.loads(decoded_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"{CONFIG_ENV_VAR} did not decode to valid JSON: {e}")
            raise HTTPException(status_code=500, detail=f"{CONFIG_ENV_VAR} did not decode to valid JSON")

        try:
            machine_shape = config_dict.get("machine_shape", "NvidiaTeslaT4")
            if machine_shape not in VALID_MACHINE_SHAPES:
                logger.warning(
                    f"{CONFIG_ENV_VAR} machine_shape={machine_shape!r} is not one of "
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
                orchestrator_api_shared_secret=config_dict.get("orchestrator_api_shared_secret", ""),
                machine_shape=machine_shape,
                enable_tpu=config_dict.get("enable_tpu", False),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to parse config: {e}")
            raise HTTPException(status_code=500, detail="Config parse failed")

    async def refresh_quotas(self):
        """
        Refresh GPU quotas for all accounts using stored Playwright
        sessions (KaggleSessionManager) rather than logging into Kaggle.
        Accounts whose session is missing/expired and not yet in their
        reauth window are simply skipped for this cycle -- their last-
        known quota fields are left untouched, and this is logged, but
        it never blocks other accounts or raises out of this loop.
        """
        for account in self.accounts.values():
            try:
                result = await self.kaggle.refresh_quota(account)

                # Mirror the session manager's current view of this
                # account into AccountState, purely for /api/accounts
                # visibility and operator debugging.
                snapshot = self.sessions.get_status_snapshot(account.account_id)
                account.session_status = snapshot["status"]
                account.session_last_verified = snapshot["last_verified"]
                account.session_next_reauth_after = snapshot["next_reauth_after"]
                account.session_last_auth_failure = snapshot["last_auth_failure"]

                if result is None:
                    logger.info(
                        f"Skipping quota update for {account.account_id} this cycle "
                        f"(session unavailable/expired; status={account.session_status})."
                    )
                    continue

                total, used, remaining, refresh_time = result
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
                detail=f"Account {account_id} has no kaggle_api_token configured in {CONFIG_ENV_VAR}."
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

orch = Orchestrator()


@app.on_event("startup")
async def startup():
    await orch.startup()


@app.on_event("shutdown")
async def shutdown():
    await orch.shutdown()


###############################################################################
# Accounts Endpoints
###############################################################################


@app.get("/api/accounts", dependencies=[Depends(require_shared_secret)])
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
                "session_status": a.session_status,
                "session_last_verified": a.session_last_verified,
                "session_next_reauth_after": a.session_next_reauth_after,
                "session_last_auth_failure": a.session_last_auth_failure,
            }
            for a in accounts_list
        ]
    }


###############################################################################
# Deployments Endpoints
###############################################################################


@app.get("/api/deployments", dependencies=[Depends(require_shared_secret)])
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


@app.get("/api/deployments/{deployment_id}", dependencies=[Depends(require_shared_secret)])
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


@app.post("/api/deployments", dependencies=[Depends(require_shared_secret)])
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


@app.delete("/api/deployments/{deployment_id}", dependencies=[Depends(require_shared_secret)])
async def undeploy(deployment_id: str):
    """Stop deployment (deletes the underlying Kaggle kernel)."""
    await orch.delete_deployment(deployment_id)
    return {"status": "stopped"}


@app.post("/api/deployments/{deployment_id}/refresh-status", dependencies=[Depends(require_shared_secret)])
async def refresh_status(deployment_id: str):
    """Force status refresh."""
    await orch.refresh_deployment(deployment_id)
    return {"status": "refreshed"}


###############################################################################
# Models Endpoints
###############################################################################


@app.get("/api/models/list-files", dependencies=[Depends(require_shared_secret)])
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


@app.get("/api/health", dependencies=[Depends(require_shared_secret)])
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
    sessions_expired = sum(
        1 for a in orch.accounts.values()
        if a.session_status == "SESSION_EXPIRED"
    )

    return {
        "status": "ok",
        "accounts_online": len(orch.accounts),
        "deployments_running": deployments_running,
        "deployments_idle": deployments_idle,
        "total_deployments": len(orch.deployments),
        "sessions_expired": sessions_expired,
    }


###############################################################################
# WebSocket Endpoint
###############################################################################


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Auth: browsers can't attach an Authorization header to a WebSocket
    handshake, so the client must instead connect with the shared secret
    as a query parameter, e.g. `wss://host/ws?secret=<orchestrator_api_shared_secret>`.
    The connection is rejected (closed with policy-violation code 1008)
    before being added to the broadcast set if the secret is missing or
    wrong.
    """
    if not _websocket_secret_is_valid(websocket):
        await websocket.close(code=WS_CLOSE_POLICY_VIOLATION)
        logger.warning("WebSocket connection rejected: missing or invalid secret query param.")
        return

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
        raw_env_value = os.environ.get(CONFIG_ENV_VAR, "")
        config_dict = json.loads(base64.b64decode(raw_env_value).decode("utf-8")) if raw_env_value else {}
        port = config_dict.get("orchestrator_port", 5000)
    except Exception as e:
        print(f"Warning: Could not read {CONFIG_ENV_VAR}, using default port 5000: {e}")
        port = 5000

    print(f"Starting Kaggle Orchestrator on port {port}...")
    uvicorn.run(
        "orchestrator:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
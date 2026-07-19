from __future__ import annotations

import json
import logging
import os
import shutil
import time
import traceback
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Request,
    Response,
    sync_playwright,
)

# --------------------------------------------------------------------------
# Module-level constants
# --------------------------------------------------------------------------

KAGGLE_BASE = "https://www.kaggle.com"
LOGIN_URL = f"{KAGGLE_BASE}/account/login?phase=emailSignIn&returnUrl=%2F"
QUOTA_ENDPOINT = "/api/i/kernels.KernelsService/GetAcceleratorQuotaStatistics"

# Per-stage timeouts (ms). Deliberately short and specific instead of one
# global 60s timeout, so failures point at the exact stage that hung.
TIMEOUT_PAGE_GOTO_MS = 20_000
TIMEOUT_EMAIL_FIELD_MS = 15_000
TIMEOUT_PASSWORD_FIELD_MS = 5_000
TIMEOUT_SUBMIT_BUTTON_MS = 5_000
TIMEOUT_REDIRECT_MS = 30_000
TIMEOUT_DOM_CONTENT_MS = 15_000
TIMEOUT_COOKIE_BANNER_MS = 3_000
TIMEOUT_QUOTA_FETCH_MS = 15_000

DIAGNOSTICS_ROOT = Path("diagnostics")

# Diagnostics upload target (the ZIP file server). Configurable via env var
# since tunnel URLs like ngrok rotate; the value below is only a fallback.
DIAGNOSTICS_UPLOAD_BASE_URL = os.environ.get(
    "DIAGNOSTICS_UPLOAD_BASE_URL",
    "https://ff08-2406-7400-35-10cd-9ddf-2254-171e-d07d.ngrok-free.app",
)
DIAGNOSTICS_UPLOAD_ENABLED = os.environ.get("DIAGNOSTICS_UPLOAD_ENABLED", "1") != "0"
DIAGNOSTICS_UPLOAD_TIMEOUT_S = 15

logger = logging.getLogger("kaggle_quota_provider")
if not logger.handlers:
    # Library-friendly default: only attach a handler if the host
    # application hasn't already configured logging.
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# --------------------------------------------------------------------------
# Timing instrumentation
# --------------------------------------------------------------------------


class StageTimer:
    """
    Tracks elapsed time since an overall start point, plus per-stage
    durations, so logs can show both "elapsed since start" and
    "this stage took X seconds". Also produces a summary table at the end.
    """

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._stage_start: Optional[float] = None
        self._stage_name: Optional[str] = None
        self.stages: List[tuple[str, float]] = []

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @contextmanager
    def stage(self, name: str):
        """
        Context manager for timing a named stage. Logs entry/exit and
        appends the duration to the summary table.
        """
        t0 = time.monotonic()
        logger.info("[%5.1fs] %s...", self.elapsed(), name)
        try:
            yield
        finally:
            duration = time.monotonic() - t0
            self.stages.append((name, duration))
            logger.info("[%5.1fs] %s done (%.1fs)", self.elapsed(), name, duration)

    def summary_table(self) -> str:
        if not self.stages:
            return "(no stages recorded)"
        name_width = max(len(name) for name, _ in self.stages)
        lines = ["Timing summary:"]
        for name, duration in self.stages:
            lines.append(f"  {name.ljust(name_width)} : {duration:6.2f}s")
        lines.append(f"  {'TOTAL'.ljust(name_width)} : {self.elapsed():6.2f}s")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Diagnostics capture
# --------------------------------------------------------------------------


@dataclass
class PageEventLog:
    """Buffers Playwright page-level events for later diagnostics dump."""

    console: List[str] = field(default_factory=list)
    page_errors: List[str] = field(default_factory=list)
    failed_requests: List[str] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    navigations: List[str] = field(default_factory=list)

    def attach(self, page: Page, timer: StageTimer) -> None:
        def on_console(msg) -> None:
            entry = f"[{timer.elapsed():6.2f}s] {msg.type}: {msg.text}"
            self.console.append(entry)

        def on_page_error(err) -> None:
            entry = f"[{timer.elapsed():6.2f}s] {err}"
            self.page_errors.append(entry)
            logger.warning("Page error: %s", err)

        def on_request_failed(request: Request) -> None:
            failure = request.failure
            entry = (
                f"[{timer.elapsed():6.2f}s] {request.method} {request.url} "
                f"failed: {failure}"
            )
            self.failed_requests.append(entry)
            logger.warning("Request failed: %s %s (%s)", request.method, request.url, failure)

        def on_response(response: Response) -> None:
            entry = f"[{timer.elapsed():6.2f}s] {response.status} {response.url}"
            self.responses.append(entry)
            # Redirects and auth-relevant responses are the most useful to
            # see immediately in the live log stream.
            if response.status in (301, 302, 303, 307, 308) or "login" in response.url:
                logger.debug("Response: %s %s", response.status, response.url)

        def on_frame_navigated(frame) -> None:
            if frame == page.main_frame:
                entry = f"[{timer.elapsed():6.2f}s] navigated -> {frame.url}"
                self.navigations.append(entry)
                logger.info("Navigation: -> %s", frame.url)

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("requestfailed", on_request_failed)
        page.on("response", on_response)
        page.on("framenavigated", on_frame_navigated)


class DiagnosticsUploader:
    """
    Best-effort uploader that zips a diagnostics directory and POSTs it to
    the file server's /upload endpoint (multipart/form-data, field "file").

    Never raises: a failed upload must never mask the original error that
    triggered diagnostics collection, and the local directory remains the
    durable record regardless of upload outcome.
    """

    def __init__(
        self,
        base_url: str = DIAGNOSTICS_UPLOAD_BASE_URL,
        enabled: bool = DIAGNOSTICS_UPLOAD_ENABLED,
        timeout_s: float = DIAGNOSTICS_UPLOAD_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.timeout_s = timeout_s

    def upload_directory(self, directory: Path) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            logger.debug("Diagnostics upload disabled; skipping.")
            return None
        if not self.base_url:
            logger.debug("No diagnostics upload base URL configured; skipping.")
            return None

        try:
            zip_path = self._zip_directory(directory)
        except Exception as e:
            logger.warning("Failed to zip diagnostics directory %s: %s", directory, e)
            return None

        try:
            result = self._post_zip(zip_path)
            logger.info(
                "Uploaded diagnostics %s -> file_id=%s (%s)",
                zip_path.name,
                result.get("file_id"),
                self.base_url,
            )
            return result
        except Exception as e:
            logger.warning(
                "Failed to upload diagnostics zip %s to %s (kept locally at %s): %s",
                zip_path.name,
                self.base_url,
                directory,
                e,
            )
            return None
        finally:
            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _zip_directory(directory: Path) -> Path:
        archive_base = directory.parent / f"{directory.name}-{uuid.uuid4().hex[:8]}"
        archive_path_str = shutil.make_archive(
            base_name=str(archive_base), format="zip", root_dir=str(directory)
        )
        return Path(archive_path_str)

    def _post_zip(self, zip_path: Path) -> Dict[str, Any]:
        boundary = uuid.uuid4().hex
        filename = zip_path.name
        data = zip_path.read_bytes()

        body = bytearray()
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        ).encode()
        body += b"Content-Type: application/zip\r\n\r\n"
        body += data
        body += f"\r\n--{boundary}--\r\n".encode()

        request = urllib.request.Request(
            url=f"{self.base_url}/upload",
            data=bytes(body),
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from diagnostics server: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach diagnostics server: {e.reason}") from e


class DiagnosticsCollector:
    """
    On failure, gathers everything useful about the current page state
    and writes it to a timestamped directory under DIAGNOSTICS_ROOT.
    Never raises: a failure to collect diagnostics must never mask or
    replace the original error.
    """

    def __init__(
        self,
        root: Path = DIAGNOSTICS_ROOT,
        uploader: Optional["DiagnosticsUploader"] = None,
    ) -> None:
        self.root = root
        self.uploader = uploader or DiagnosticsUploader()

    def collect(
        self,
        page: Optional[Page],
        event_log: Optional[PageEventLog],
        error: BaseException,
        stage: str,
        timer: Optional[StageTimer] = None,
    ) -> Optional[Path]:
        try:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = self.root / ts
            out_dir.mkdir(parents=True, exist_ok=True)

            metadata: Dict[str, Any] = {
                "timestamp": ts,
                "stage": stage,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "elapsed_seconds": timer.elapsed() if timer else None,
            }

            if page is not None:
                try:
                    metadata["url"] = page.url
                except Exception as e:
                    metadata["url_error"] = str(e)

                try:
                    metadata["title"] = page.title()
                except Exception as e:
                    metadata["title_error"] = str(e)

                self._safe_write_bytes(
                    out_dir / "screenshot.png",
                    lambda: page.screenshot(full_page=True),
                )
                self._safe_write_text(
                    out_dir / "page.html",
                    lambda: page.content(),
                )
                self._safe_write_json(
                    out_dir / "cookies.json",
                    lambda: page.context.cookies(),
                )
                self._safe_write_json(
                    out_dir / "storage.json",
                    lambda: {
                        "localStorage": page.evaluate(
                            "() => Object.assign({}, window.localStorage)"
                        ),
                        "sessionStorage": page.evaluate(
                            "() => Object.assign({}, window.sessionStorage)"
                        ),
                    },
                )

            if event_log is not None:
                self._safe_write_text(
                    out_dir / "console.log",
                    lambda: "\n".join(event_log.console) or "(no console output)",
                )
                self._safe_write_text(
                    out_dir / "network.log",
                    lambda: "\n".join(
                        [
                            "--- responses ---",
                            *event_log.responses,
                            "",
                            "--- failed requests ---",
                            *event_log.failed_requests,
                            "",
                            "--- navigations ---",
                            *event_log.navigations,
                            "",
                            "--- page errors ---",
                            *event_log.page_errors,
                        ]
                    ),
                )

            metadata["traceback"] = traceback.format_exc()
            self._safe_write_json(out_dir / "metadata.json", lambda: metadata)

            logger.error("Diagnostics written to %s", out_dir)

            # Best-effort upload of the full directory to the remote file
            # server. Failures here are logged, never raised — the local
            # diagnostics directory is already the durable record.
            upload_result = self.uploader.upload_directory(out_dir)
            if upload_result:
                self._safe_write_json(out_dir / "upload.json", lambda: upload_result)

            return out_dir
        except Exception as diag_error:
            # Diagnostics collection must never mask the original error.
            logger.error(
                "Failed to collect diagnostics (original error still raised): %s",
                diag_error,
            )
            return None

    @staticmethod
    def _safe_write_bytes(path: Path, producer) -> None:
        try:
            data = producer()
            path.write_bytes(data)
        except Exception as e:
            logger.debug("Could not write %s: %s", path.name, e)

    @staticmethod
    def _safe_write_text(path: Path, producer) -> None:
        try:
            data = producer()
            path.write_text(data, encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("Could not write %s: %s", path.name, e)

    @staticmethod
    def _safe_write_json(path: Path, producer) -> None:
        try:
            data = producer()
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.debug("Could not write %s: %s", path.name, e)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class KaggleLoginError(RuntimeError):
    """
    Raised when the login flow fails at any stage. Carries rich context
    (stage, current URL/title, elapsed time, likely causes) so the caller
    doesn't just see "TimeoutError".
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        page: Optional[Page] = None,
        elapsed: Optional[float] = None,
        likely_causes: Optional[List[str]] = None,
        diagnostics_dir: Optional[Path] = None,
    ) -> None:
        self.stage = stage
        self.elapsed = elapsed
        self.likely_causes = likely_causes or []
        self.diagnostics_dir = diagnostics_dir

        url = None
        title = None
        if page is not None:
            try:
                url = page.url
            except Exception:
                url = "<unavailable>"
            try:
                title = page.title()
            except Exception:
                title = "<unavailable>"

        parts = [message]
        if elapsed is not None:
            parts.append(f"Elapsed: {elapsed:.1f}s")
        parts.append(f"Stage: {stage}")
        if url is not None:
            parts.append(f"Current URL: {url}")
        if title is not None:
            parts.append(f"Current title: {title}")
        if self.likely_causes:
            parts.append("Possible causes:")
            parts.extend(f"  - {cause}" for cause in self.likely_causes)
        if diagnostics_dir is not None:
            parts.append(f"Diagnostics saved to: {diagnostics_dir}")

        super().__init__("\n".join(parts))


# --------------------------------------------------------------------------
# Data model (unchanged shape, preserved for API compatibility)
# --------------------------------------------------------------------------


@dataclass
class KaggleQuotas:
    quota_refresh_time: str
    tpu_quota: Dict[str, Any]
    gpu_quota: Dict[str, Any]

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "KaggleQuotas":
        try:
            return cls(
                quota_refresh_time=data["quotaRefreshTime"],
                tpu_quota=data["tpuQuota"],
                gpu_quota=data["gpuQuota"],
            )
        except KeyError as e:
            raise KaggleLoginError(
                f"Quota response missing expected field: {e}",
                stage="parse_quota_response",
                likely_causes=[
                    "Kaggle changed the quota API response schema",
                    "Response was an error/HTML page instead of JSON",
                ],
            ) from e


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


class KaggleQuotaProvider:
    """
    Public API is unchanged: login(), scrape_usage(), login_and_scrape().
    Internals are refactored into small, independently testable helpers,
    with structured logging, per-stage timeouts, and automatic diagnostics
    collection on failure.
    """

    def __init__(self, headless: bool = True, timeout_ms: int = 60_000) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._diagnostics = DiagnosticsCollector()

    def __enter__(self) -> "KaggleQuotaProvider":
        t0 = time.monotonic()
        logger.info("Starting Playwright...")
        self._pw = sync_playwright().start()
        logger.info("Launching Chromium (headless=%s)...", self.headless)
        # Flags chosen for constrained cloud environments (Render/Railway/
        # Fly.io style shared-CPU dynos): avoid /dev/shm exhaustion, skip
        # sandboxing (already namespaced by the host container), and skip
        # GPU init that's unused in headless mode anyway.
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-setuid-sandbox",
            ],
        )
        logger.info("Chromium launched in %.1fs", time.monotonic() - t0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Cleanup errors are logged, never raised, so they can't mask a
        # real exception propagating out of the `with` block.
        if self._browser is not None:
            try:
                self._browser.close()
                logger.info("Browser closed.")
            except Exception as e:
                logger.warning("Error closing browser (ignored): %s", e)
        if self._pw is not None:
            try:
                self._pw.stop()
            except Exception as e:
                logger.warning("Error stopping Playwright (ignored): %s", e)

    # ---- context / page setup -------------------------------------------

    def _new_context(self) -> BrowserContext:
        assert self._browser is not None
        return self._browser.new_context(
            viewport={"width": 1440, "height": 1000},
            locale="en-US",
        )

    @staticmethod
    def _first_visible(page: Page, selectors: List[str]):
        for selector in selectors:
            loc = page.locator(selector)
            try:
                if loc.count() and loc.first.is_visible():
                    return loc.first
            except Exception:
                pass
        return None

    def _dismiss_cookie_banner(self, page: Page) -> None:
        """
        Best-effort dismissal of a cookie consent banner, if present.
        Never blocks the flow if none appears.
        """
        selectors = [
            "button:has-text('Accept')",
            "button:has-text('I agree')",
            "button:has-text('Got it')",
            "[data-testid='cookieBanner'] button",
        ]
        for selector in selectors:
            try:
                loc = page.locator(selector)
                loc.first.wait_for(state="visible", timeout=TIMEOUT_COOKIE_BANNER_MS)
                loc.first.click(timeout=TIMEOUT_COOKIE_BANNER_MS)
                logger.info("Dismissed cookie banner via selector: %s", selector)
                return
            except Exception:
                continue
        logger.debug("No cookie banner detected (or none matched known selectors).")

    # ---- login stages -----------------------------------------------------

    def _open_login_page(self, page: Page, timer: StageTimer) -> None:
        with timer.stage("Navigating to login page"):
            try:
                page.goto(
                    LOGIN_URL,
                    wait_until="domcontentloaded",
                    timeout=TIMEOUT_PAGE_GOTO_MS,
                )
            except Exception as e:
                raise KaggleLoginError(
                    "Timed out loading the Kaggle login page.",
                    stage="open_login_page",
                    page=page,
                    elapsed=timer.elapsed(),
                    likely_causes=[
                        "Network issue reaching kaggle.com",
                        "Render/host CPU starvation delaying page load",
                        "Kaggle outage or rate limiting",
                    ],
                ) from e

        logger.info("Login page URL: %s", page.url)
        logger.info("Login page title: %s", page.title())
        self._dismiss_cookie_banner(page)

    def _wait_for_email_field(self, page: Page, timer: StageTimer):
        """
        Resilient locator strategy: Kaggle's login form has been rewritten
        multiple times, so we try several selectors rather than assuming
        one exact `input[name='email']` structure.
        """
        selectors = [
            "input[name='email']",
            "input[type='email']",
            "input#email",
            "input[autocomplete='username']",
        ]
        with timer.stage("Waiting for email field"):
            for selector in selectors:
                try:
                    loc = page.locator(selector).first
                    loc.wait_for(state="visible", timeout=TIMEOUT_EMAIL_FIELD_MS)
                    logger.info("Email field found via selector: %s", selector)
                    return loc
                except Exception:
                    continue

        raise KaggleLoginError(
            "Timed out waiting for the email field.",
            stage="wait_for_email_field",
            page=page,
            elapsed=timer.elapsed(),
            likely_causes=[
                "Kaggle login page layout/selectors changed",
                "Redirected to Google SSO or an account-chooser screen",
                "Page failed to fully render (CPU starvation on host)",
            ],
        )

    def _fill_credentials(self, page: Page, email_field, username: str, password: str, timer: StageTimer) -> None:
        with timer.stage("Filling credentials"):
            try:
                email_field.fill(username)
                logger.info("Filled email field.")
            except Exception as e:
                raise KaggleLoginError(
                    "Failed to fill the email field.",
                    stage="fill_email",
                    page=page,
                    elapsed=timer.elapsed(),
                    likely_causes=["Field became detached/stale", "Page re-rendered mid-fill"],
                ) from e

            password_selectors = [
                "input[name='password']",
                "input[type='password']",
                "input#password",
                "input[autocomplete='current-password']",
            ]
            password_field = self._first_visible(page, password_selectors)
            if password_field is None:
                # Some Kaggle flows show the password field only after the
                # email step is submitted (two-step form).
                try:
                    page.locator(password_selectors[0]).first.wait_for(
                        state="visible", timeout=TIMEOUT_PASSWORD_FIELD_MS
                    )
                    password_field = page.locator(password_selectors[0]).first
                except Exception as e:
                    raise KaggleLoginError(
                        "Timed out waiting for the password field.",
                        stage="wait_for_password_field",
                        page=page,
                        elapsed=timer.elapsed(),
                        likely_causes=[
                            "Login flow requires a separate 'Next' click before password appears",
                            "Kaggle login page layout changed",
                        ],
                    ) from e

            try:
                password_field.fill(password)
                logger.info("Filled password field.")
            except Exception as e:
                raise KaggleLoginError(
                    "Failed to fill the password field.",
                    stage="fill_password",
                    page=page,
                    elapsed=timer.elapsed(),
                    likely_causes=["Field became detached/stale", "Page re-rendered mid-fill"],
                ) from e

    def _submit_login(self, page: Page, timer: StageTimer) -> None:
        selectors = [
            "button[type='submit']",
            "button:has-text('Sign In')",
            "button:has-text('Sign in')",
            "button:has-text('Log in')",
        ]
        with timer.stage("Submitting login form"):
            submit_button = self._first_visible(page, selectors)
            if submit_button is None:
                for selector in selectors:
                    try:
                        loc = page.locator(selector).first
                        loc.wait_for(state="visible", timeout=TIMEOUT_SUBMIT_BUTTON_MS)
                        submit_button = loc
                        break
                    except Exception:
                        continue

            if submit_button is None:
                raise KaggleLoginError(
                    "Could not find a login submit button.",
                    stage="find_submit_button",
                    page=page,
                    elapsed=timer.elapsed(),
                    likely_causes=[
                        "Kaggle changed the submit button markup/text",
                        "Form requires additional steps before a submit button appears",
                    ],
                )

            try:
                self._screenshot_best_effort(page, "before_submit")
                submit_button.click(timeout=TIMEOUT_SUBMIT_BUTTON_MS)
                logger.info("Clicked login submit button.")
            except Exception as e:
                raise KaggleLoginError(
                    "Failed to click the login submit button.",
                    stage="click_submit",
                    page=page,
                    elapsed=timer.elapsed(),
                    likely_causes=["Button obscured by an overlay", "Button became stale"],
                ) from e

    def _wait_for_authenticated_page(self, page: Page, timer: StageTimer) -> None:
        with timer.stage("Waiting for post-login redirect"):
            try:
                page.wait_for_url(
                    lambda url: "/account/login" not in url,
                    timeout=TIMEOUT_REDIRECT_MS,
                )
            except Exception as e:
                self._screenshot_best_effort(page, "redirect_timeout")
                raise KaggleLoginError(
                    "Timed out waiting for the post-login redirect.",
                    stage="wait_for_redirect",
                    page=page,
                    elapsed=timer.elapsed(),
                    likely_causes=[
                        "Invalid credentials",
                        "CAPTCHA or bot-detection challenge",
                        "Additional account verification step (email/2FA)",
                        "Render/host CPU starvation delaying navigation",
                        "Kaggle login flow changed (e.g. Google SSO redirect, account chooser)",
                    ],
                ) from e

        with timer.stage("Waiting for DOMContentLoaded on landing page"):
            try:
                page.wait_for_load_state(
                    "domcontentloaded", timeout=TIMEOUT_DOM_CONTENT_MS
                )
            except Exception as e:
                raise KaggleLoginError(
                    "Timed out waiting for the post-login page to finish loading.",
                    stage="wait_for_dom_content",
                    page=page,
                    elapsed=timer.elapsed(),
                    likely_causes=[
                        "Slow page render under CPU starvation",
                        "Heavy background requests delaying DOMContentLoaded",
                    ],
                ) from e

        self._screenshot_best_effort(page, "after_redirect")
        logger.info("Post-login URL: %s", page.url)
        logger.info("Post-login title: %s", page.title())

        if "/account/login" in page.url:
            raise KaggleLoginError(
                "Still on the login page after the redirect wait completed.",
                stage="verify_login_success",
                page=page,
                elapsed=timer.elapsed(),
                likely_causes=[
                    "Invalid credentials",
                    "Account requires additional verification",
                ],
            )

        logger.info("Login successful.")

    @staticmethod
    def _screenshot_best_effort(page: Page, label: str) -> None:
        """
        Ad-hoc screenshots at key moments (before submit, after redirect)
        kept in-memory-free — written straight to a scratch path so they
        don't depend on the failure/diagnostics path succeeding.
        """
        try:
            path = DIAGNOSTICS_ROOT / "_live" / f"{label}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(path))
        except Exception as e:
            logger.debug("Best-effort screenshot '%s' failed: %s", label, e)

    # ---- public API --------------------------------------------------------

    def login(self, username: str, password: str) -> BrowserContext:
        """
        Opens the Kaggle login page, signs in, and returns an authenticated
        browser context. Signature and return type unchanged.
        """
        timer = StageTimer()
        context = self._new_context()
        page = context.new_page()
        page.set_default_timeout(self.timeout_ms)

        event_log = PageEventLog()
        event_log.attach(page, timer)

        try:
            self._open_login_page(page, timer)
            email_field = self._wait_for_email_field(page, timer)
            self._fill_credentials(page, email_field, username, password, timer)
            self._submit_login(page, timer)
            self._wait_for_authenticated_page(page, timer)
        except KaggleLoginError as e:
            diag_dir = self._diagnostics.collect(page, event_log, e, e.stage, timer)
            e.diagnostics_dir = diag_dir
            logger.error(timer.summary_table())
            try:
                context.close()
            except Exception as close_err:
                logger.warning("Error closing context after failure (ignored): %s", close_err)
            raise
        except Exception as e:
            # Catch-all so unexpected Playwright/browser errors still get
            # full diagnostics instead of a bare traceback.
            diag_dir = self._diagnostics.collect(page, event_log, e, "unknown", timer)
            logger.error(timer.summary_table())
            try:
                context.close()
            except Exception as close_err:
                logger.warning("Error closing context after failure (ignored): %s", close_err)
            raise KaggleLoginError(
                f"Unexpected error during login: {e}",
                stage="unknown",
                page=page,
                elapsed=timer.elapsed(),
                likely_causes=["Unhandled Playwright/browser-level failure"],
                diagnostics_dir=diag_dir,
            ) from e

        logger.info(timer.summary_table())
        return context

    @staticmethod
    def _get_xsrf_token(page: Page) -> str:
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

    def scrape_usage(self, page: Page) -> KaggleQuotas:
        """
        Uses the authenticated browser session to fetch quota stats.
        Signature and return type unchanged.
        """
        timer = StageTimer()

        with timer.stage("Extracting XSRF token"):
            xsrf = self._get_xsrf_token(page)
            if xsrf:
                logger.info("XSRF token found (len=%d).", len(xsrf))
            else:
                logger.warning("No XSRF token found in cookies; proceeding without it.")

        with timer.stage("Calling quota endpoint"):
            try:
                response_json = page.evaluate(
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
                    {"endpoint": QUOTA_ENDPOINT, "xsrf": xsrf},
                )
            except Exception as e:
                diag_dir = self._diagnostics.collect(
                    page, None, e, "quota_fetch", timer
                )
                raise KaggleLoginError(
                    f"Failed to fetch or parse the quota endpoint: {e}",
                    stage="quota_fetch",
                    page=page,
                    elapsed=timer.elapsed(),
                    likely_causes=[
                        "Session expired or XSRF token invalid",
                        "Kaggle changed the quota API endpoint/schema",
                        "Network issue reaching kaggle.com from within the page context",
                    ],
                    diagnostics_dir=diag_dir,
                ) from e

        with timer.stage("Parsing quota response"):
            quotas = KaggleQuotas.from_api_response(response_json)

        logger.info(timer.summary_table())
        return quotas

    def login_and_scrape(self, username: str, password: str) -> KaggleQuotas:
        """
        Convenience wrapper: login then scrape, always closing the context.
        Signature and return type unchanged.
        """
        overall_start = time.monotonic()
        context = self.login(username, password)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(self.timeout_ms)
            quotas = self.scrape_usage(page)
            logger.info(
                "login_and_scrape completed in %.1fs total.",
                time.monotonic() - overall_start,
            )
            return quotas
        finally:
            try:
                context.close()
            except Exception as e:
                logger.warning("Error closing context in login_and_scrape (ignored): %s", e)
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

KAGGLE_BASE = "https://www.kaggle.com"
LOGIN_URL = f"{KAGGLE_BASE}/account/login?phase=emailSignIn&returnUrl=%2F"
QUOTA_ENDPOINT = "/api/i/kernels.KernelsService/GetAcceleratorQuotaStatistics"

@dataclass
class KaggleQuotas:
    quota_refresh_time: str
    tpu_quota: Dict[str, Any]
    gpu_quota: Dict[str, Any]

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "KaggleQuotas":
        return cls(
            quota_refresh_time=data["quotaRefreshTime"],
            tpu_quota=data["tpuQuota"],
            gpu_quota=data["gpuQuota"],
        )

class KaggleQuotaProvider:
    def __init__(self, headless: bool = True, timeout_ms: int = 60_000) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None

    def __enter__(self) -> "KaggleQuotaProvider":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._pw is not None:
            self._pw.stop()

    def _new_context(self) -> BrowserContext:
        assert self._browser is not None
        return self._browser.new_context(
            viewport={"width": 1440, "height": 1000},
            locale="en-US",
        )

    @staticmethod
    def _first_visible(page: Page, selectors: list[str]):
        for selector in selectors:
            loc = page.locator(selector)
            try:
                if loc.count() and loc.first.is_visible():
                    return loc.first
            except Exception:
                pass
        return None

    def login(self, username: str, password: str) -> BrowserContext:
        """
        Opens the Kaggle login page, signs in, and returns an authenticated browser context.
        """
        context = self._new_context()
        page = context.new_page()
        page.set_default_timeout(60000)

        # Open login page
        page.goto(LOGIN_URL, wait_until="domcontentloaded")

        print(f"URL   : {page.url}")
        print(f"Title : {page.title()}")

        # Wait for login form
        page.wait_for_selector(
            "input[name='email']",
            state="visible",
        )

        # Fill credentials
        page.fill("input[name='email']", username)
        page.fill("input[name='password']", password)

        print("Signing in...")

        # Submit
        page.click("button[type='submit']")

        # Wait until we leave the login page
        page.wait_for_url(
            lambda url: "/account/login" not in url,
            timeout=60000,
        )

        # Wait for the new document to load.
        # Do NOT use networkidle—Kaggle continuously makes background requests.
        page.wait_for_load_state("domcontentloaded")

        print(f"Current URL  : {page.url}")
        print(f"Current Title: {page.title()}")

        # Verify login succeeded
        if "/account/login" in page.url:
            page.screenshot(
                path="kaggle_login_failed.png",
                full_page=True,
            )
            raise RuntimeError(
                "Login failed. Still on the login page."
            )

        print("Successfully logged in!")

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
        # Token may be URL-encoded in cookies.
        try:
            from urllib.parse import unquote
            return unquote(token)
        except Exception:
            return token

    def scrape_usage(self, page: Page) -> KaggleQuotas:
        """
        Uses the authenticated browser session to fetch quota stats.
        """
        xsrf = self._get_xsrf_token(page)

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

        return KaggleQuotas.from_api_response(response_json)

    def login_and_scrape(self, username: str, password: str) -> KaggleQuotas:
        context = self.login(username, password)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(self.timeout_ms)
            return self.scrape_usage(page)
        finally:
            context.close()


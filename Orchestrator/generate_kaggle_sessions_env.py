#!/usr/bin/env python3
"""
generate_kaggle_sessions_env.py

Offline helper (run locally, NOT on the server) that logs into Kaggle
once per account with Playwright, captures each account's authenticated
browser `storage_state`, and packages all of them into the base64-
encoded JSON blob the orchestrator expects in its KAGGLE_SESSIONS_JSON
environment variable.

This is the *only* place interactive/CAPTCHA-assisted Kaggle login is
expected to happen. The orchestrator itself never logs in during normal
operation -- it loads the storage states this script produces and only
falls back to a Playwright login on the rare occasion a given account's
session has both (a) actually failed, and (b) passed that account's own
randomized 2-5 day reauth window. See orchestrator.py's module
docstring for the full picture.

USAGE

    python3 generate_kaggle_sessions_env.py --config orchestrator_config.json

    # or read the same config shape from an env var, e.g. the same
    # ORCHESTRATOR_CONFIG payload the server uses (base64 JSON):
    python3 generate_kaggle_sessions_env.py --config-env ORCHESTRATOR_CONFIG

The input config is the same shape as ORCHESTRATOR_CONFIG's decoded
JSON:

    {
      "proxy_url": "...",
      "proxy_shared_secret": "...",
      "orchestrator_port": 5000,
      "accounts": [
        {
          "username": "...",
          "password": "...",
          "kaggle_username": "...",
          "kaggle_api_token": "..."
        }
      ]
    }

Only `username`, `password`, and `kaggle_username` are used by this
script; `kaggle_api_token` (and other top-level fields) are read from
the same file for convenience but ignored here.

OUTPUT

  - Prints ONLY the final base64 string to stdout (nothing else --
    logging goes to stderr so stdout can be piped/captured directly,
    e.g. `... > sessions.b64` or straight into a secret manager).
  - Also writes `kaggle_sessions.json` (pretty-printed, un-encoded) next
    to this script for local inspection/debugging. This file contains
    live session cookies -- treat it exactly like a password file
    (gitignore it, delete it once you've set the env var, etc).

LOGIN FLOW / CAPTCHA HANDLING

  Runs headed (a visible browser window) by default so a human can step
  in if Kaggle shows a CAPTCHA or an unexpected verification screen: the
  script fills in username/password and clicks submit automatically,
  then waits for the URL to leave the login page. If that doesn't happen
  within the initial short timeout, it prints a message asking you to
  complete whatever is on-screen (CAPTCHA, 2FA, etc.) and waits (with a
  much longer timeout) for you to do so before continuing on its own --
  no need to restart the script or intervene in code.

  Use --headless to skip the visible browser (only recommended if you
  know a given account won't hit a CAPTCHA, e.g. it logged in manually
  very recently).

ONE BROWSER, MANY ACCOUNTS

  A single Playwright browser process is launched once and reused
  across all accounts; each account still gets its own fresh, isolated
  BrowserContext (so cookies/storage never leak between accounts), but
  we avoid the overhead of relaunching the browser binary per account.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import uniform
from typing import Any, Dict, List, Optional

logger = logging.getLogger("generate_kaggle_sessions_env")

# Matches orchestrator.py's REAUTH_MIN_DAYS / REAUTH_MAX_DAYS exactly,
# so sessions generated here start out on the same randomized schedule
# the server itself would assign on a later in-memory reauth.
REAUTH_MIN_DAYS = 2.0
REAUTH_MAX_DAYS = 5.0

KAGGLE_LOGIN_URL = "https://www.kaggle.com/account/login?phase=emailSignIn&returnUrl=%2F"

# How long we wait, initially, for a normal (no-CAPTCHA) login to
# resolve before assuming manual intervention is needed.
AUTOMATIC_LOGIN_TIMEOUT_MS = 20_000
# How long we're willing to wait for a human to clear a CAPTCHA/2FA
# screen once we've asked them to. Generous on purpose.
MANUAL_INTERVENTION_TIMEOUT_MS = 5 * 60_000

OUTPUT_JSON_PATH = Path(__file__).resolve().parent / "kaggle_sessions.json"


###############################################################################
# Config loading
###############################################################################


def _load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads the orchestrator-config-shaped JSON from --config (file) or --config-env (env var, base64)."""
    if args.config:
        path = Path(args.config)
        if not path.exists():
            logger.error(f"Config file not found: {path}")
            sys.exit(1)
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as e:
            logger.error(f"Config file {path} is not valid JSON: {e}")
            sys.exit(1)

    if args.config_env:
        import os
        raw = os.environ.get(args.config_env)
        if not raw:
            logger.error(f"Env var {args.config_env!r} is not set.")
            sys.exit(1)
        try:
            decoded = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as e:
            logger.error(f"Env var {args.config_env!r} is not valid base64: {e}")
            sys.exit(1)
        try:
            return json.loads(decoded.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Env var {args.config_env!r} did not decode to valid JSON: {e}")
            sys.exit(1)

    logger.error("Provide either --config <path> or --config-env <ENV_VAR_NAME>.")
    sys.exit(1)


def _extract_accounts(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Pulls the (username, password, kaggle_username) triples this script needs out of the config."""
    accounts = config.get("accounts")
    if not isinstance(accounts, list) or not accounts:
        logger.error("Config has no non-empty 'accounts' list.")
        sys.exit(1)

    result = []
    for i, acct in enumerate(accounts):
        username = acct.get("username")
        password = acct.get("password")
        kaggle_username = acct.get("kaggle_username")
        missing = [
            name for name, value in (
                ("username", username), ("password", password), ("kaggle_username", kaggle_username)
            ) if not value
        ]
        if missing:
            logger.warning(
                f"Skipping accounts[{i}]: missing required field(s) {missing}."
            )
            continue
        result.append({
            "username": username,
            "password": password,
            "kaggle_username": kaggle_username,
        })

    if not result:
        logger.error("No usable accounts found in config (all missing required fields).")
        sys.exit(1)

    return result


###############################################################################
# Random reauth deadline (mirrors orchestrator.py's scheduling)
###############################################################################


def _random_reauth_deadline(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    days = uniform(REAUTH_MIN_DAYS, REAUTH_MAX_DAYS)
    return (now + timedelta(days=days)).isoformat()


###############################################################################
# Playwright login
###############################################################################


def _login_one_account(browser, account: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Logs into Kaggle as `account` in a fresh, isolated BrowserContext and
    returns that context's storage_state() dict on success, or None on
    failure (already logged; caller decides whether that's fatal).
    """
    kaggle_username = account["kaggle_username"]
    logger.info(f"[{kaggle_username}] Starting login...")

    context = browser.new_context()
    try:
        page = context.new_page()
        t0 = time.monotonic()

        page.goto(KAGGLE_LOGIN_URL, wait_until="domcontentloaded", timeout=30_000)

        try:
            page.fill('input[name="email"]', account["username"], timeout=10_000)
            page.fill('input[name="password"]', account["password"], timeout=10_000)
            page.click('button[type="submit"]', timeout=10_000)
        except Exception as e:
            logger.warning(
                f"[{kaggle_username}] Could not fill/submit the standard login form "
                f"automatically ({e}); the page may already need manual attention."
            )

        # First, give the automatic path a normal amount of time.
        try:
            page.wait_for_url(lambda url: "/account/login" not in url, timeout=AUTOMATIC_LOGIN_TIMEOUT_MS)
        except Exception:
            # Likely a CAPTCHA, 2FA prompt, or unusual verification step.
            # Hand off to the human instead of failing outright.
            print(
                f"\n>>> [{kaggle_username}] Login did not complete automatically. "
                f"If a CAPTCHA or verification step is showing in the opened browser "
                f"window, please complete it now. Waiting up to "
                f"{MANUAL_INTERVENTION_TIMEOUT_MS // 1000}s for you to finish... <<<\n",
                file=sys.stderr,
            )
            try:
                page.wait_for_url(lambda url: "/account/login" not in url, timeout=MANUAL_INTERVENTION_TIMEOUT_MS)
            except Exception:
                logger.error(
                    f"[{kaggle_username}] Still on the login page after manual-intervention "
                    f"window elapsed. Skipping this account."
                )
                return None

        final_url = page.url or ""
        if "/account/login" in final_url:
            logger.error(f"[{kaggle_username}] Login did not succeed (still on login page).")
            return None

        elapsed = time.monotonic() - t0
        logger.info(f"[{kaggle_username}] Login succeeded in {elapsed:.1f}s.")

        storage_state = context.storage_state()
        return storage_state
    finally:
        context.close()


def _generate_sessions(accounts: List[Dict[str, str]], headless: bool) -> Dict[str, Any]:
    """
    Launches one shared Playwright browser and logs into every account
    in turn (each in its own isolated context), building the final
    {"version": 1, "generated_at": ..., "accounts": {...}} payload.

    An account that fails to log in is omitted from the output (logged
    clearly) rather than aborting the whole run -- so one bad password
    or one stuck CAPTCHA doesn't block generating sessions for the rest.
    """
    from playwright.sync_api import sync_playwright

    now_iso = datetime.now(timezone.utc).isoformat()
    accounts_out: Dict[str, Any] = {}
    succeeded = 0
    failed = 0

    with sync_playwright() as p:
        logger.info(f"Launching browser (headless={headless})...")
        browser = p.chromium.launch(headless=headless)
        try:
            for account in accounts:
                kaggle_username = account["kaggle_username"]
                try:
                    storage_state = _login_one_account(browser, account)
                except Exception as e:
                    logger.error(f"[{kaggle_username}] Unexpected error during login: {e}")
                    storage_state = None

                if storage_state is None:
                    failed += 1
                    continue

                accounts_out[kaggle_username] = {
                    "storage_state": storage_state,
                    "generated_at": now_iso,
                    "last_verified": now_iso,
                    "next_reauth_after": _random_reauth_deadline(),
                    "last_auth_failure": None,
                }
                succeeded += 1
        finally:
            browser.close()

    logger.info(f"Done: {succeeded} account(s) succeeded, {failed} failed.")

    if not accounts_out:
        logger.error("No accounts were successfully logged in; refusing to emit an empty sessions blob.")
        sys.exit(1)

    return {
        "version": 1,
        "generated_at": now_iso,
        "accounts": accounts_out,
    }


###############################################################################
# Main
###############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate KAGGLE_SESSIONS_JSON (base64-encoded Playwright storage states) for the orchestrator."
    )
    parser.add_argument("--config", help="Path to a JSON file in the orchestrator config shape.")
    parser.add_argument(
        "--config-env",
        help="Name of an env var holding the same config, base64-encoded JSON (e.g. ORCHESTRATOR_CONFIG).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Playwright headless. Default is headed, so a human can clear CAPTCHAs interactively.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,  # keep stdout clean for the final base64 output
    )

    config = _load_config(args)
    accounts = _extract_accounts(config)
    logger.info(f"Loaded {len(accounts)} account(s) from config.")

    payload = _generate_sessions(accounts, headless=args.headless)

    try:
        OUTPUT_JSON_PATH.write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote un-encoded sessions to {OUTPUT_JSON_PATH} for inspection.")
    except Exception as e:
        logger.warning(f"Could not write {OUTPUT_JSON_PATH}: {e}")

    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

    # ONLY the final base64 string goes to stdout, per spec -- everything
    # else (logging, progress, warnings) has gone to stderr above.
    print(encoded)


if __name__ == "__main__":
    main()
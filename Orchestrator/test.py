#!/usr/bin/env python3

import os
import sys
import json
import base64
import binascii


CONFIG_ENV_VAR = "ORCHESTRATOR_CONFIG"


def load_config():
    raw = os.environ.get(CONFIG_ENV_VAR)
    if not raw:
        raise RuntimeError(f"{CONFIG_ENV_VAR} is not set")

    try:
        config = json.loads(base64.b64decode(raw, validate=True).decode("utf-8"))
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RuntimeError(f"Invalid {CONFIG_ENV_VAR}: {e}")

    return config


def purge_kaggle_modules():
    for name in list(sys.modules):
        if name.startswith(("kaggle", "kagglesdk")):
            del sys.modules[name]


def authenticate(account):
    token = account["kaggle_api_token"]

    os.environ["KAGGLE_API_TOKEN"] = token
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)

    purge_kaggle_modules()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def get(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return default


def main():
    config = load_config()

    if not config.get("accounts"):
        raise RuntimeError("No accounts found in config")

    account = config["accounts"][0]

    print(f"Using account: {account['kaggle_username']}")
    print("-" * 80)

    api = authenticate(account)

    kernels = api.kernels_list(mine=True, page_size=200)

    if not kernels:
        print("No kernels found.")
        return

    for i, k in enumerate(kernels, 1):
        ref = get(k, "ref", "kernelSlug", default="") or ""
        slug = ref.split("/")[-1] if "/" in ref else ref
        if not slug:
            slug = get(k, "slug", default="")

        title = get(k, "title", default=slug)
        status = get(k, "status", default="unknown")
        last_run = get(k, "lastRunTime", "last_run_time", "lastRunAt", default="-")

        print(f"[{i}]")
        print(f"Title    : {title}")
        print(f"Slug     : {slug}")
        print(f"Ref      : {ref}")
        print(f"Status   : {status}")
        print(f"Last Run : {last_run}")
        print()


if __name__ == "__main__":
    main()
from kaggle_quota_provider import KaggleQuotaProvider
import json
import os

def main() -> None:
    username = "kaggle1@avinash9.in"
    password = "REDACTED_PASSWORD"

    if not username or not password:
        raise SystemExit(
            "Set KAGGLE_USERNAME and KAGGLE_PASSWORD environment variables first."
        )

    with KaggleQuotaProvider() as scraper:
        quotas = scraper.login_and_scrape(username, password)

    print(json.dumps(
        {
            "quotaRefreshTime": quotas.quota_refresh_time,
            "tpuQuota": quotas.tpu_quota,
            "gpuQuota": quotas.gpu_quota,
        },
        indent=2,
    ))

if __name__ == "__main__":
    main()
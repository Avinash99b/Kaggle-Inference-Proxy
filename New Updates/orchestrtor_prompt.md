# PROMPT 3: Orchestrator (Account & Deployment Manager)

## Overview
You are building an **orchestrator service** that:
- Manages multiple Kaggle accounts (credentials from `config.json`)
- Queries GPU quota for each account (via `KaggleQuotaProvider`)
- Deploys inference servers to accounts (uploads notebook via Kaggle API)
- Tracks deployment status and notebook URLs
- Exposes REST API + WebSocket for real-time state updates
- Persists state to JSON file

**No reference code provided** — you're building from scratch based on requirements.

---

## Requirements

### 1. Account Management

#### Load Accounts from Config
- **File**: `orchestrator_config.json`
- **Structure**:
  ```json
  {
    "proxy_url": "https://proxy.example.com",
    "proxy_shared_secret": "secret-key",
    "accounts": [
      {
        "username": "kaggle1@example.com",
        "password": "password1",
        "kaggle_username": "kaggle_user_1"
      },
      {
        "username": "kaggle2@example.com",
        "password": "password2",
        "kaggle_username": "kaggle_user_2"
      }
    ],
    "hf_token": "optional-huggingface-token",
    "notebook_template_url": "https://raw.githubusercontent.com/.../inference_server_kaggle.py"
  }
  ```

#### Startup: Fetch Quota for Each Account
- On orchestrator startup:
  1. For each account in config:
     - Call `KaggleQuotaProvider().login_and_scrape(username, password)`
     - Extract GPU quota (total, used, remaining)
     - Store in memory: `AccountState`
  2. Persist to file: `orchestrator_state.json`
  3. Log: "Loaded N accounts with quota info"

#### Quota State Structure
```python
@dataclass
class AccountState:
    account_id: str                    # Unique ID (e.g., "kaggle_user_1")
    username: str
    password: str
    kaggle_username: str
    gpu_quota_total_seconds: int       # e.g., 108000 (30 hours)
    gpu_quota_used_seconds: int
    gpu_quota_remaining_seconds: int
    gpu_quota_refresh_time: str        # ISO timestamp
    last_quota_update: float           # time.time()
    deployment: Optional[DeploymentState]  # Current deployment (if any)
```

#### Quota Update Task
- **Background task**: Every 10 minutes, refresh quota for all accounts
- **Also on-demand**: When orchestrator starts or user requests
- **Update logic**:
  ```python
  async def refresh_quotas():
      for account in accounts.values():
          try:
              quota = await loop.run_in_executor(None, 
                  lambda: scraper.login_and_scrape(account.username, account.password))
              account.gpu_quota_total = quota.gpu_quota.totalTimeAllowed
              account.gpu_quota_used = quota.gpu_quota.timeUsed
              account.gpu_quota_remaining = account.gpu_quota_total - account.gpu_quota_used
              account.last_quota_update = time.time()
              logger.info(f"Updated quota for {account.account_id}: {account.gpu_quota_remaining}s remaining")
          except Exception as e:
              logger.warning(f"Failed to refresh quota for {account.account_id}: {e}")
  ```

### 2. Deployment Management

#### DeploymentState Structure
```python
@dataclass
class DeploymentState:
    deployment_id: str                 # Unique ID (e.g., "dep-2026-07-15-12345")
    account_id: str                    # Which account this runs on
    model_name: str                    # User-friendly name (e.g., "qwen3-8b")
    model_repo: str                    # HuggingFace repo (e.g., "Qwen/Qwen3-8B-GGUF")
    model_file: str                    # File in repo (e.g., "Qwen3-8B-Q8_0.gguf")
    notebook_id: str                   # Kaggle notebook ID (from API response)
    notebook_url: str                  # e.g., "https://www.kaggle.com/.../code/..."
    notebook_status: str               # "idle", "running", "error", "completed"
    worker_id: str                     # e.g., "kaggle-account-1" (same as account_id for now)
    created_at: float
    started_at: Optional[float]
    last_status_check: float
    error_message: Optional[str]
    quota_reserved_seconds: int        # How much quota we expect to use
```

#### Deploy Endpoint
- **`POST /api/deployments`**
- **Request**:
  ```json
  {
    "account_id": "kaggle_user_1",
    "model_repo": "Qwen/Qwen3-8B-GGUF",
    "model_file": "Qwen3-8B-Q8_0.gguf",
    "model_name": "qwen3-8b",
    "estimated_quota_hours": 2
  }
  ```
- **Validation**:
  - Account exists and is online
  - Account has enough quota (estimated_quota_hours * 3600 <= remaining)
  - Model file can be resolved (optional pre-check via HF API)
- **Action**:
  1. Create `DeploymentState`
  2. Generate notebook from template (see below)
  3. Upload notebook to account via Kaggle API
  4. Set status to "created" (notebook not yet running)
  5. Return deployment info + notebook URL
- **Response**:
  ```json
  {
    "deployment_id": "dep-2026-07-15-12345",
    "account_id": "kaggle_user_1",
    "notebook_url": "https://www.kaggle.com/.../code/...",
    "status": "created",
    "worker_id": "kaggle-account-1"
  }
  ```

#### Notebook Generation
- **Pre-built template**: Download from `notebook_template_url` (a .ipynb JSON file)
- **Template structure** (minimal):
  ```python
  # Cell 1: Configuration
  import os
  TARGET_MODEL = "{TARGET_MODEL}"
  PROXY_URL = "{PROXY_URL}"
  WORKER_ID = "{WORKER_ID}"
  WORKER_SECRET = "{WORKER_SECRET}"
  HF_TOKEN = "{HF_TOKEN}"
  
  os.environ.update({
      "TARGET_MODEL": TARGET_MODEL,
      "PROXY_URL": PROXY_URL,
      "WORKER_ID": WORKER_ID,
      "WORKER_SECRET": WORKER_SECRET,
      "HUGGINGFACE_TOKEN": HF_TOKEN,
  })
  
  # Cell 2: Download and run inference server
  import subprocess
  import os
  
  url = "{NOTEBOOK_TEMPLATE_URL}"
  subprocess.run(["wget", "-q", url, "-O", "inference_server_kaggle.py"], check=True)
  
  # Run as subprocess so it survives notebook restarts
  import subprocess
  subprocess.Popen(["python", "inference_server_kaggle.py"])
  ```
- **Generate**: Replace `{...}` placeholders, return as `.ipynb` JSON

#### Upload Notebook to Kaggle
- **Use Kaggle API**: `kaggle.api.notebooks_create()`
- **Notebook metadata**:
  ```python
  notebook = {
      "id": None,  # Kaggle auto-assigns
      "metadata": {
          "id": f"orchestrated-inference-{deployment_id}",
          "title": f"Orchestrated Inference: {model_name}",
          "language": "python",
          "kernelType": "python",
          "isGpu": True,
          "isPrivate": True,
          "enableGpu": True,
      },
      "cells": [...generated cells...],
  }
  ```
- **Response**: Extract notebook ID and URL

#### Deployment Status Polling
- **Background task**: Every 30 seconds, check status of all active deployments
- **Check logic**:
  1. Query Kaggle API for notebook status (via ID or slug)
  2. If status changed, update `DeploymentState` and notify via WebSocket
  3. If notebook is "running", mark deployment as "running"
  4. If notebook has errors, mark as "error" and extract error message
  5. Log status changes
- **Kaggle API call**: (TBD based on Kaggle SDK available methods)

#### Undeploy / Stop Deployment
- **`DELETE /api/deployments/{deployment_id}`**
- **Action**:
  1. Find deployment
  2. Stop/delete notebook via Kaggle API
  3. Mark deployment as "stopped"
  4. Free up quota reservation
  5. Persist state
- **Response**: `{"status": "stopped"}`

### 3. REST API Endpoints

All responses include proper status codes and error messages.

#### Accounts
- **`GET /api/accounts`** — List all accounts with current quota
  ```json
  {
    "accounts": [
      {
        "account_id": "kaggle_user_1",
        "username": "kaggle1@example.com",
        "gpu_quota_total_seconds": 108000,
        "gpu_quota_used_seconds": 10000,
        "gpu_quota_remaining_seconds": 98000,
        "gpu_quota_refresh_time": "2026-07-18T00:00:00Z",
        "last_quota_update": 1721030400.0,
        "has_deployment": false
      },
      ...
    ]
  }
  ```

#### Deployments
- **`GET /api/deployments`** — List all deployments
- **`GET /api/deployments/{deployment_id}`** — Get single deployment status
- **`POST /api/deployments`** — Create new deployment (see above)
- **`DELETE /api/deployments/{deployment_id}`** — Stop deployment
- **`POST /api/deployments/{deployment_id}/refresh-status`** — Force status check

#### Health
- **`GET /api/health`** — Orchestrator status
  ```json
  {
    "status": "ok",
    "accounts_online": 2,
    "deployments_running": 1,
    "deployments_idle": 0
  }
  ```

### 4. WebSocket Real-Time Updates

- **URL**: `GET /ws`
- **Messages to client** (server push):
  ```json
  {
    "event": "quota_update",
    "account_id": "kaggle_user_1",
    "gpu_quota_remaining_seconds": 98000,
    "timestamp": "2026-07-15T10:00:00Z"
  }
  
  {
    "event": "deployment_status_changed",
    "deployment_id": "dep-2026-07-15-12345",
    "status": "running",
    "notebook_url": "https://...",
    "timestamp": "2026-07-15T10:00:00Z"
  }
  
  {
    "event": "deployment_error",
    "deployment_id": "dep-2026-07-15-12345",
    "error": "CUDA OOM while loading model",
    "timestamp": "2026-07-15T10:00:00Z"
  }
  ```

### 5. State Persistence

- **File**: `orchestrator_state.json`
- **Structure**:
  ```json
  {
    "last_updated": 1721030400.0,
    "accounts": {
      "kaggle_user_1": {
        "account_id": "kaggle_user_1",
        "username": "...",
        "gpu_quota_total_seconds": 108000,
        "gpu_quota_used_seconds": 10000,
        "gpu_quota_remaining_seconds": 98000,
        "last_quota_update": 1721030400.0,
        "deployment": {...}
      },
      ...
    },
    "deployments": {
      "dep-2026-07-15-12345": {
        "deployment_id": "dep-2026-07-15-12345",
        "account_id": "kaggle_user_1",
        "model_repo": "Qwen/Qwen3-8B-GGUF",
        "model_file": "Qwen3-8B-Q8_0.gguf",
        "notebook_id": "abc123",
        "notebook_url": "https://...",
        "status": "running",
        "created_at": 1721030400.0,
        "started_at": 1721030420.0
      },
      ...
    }
  }
  ```
- **Load on startup**: Restore all accounts and deployments
- **Save periodically**: After quota updates, deployment changes

### 6. Model HuggingFace Integration

- **List files in repo**: When user enters model repo ID, list available GGUF files
  - **Endpoint**: `GET /api/models/list-files?repo={repo_id}`
  - **Returns**: `{"files": ["file1.gguf", "file2.gguf", ...]}`
  - **Use**: HuggingFace API (hf_hub_list_repo_files or similar)
  - **Caching**: Cache for 1 hour to avoid rate limits

---

## File Structure
**Two files**:
1. `orchestrator.py` (< 1000 lines) — Main service, FastAPI app, REST/WS endpoints
2. `orchestrator_config.json` — User configuration (accounts, proxy URL)

---

## Configuration Example

```json
{
  "proxy_url": "https://kaggle-inference-proxy.onrender.com",
  "proxy_shared_secret": "change-me-worker-secret",
  "orchestrator_port": 5000,
  "accounts": [
    {
      "username": "user1@example.com",
      "password": "password1",
      "kaggle_username": "user1"
    },
    {
      "username": "user2@example.com",
      "password": "password2",
      "kaggle_username": "user2"
    }
  ],
  "hf_token": "hf_xxxxxxxxxxxx",
  "notebook_template_url": "https://raw.githubusercontent.com/your-repo/main/inference_server_kaggle.py",
  "quota_refresh_interval_minutes": 10,
  "deployment_status_check_interval_seconds": 30
}
```

---

## Startup Flow

```
1. Load config.json
2. For each account:
   - Authenticate with Kaggle
   - Fetch GPU quota via KaggleQuotaProvider
   - Store in AccountState
3. Load or create orchestrator_state.json
4. Restore deployments from state file
5. For each deployment:
   - Query Kaggle notebook status
   - Update if changed
6. Start background tasks:
   - Quota refresh loop (every 10 min)
   - Deployment status check loop (every 30 sec)
7. Start FastAPI server
8. Listen for REST API + WebSocket connections
```

---

## Testing Checklist

- [ ] Loads config and fetches quota for all accounts
- [ ] Displays accounts sorted by remaining quota (ascending)
- [ ] User can select account and deploy model
- [ ] Generates notebook from template with correct placeholders
- [ ] Uploads notebook to Kaggle via API
- [ ] Notebook URL is trackable
- [ ] Background quota refresh updates in real-time
- [ ] Deployment status polling detects when notebook is running
- [ ] WebSocket sends quota_update events every 10 min
- [ ] WebSocket sends deployment status changes
- [ ] State persists to JSON file
- [ ] Orchestrator restarts and restores state correctly
- [ ] Can undeploy (stop notebook)
- [ ] REST API returns proper error codes
- [ ] HF file listing works for valid repos
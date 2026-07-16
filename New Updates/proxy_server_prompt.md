# PROMPT 2: Proxy Server (Multi-Inference-Server Router)

## Overview
You are building a **multi-worker proxy server** that:
- Manages **multiple inference servers** (one per Kaggle account), each running one model
- Routes client requests to the best-available server based on:
  1. Server has the requested model loaded
  2. Server with least-busy GPU (fewest in-flight jobs, lowest GPU utilization)
  3. Round-robin as tiebreaker
- Exposes standard OpenAI-compatible API endpoints
- Provides real-time server status via WebSocket and HTTP
- Handles job queuing, streaming, and error recovery

**Reference**: The provided `proxy_server.py` is your foundation. Adapt it for multiple workers.

---

## Requirements

### 1. Multi-Worker Registry
- **Track multiple workers**: `Dict[worker_id → WorkerState]` instead of single `WORKER`
- **WorkerState per worker**:
  ```python
  @dataclass
  class WorkerState:
      worker_id: str
      transport: str                    # "ws" or "poll"
      ws: Optional[WebSocket]           # if WS connected
      last_heartbeat: float
      models: Dict[str, Any]            # models this worker can serve
      status: str                       # "idle", "busy", "loading_model"
      current_model: Optional[str]      # what's currently loaded
      in_flight_jobs: int               # count of unfinished jobs
      gpu_stats: dict                   # {"total_mb": X, "used_mb": Y, "free_mb": Z}
      lock: asyncio.Lock
  ```

### 2. Request Routing Logic
**When a client sends `/v1/chat/completions` or `/v1/completions`:**

1. **Validate request**: Parse model, messages/prompt
2. **Find candidate workers**: Filter workers that have `model in worker.models.keys()`
3. **Score each candidate**:
   - `priority = (in_flight_jobs, gpu_utilization_percent, last_heartbeat_age)`
   - Lower score = better (fewest jobs, lowest GPU%, most recent heartbeat)
4. **Pick best worker**: Deterministic tie-break with round-robin counter
5. **Queue job to that worker**: Add job to that worker's queue (NOT global queue)
6. **Wait for result**: Return response to client when worker completes job

**If no workers have the model:**
- Return `503 Service Unavailable`: "No inference server currently has model 'X' loaded"

### 3. API Endpoints (Keep from Reference, Adapt for Multiple Workers)

#### Client Endpoints (unchanged)
- `GET /v1/models` — List all models across all online workers
- `POST /v1/chat/completions` — Chat completion (route via scoring logic)
- `POST /v1/completions` — Text completion (route via scoring logic)
- `POST /v1/embeddings` — Return 400 (not supported)
- `GET /health` — Return status of ALL workers
- `GET /dashboard` — HTML dashboard showing all workers

#### Worker Registration (adapt for multiple)
- `POST /worker/register` (long-poll)
- `GET /worker/poll` (long-poll)
- `POST /worker/result` (long-poll)
- `POST /worker/heartbeat` (long-poll)
- `POST /worker/deregister` (long-poll)
- `WebSocket /worker/ws` (primary)

### 4. Job Management
- **Per-worker job queues**: Each worker gets its own queue instead of global queue
- **Job structure**: Same as reference (id, kind, payload, future, etc.)
- **Job delivery**: Dispatcher waits for job to resolve before giving worker the next one
- **Timeout handling**: Watchdog still fires jobs that take too long, but now per-worker

### 5. WebSocket Worker Connection (Adapt from Reference)
- **Hello handshake**: Same (auth via HMAC, send models)
- **Dispatcher**: Pull from this worker's queue, send jobs one-at-a-time
- **Heartbeat messages**: Parse GPU stats (new field):
  ```json
  {
    "type": "heartbeat",
    "status": "idle",
    "current_model": "general-qwen3-8b",
    "gpu_stats": {
      "total_mb": 16384,
      "used_mb": 8192,
      "free_mb": 8192,
      "utilization_percent": 50
    }
  }
  ```
- **Result messages**: Same format (result, error, stream_chunk, stream_done)
- **Disconnect handling**: Mark that specific worker offline, requeue/fail its in-flight jobs

### 6. Long-Poll Worker Transport (Adapt from Reference)
- **Register**: Same (creates worker entry)
- **Poll**: Return next job from this worker's queue (not global queue)
- **Result**: Mark job complete for this worker
- **Heartbeat**: Update this worker's GPU stats
- **Deregister**: Mark offline, requeue/fail in-flight jobs

### 7. Streaming Support
- **True streaming** (WebSocket): Same as reference — stream_chunk messages pushed to client via SSE
- **Fake streaming** (long-poll): Still emit full result as single SSE chunk (no real incremental push)
- **Multi-worker context**: Streaming job MUST stay with same worker (no mid-stream switchover)

### 8. Load Balancing Scoring
```python
def score_worker(worker: WorkerState) -> tuple:
    """Lower score = better. Used for routing decisions."""
    utilization = worker.gpu_stats.get("utilization_percent", 0)
    in_flight = worker.in_flight_jobs
    recency = time.time() - worker.last_heartbeat
    return (in_flight, utilization, recency)
```

### 9. Health & Dashboard
- **`GET /health`**: Return status of ALL workers:
  ```json
  {
    "status": "ok",
    "timestamp": "2026-07-15T10:00:00Z",
    "workers": [
      {
        "worker_id": "kaggle-account-1",
        "online": true,
        "transport": "ws",
        "status": "idle",
        "current_model": "general-qwen3-8b",
        "in_flight_jobs": 2,
        "gpu_stats": {...},
        "known_models": ["general-qwen3-8b"],
        "last_heartbeat_age_seconds": 2.3
      },
      ...
    ]
  }
  ```
- **`GET /dashboard`**: HTML table showing all workers, their models, GPU usage, job counts

---

## Configuration

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "client_api_keys": ["sk-your-api-key-here"],
  "worker_shared_secret": "change-me-worker-secret",
  "request_timeout_seconds": 300,
  "long_poll_timeout_seconds": 40,
  "job_delivery_ack_timeout_seconds": 15,
  "worker_offline_after_seconds": 90,
  "auth_timestamp_skew_seconds": 300,
  "websocket": {
    "heartbeat_interval_seconds": 20,
    "heartbeat_timeout_seconds": 60
  },
  "retry": {
    "max_job_retries": 1
  },
  "routing": {
    "prefer_least_busy_gpu": true,
    "fallback_to_round_robin": true
  },
  "logging": {
    "level": "INFO",
    "file": "proxy_server.log"
  },
  "dashboard": {
    "enabled": true,
    "refresh_seconds": 5
  },
  "cors": {
    "enabled": true,
    "allow_origins": ["*"]
  }
}
```

---

## File Structure
**Single file**: `proxy_server_multi.py` (< 1500 lines)

---

## Key Adaptations from Reference

| Feature | Keep | Adapt | Remove |
|---------|------|-------|--------|
| Job queue structure | ✓ | Per-worker queues instead of global | Global job queue |
| Worker registry | ✓ | `Dict[worker_id → WorkerState]` | Single `WORKER` |
| WebSocket protocol | ✓ | Accept GPU stats in heartbeat | - |
| Long-poll protocol | ✓ | Per-worker queues | - |
| Streaming support | ✓ | Same | - |
| Auth (client & worker) | ✓ | Same | - |
| Job timeout watchdog | ✓ | Adapt for per-worker jobs | - |
| Scoring/routing | NEW | Implement least-busy-GPU scoring | Round-robin only |
| Dashboard | ✓ | Show all workers + GPU stats | Show single worker |

---

## Code Segments to Reuse

### From reference `proxy_server.py`:
1. **Job dataclass**: Keep as-is
2. **WorkerState dataclass**: Adapt to add `gpu_stats`, `in_flight_jobs`, per-worker queue
3. **Auth functions**: `check_client_key()`, `verify_worker_auth()`, etc. — unchanged
4. **OpenAI response formatting**: `format_chat_completion()`, etc. — unchanged
5. **Streaming generators**: `real_stream_generator()`, `fake_sse_stream()` — unchanged
6. **WebSocket hello/dispatch**: Adapt to route to specific worker
7. **Long-poll endpoints**: Adapt to route to specific worker
8. **Watchdog loop**: Adapt to iterate over all workers' in-flight jobs
9. **Main loop**: Adapt startup/shutdown

### Removals:
- Single global `WORKER` state
- Single global `JOBS` queue
- Worker "exactly one active" conflict detection (now allow many)

---

## Routing Algorithm (Pseudocode)

```
def route_request(model: str, payload: dict) -> Optional[WorkerState]:
    # Find all workers that have this model loaded
    candidates = [w for w in workers.values() 
                  if w.is_online() and model in w.models]
    
    if not candidates:
        return None  # No worker has this model
    
    # Score each candidate
    scores = [(score_worker(w), w) for w in candidates]
    scores.sort(key=lambda x: x[0])  # Sort by score (lower = better)
    
    # Pick first (best scorer)
    return scores[0][1]

def score_worker(w: WorkerState) -> tuple:
    return (
        w.in_flight_jobs,                    # Prefer fewer jobs
        w.gpu_stats.get("utilization_percent", 0),  # Prefer lower GPU%
        time.time() - w.last_heartbeat       # Prefer more recent heartbeat
    )
```

---

## Testing Checklist

- [ ] Starts with empty worker registry
- [ ] Accepts worker registration (WS and long-poll)
- [ ] Tracks multiple workers simultaneously
- [ ] Routes requests to worker with that model
- [ ] Picks least-busy worker when multiple have same model
- [ ] Returns 503 if no worker has model
- [ ] Accepts client requests (chat/completions)
- [ ] Returns proper OpenAI-compatible responses
- [ ] Streams tokens from workers to clients
- [ ] Handles worker disconnect and requeues jobs
- [ ] Watchdog times out stalled jobs
- [ ] Dashboard shows all workers and GPU stats
- [ ] `/health` returns multi-worker status
- [ ] Logs all routing decisions and job completions
# PROMPT 4: Dashboard UI (Orchestrator Frontend)

## Overview
You are building a **single-page web dashboard** that:
- Connects to the orchestrator REST API + WebSocket
- Displays Kaggle accounts, their GPU quota, and available quota
- Allows users to deploy models to accounts (with live model file suggestions)
- Shows active deployments, their status, and notebook URLs
- Displays real-time quota updates via WebSocket
- Provides one-click access to live notebooks

**Frontend stack**: React + TypeScript + Tailwind CSS

---

## Requirements

### 1. Layout & Screens

#### Screen 1: Accounts Overview (Default)
- **Header**: "Orchestrator Dashboard" + "Refresh Now" button
- **Content**:
  - Table of all accounts (sortable by remaining quota)
  - Columns:
    - Account Name (kaggle_username)
    - GPU Quota Total (e.g., "30h")
    - GPU Quota Used (e.g., "2.5h")
    - GPU Quota Remaining (e.g., "27.5h")
    - Status (green "Online" / red "Offline")
    - Actions: "Deploy Model" button
  - Color coding: Remaining quota bar (green → yellow → red as quota runs low)
  - Auto-refresh every 10 seconds (via polling or WebSocket)

#### Screen 2: Deployments Tab
- **Header**: "Active Deployments"
- **Content**:
  - List of all deployments (if any)
  - Each deployment card shows:
    - Model name (e.g., "qwen3-8b")
    - Account (e.g., "user1")
    - Status badge (running / stopped / error)
    - Notebook URL (clickable link)
    - Created at (timestamp)
    - Actions: "Stop" button
  - Empty state message if no deployments

#### Screen 3: Deploy Model Modal
- **Triggered by**: "Deploy Model" button on account row
- **Form fields**:
  1. **Selected Account** (read-only, pre-filled)
  2. **Model Repo ID** (text input with live suggestions)
     - On keystroke: Call `/api/models/list-files?repo={input}`
     - Show dropdown with `.gguf` files from repo
     - User selects one file
  3. **Model Name** (text input, optional label for this deployment)
  4. **Estimated Usage** (text input, hours)
     - Validation: "Account only has X hours available"
     - Warning if too close to quota limit
  5. **Deploy** button
  6. **Cancel** button

### 2. Real-Time Updates via WebSocket

- **Connection**: On page load, connect to `ws://orchestrator:5000/ws`
- **Listen for events**:
  ```json
  {
    "event": "quota_update",
    "account_id": "kaggle_user_1",
    "gpu_quota_remaining_seconds": 98000,
    "timestamp": "2026-07-15T10:00:00Z"
  }
  ```
  → Update quota display for that account immediately (no refresh needed)

  ```json
  {
    "event": "deployment_status_changed",
    "deployment_id": "dep-2026-07-15-12345",
    "status": "running",
    "notebook_url": "https://...",
    "timestamp": "2026-07-15T10:00:00Z"
  }
  ```
  → Update deployment card status, show notebook link (if "running")

  ```json
  {
    "event": "deployment_error",
    "deployment_id": "dep-2026-07-15-12345",
    "error": "CUDA OOM while loading model",
    "timestamp": "2026-07-15T10:00:00Z"
  }
  ```
  → Show error banner on deployment card, log full error

- **Reconnection**: Auto-reconnect with exponential backoff if WS drops

### 3. API Integration

#### Fetch Accounts (on load + every 10s)
```
GET /api/accounts

Response:
{
  "accounts": [
    {
      "account_id": "kaggle_user_1",
      "username": "user1@example.com",
      "gpu_quota_total_seconds": 108000,
      "gpu_quota_used_seconds": 10000,
      "gpu_quota_remaining_seconds": 98000,
      "gpu_quota_refresh_time": "2026-07-18T00:00:00Z",
      "last_quota_update": 1721030400.0,
      "has_deployment": false
    }
  ]
}
```

#### Deploy Model
```
POST /api/deployments

Request:
{
  "account_id": "kaggle_user_1",
  "model_repo": "Qwen/Qwen3-8B-GGUF",
  "model_file": "Qwen3-8B-Q8_0.gguf",
  "model_name": "qwen3-8b",
  "estimated_quota_hours": 2
}

Response:
{
  "deployment_id": "dep-2026-07-15-12345",
  "account_id": "kaggle_user_1",
  "notebook_url": "https://www.kaggle.com/.../code/...",
  "status": "created",
  "worker_id": "kaggle-account-1"
}
```

#### Fetch Deployments
```
GET /api/deployments

Response:
{
  "deployments": [
    {
      "deployment_id": "dep-2026-07-15-12345",
      "account_id": "kaggle_user_1",
      "model_repo": "Qwen/Qwen3-8B-GGUF",
      "model_file": "Qwen3-8B-Q8_0.gguf",
      "model_name": "qwen3-8b",
      "notebook_url": "https://...",
      "status": "running",
      "created_at": 1721030400.0,
      "started_at": 1721030420.0,
      "error_message": null
    }
  ]
}
```

#### List HF Files
```
GET /api/models/list-files?repo=Qwen/Qwen3-8B-GGUF

Response:
{
  "repo": "Qwen/Qwen3-8B-GGUF",
  "files": [
    "Qwen3-8B-Q8_0.gguf",
    "Qwen3-8B-Q4_K_M.gguf",
    "Qwen3-8B-Q5_K_M.gguf"
  ]
}
```

#### Stop Deployment
```
DELETE /api/deployments/dep-2026-07-15-12345

Response:
{
  "status": "stopped"
}
```

### 4. UI/UX Details

#### Quota Display
- Format remaining quota as human-readable: `27.5h` or `1649min`
- Progress bar: 0-100% (100% = fully used quota)
- Color:
  - Green: > 50% remaining
  - Yellow: 25-50% remaining
  - Red: < 25% remaining

#### Status Badges
- `running` → Green badge
- `stopped` → Gray badge
- `error` → Red badge with error icon
- `created` → Yellow/loading badge (notebook uploaded but not yet running)

#### Timestamps
- Format all timestamps as relative: "5 minutes ago", "2 hours ago"
- Hover to show full ISO timestamp

#### Error Handling
- Network errors: Show toast notification ("Failed to fetch accounts", retry button)
- Deployment errors: Show in red on deployment card, clickable to see full error
- Validation errors on deploy form: Inline error messages

#### Loading States
- Skeleton loaders for accounts table while fetching
- Spinner on "Deploy" button while uploading notebook
- Disable form fields while deployment is in progress

### 5. Component Structure

```
<App>
  ├── <Header />
  ├── <Navigation /> (tabs: Accounts, Deployments)
  ├── <AccountsTab>
  │   ├── <AccountsTable>
  │   │   ├── <AccountRow>
  │   │   │   └── <DeployButton />
  │   │   └── <RefreshButton />
  │   └── <QuotaVisualization />
  ├── <DeploymentsTab>
  │   └── <DeploymentsList>
  │       ├── <DeploymentCard>
  │       │   ├── <StatusBadge />
  │       │   ├── <NotebookLink />
  │       │   └── <StopButton />
  │       └── <EmptyState />
  ├── <DeployModal>
  │   ├── <AccountSelector />
  │   ├── <ModelRepoInput /> (with live suggestions)
  │   ├── <ModelFileDropdown />
  │   ├── <ModelNameInput />
  │   ├── <QuotaEstimateInput />
  │   ├── <SubmitButton />
  │   └── <ValidationErrors />
  ├── <WebSocketHandler /> (invisible, manages connection)
  └── <Toast /> (notifications)
```

### 6. Styling (Tailwind CSS)

- **Color scheme**: Dark theme (dark blue background, light text)
  - Primary: Blue-600
  - Success: Green-600
  - Error: Red-600
  - Warning: Yellow-600
- **Layout**: Responsive grid (1 col on mobile, 2+ on desktop)
- **Tables**: Striped rows, hover effects
- **Forms**: Standard input styling, focus states
- **Spacing**: Consistent padding/margin (4px units)

---

## File Structure
**Single file (for now)**: `src/App.tsx`

Or modularized:
```
src/
├── App.tsx (main entry)
├── components/
│   ├── AccountsTab.tsx
│   ├── DeploymentsTab.tsx
│   ├── DeployModal.tsx
│   ├── AccountRow.tsx
│   ├── DeploymentCard.tsx
│   └── ... other components
├── hooks/
│   ├── useApi.ts (REST calls)
│   ├── useWebSocket.ts (WebSocket connection)
│   └── useToast.ts (notifications)
├── types.ts (TypeScript interfaces)
└── index.tsx (React entry point)
```

---

## TypeScript Types

```typescript
interface Account {
  account_id: string;
  username: string;
  gpu_quota_total_seconds: number;
  gpu_quota_used_seconds: number;
  gpu_quota_remaining_seconds: number;
  gpu_quota_refresh_time: string;
  last_quota_update: number;
  has_deployment: boolean;
}

interface Deployment {
  deployment_id: string;
  account_id: string;
  model_repo: string;
  model_file: string;
  model_name: string;
  notebook_url: string;
  status: "created" | "running" | "stopped" | "error";
  created_at: number;
  started_at?: number;
  error_message?: string;
}

interface WebSocketEvent {
  event: "quota_update" | "deployment_status_changed" | "deployment_error";
  account_id?: string;
  deployment_id?: string;
  gpu_quota_remaining_seconds?: number;
  status?: string;
  notebook_url?: string;
  error?: string;
  timestamp: string;
}
```

---

## Key Features

### Auto-Refresh & Real-Time Updates
```typescript
// Polling fallback if WebSocket unavailable
useEffect(() => {
  const interval = setInterval(() => {
    fetchAccounts();
    fetchDeployments();
  }, 10000); // Every 10 seconds
  return () => clearInterval(interval);
}, []);

// WebSocket primary
useWebSocket(
  `ws://${orchestratorUrl}/ws`,
  (event: WebSocketEvent) => {
    if (event.event === "quota_update") {
      // Update accounts[account_id].gpu_quota_remaining_seconds
      updateAccountQuota(event.account_id, event.gpu_quota_remaining_seconds);
    }
    if (event.event === "deployment_status_changed") {
      updateDeploymentStatus(event.deployment_id, event.status);
    }
  }
);
```

### Live Model Suggestions
```typescript
const [repoInput, setRepoInput] = useState("");
const [suggestions, setSuggestions] = useState<string[]>([]);

useEffect(() => {
  if (repoInput.length > 2) {
    // Debounced API call
    const timer = setTimeout(async () => {
      const result = await fetch(`/api/models/list-files?repo=${repoInput}`);
      const data = await result.json();
      setSuggestions(data.files || []);
    }, 300);
    return () => clearTimeout(timer);
  }
}, [repoInput]);
```

### Quota Validation
```typescript
const validateQuotaEstimate = (hours: number, account: Account) => {
  const required_seconds = hours * 3600;
  const remaining_seconds = account.gpu_quota_remaining_seconds;
  
  if (required_seconds > remaining_seconds) {
    return {
      valid: false,
      error: `Not enough quota. Requested: ${hours}h, Available: ${remaining_seconds / 3600}h`
    };
  }
  if (required_seconds > remaining_seconds * 0.8) {
    return {
      valid: true,
      warning: `Using ${((required_seconds / remaining_seconds) * 100).toFixed(1)}% of remaining quota`
    };
  }
  return { valid: true };
};
```

---

## Deployment & Configuration

### Environment Variables
```bash
REACT_APP_ORCHESTRATOR_URL=http://localhost:5000
REACT_APP_API_BASE_URL=http://localhost:5000/api
REACT_APP_WS_URL=ws://localhost:5000/ws
```

### Build & Run
```bash
npm install
npm run build
npm run start

# Or Docker
docker build -t orchestrator-dashboard .
docker run -p 3000:3000 -e REACT_APP_ORCHESTRATOR_URL=... orchestrator-dashboard
```

---

## Testing Checklist

- [ ] Page loads and fetches accounts from API
- [ ] Accounts displayed in table, sorted by remaining quota
- [ ] "Deploy Model" button opens modal
- [ ] Model repo input shows file suggestions as you type
- [ ] Deploy form validates quota (shows warning/error)
- [ ] Deploy button creates deployment, shows notebook URL
- [ ] Deployments tab shows all active deployments
- [ ] Status badges update in real-time
- [ ] WebSocket updates quota display without page refresh
- [ ] WebSocket shows deployment status changes
- [ ] Error notifications appear for failed API calls
- [ ] Stop button removes deployment
- [ ] Notebook link is clickable
- [ ] Responsive on mobile (1 column)
- [ ] Dark theme applied
- [ ] Timestamps display as relative time
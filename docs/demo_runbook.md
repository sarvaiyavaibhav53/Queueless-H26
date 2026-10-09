# PRAVAAH — Government Queueless Demo Runbook

Complete step-by-step instructions for running, seeding, and demonstrating the PRAVAAH backend locally on Windows (PowerShell) or switching to PostgreSQL.

---

## 1. Prerequisites & Environment Setup

Open PowerShell and navigate to the repository root:

```powershell
cd C:\Users\vaibh\OneDrive\Desktop\h-26\pravaah
```

### Activate Virtual Environment
If `.venv` already exists:
```powershell
.\.venv\Scripts\Activate.ps1
```

If creating a fresh virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Environment Configuration

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

### Default Local Configuration (SQLite)
Edit `.env` (or let defaults apply):
```ini
# Default SQLite database for local demo
DATABASE_URL="sqlite:///./test.db"

# JWT Secret for authentication
JWT_SECRET="super-secret-key-for-development"
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Local CORS Allowlist (comma-separated origins)
CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"

# Congestion Queue Length Threshold (for fluid approximation, default: 15)
CONGESTION_QUEUE_THRESHOLD=15
```

---

## 3. Database Initialization & Idempotent Seeding

Run the idempotent database seed script to initialize tables, offices, services, physical counters, and synthetic demo users:

```powershell
python scripts/seed_data.py
```

### Seed Verification
The script is safe to run repeatedly. It ensures:
- **Offices:** Municipality (ID 1, `MUNI`), RTO (ID 2, `RTO`).
- **Services:** Property Tax (ID 1, `PT`), Water Connection (ID 2, `WC`), Birth Certificate (ID 3, `BC`), Ownership Transfer (ID 4, `OT`), Driving License Renewal (ID 5, `DR`).
- **Counters:** C1..C6 (Municipality), R1..R3 (RTO) with canonical service eligibility.
- **Synthetic Users:**
  - Admin: `admin@gov.in` / `Admin@123`
  - Employee: `employee@gov.in` / `Employee@123`
  - Citizen: `citizen@example.com` / `Citizen@123`

---

## 4. Starting the FastAPI Backend (Single Worker Required)

> [!IMPORTANT]
> The demo backend enforces in-process queue transition synchronization using process-local mutexes (`threading.Lock` / `asyncio.Lock`). Therefore, the backend **must be run with a single worker (`--workers 1`)**. Multi-worker concurrency guarantees are not supported for this process-local design.

Run the server using Uvicorn from the `backend` directory (or specify python path):

```powershell
$env:PYTHONPATH = "backend;$env:PYTHONPATH"
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1 --reload
```

Alternatively, from the repository root:
```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1 --reload
```

---

## 5. Health Check & Interactive Documentation

Once started, verify the API is responding:
- **Health Check:** [http://localhost:8000/health/db](http://localhost:8000/health/db)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI JSON:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 6. Frontend Dev Server Connection & CORS

The backend includes built-in CORS support for development servers:
- **React / Next.js:** `http://localhost:3000` / `http://127.0.0.1:3000`
- **Vite:** `http://localhost:5173` / `http://127.0.0.1:5173`
- **Preview / Alternative:** `http://localhost:8080`, `http://localhost:4173`

> [!IMPORTANT]
> Do NOT open the frontend directly from `file://` (which sends `Origin: null`). Always run through `npm run dev` or equivalent development server.

---

## 7. Scenario Loading & End-to-End Demo Workflow

### List Available Scenarios
```powershell
curl -X GET "http://localhost:8000/api/v1/admin/scenarios" `
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

### Load Canonical Scenario E (`E_known` / Counter Failure at t=0)
Resets Office 1 (Municipality) demo state and loads counter outage with backdated waiting tokens:
```powershell
curl -X POST "http://localhost:8000/api/v1/admin/offices/1/scenarios/load" `
  -H "Authorization: Bearer <ADMIN_TOKEN>" `
  -H "Content-Type: application/json" `
  -d '{\"scenario_id\": \"E_known\"}'
```

### Get Command Center Operations Summary
Inspect queue depth, active/busy/unavailable counters, $\mu$, $\rho$, stability classification, and fluid time-to-congestion:
```powershell
curl -X GET "http://localhost:8000/api/v1/admin/offices/1/summary" `
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

### Evaluate & Recommend Interventions
Simulate 60-minute look-ahead with discrete-event simulation:
```powershell
curl -X POST "http://localhost:8000/api/v1/interventions/evaluate" `
  -H "Authorization: Bearer <ADMIN_TOKEN>" `
  -H "Content-Type: application/json" `
  -d '{\"office_id\": 1, \"horizon_minutes\": 60, \"replications\": 5}'
```

### Apply Counter Reassignment
Reassign Counter 3 (donor Water Connection) to Property Tax:
```powershell
curl -X POST "http://localhost:8000/api/v1/interventions/apply" `
  -H "Authorization: Bearer <ADMIN_TOKEN>" `
  -H "Content-Type: application/json" `
  -d '{\"office_id\": 1, \"counter_id\": 3, \"target_service_id\": 1}'
```
*Note: Reassignment succeeds immediately even if Counter 3 is currently serving a token. The active token completes under Water Connection, and subsequent calls will serve Property Tax.*

### Restore Counter to Home Service
Revert Counter 3 back to canonical home service (Water Connection):
```powershell
curl -X POST "http://localhost:8000/api/v1/interventions/restore" `
  -H "Authorization: Bearer <ADMIN_TOKEN>" `
  -H "Content-Type: application/json" `
  -d '{\"office_id\": 1, \"counter_id\": 3}'
```

---

## 8. Database Environment (SQLite Demo vs PostgreSQL)

> [!NOTE]
> **Hackathon Demo Database:** SQLite (`test.db`) is the officially supported, tested, and verified database for the local hackathon demonstration. All deterministic tests, scenario resets, and rehearsals are validated on SQLite.
>
> While models and queries use standard SQLAlchemy constructs intended to remain syntax-compatible with PostgreSQL, **PostgreSQL runtime validation has not been exercised** in this test environment. Multi-process/PostgreSQL concurrency guarantees must not be claimed.

If connecting to a PostgreSQL instance in a future production phase:
1. Create the database:
   ```sql
   CREATE DATABASE queueless_db;
   ```
2. Update `DATABASE_URL` in `.env`:
   ```ini
   DATABASE_URL="postgresql+psycopg://postgres:YourPassword@localhost:5432/queueless_db"
   ```
3. Run the idempotent seed script against PostgreSQL:
   ```powershell
   python scripts/seed_data.py
   ```
4. Start the backend with a single worker:
   ```powershell
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1 --reload
   ```

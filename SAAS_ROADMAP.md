# Position Tracker SaaS — Roadmap

> Persistent reference file for cross-session development planning.
> Last updated after Phase 2 completion (78 tests passing).

---

## Architecture Overview

- **Stack**: Python 3.14, FastAPI 0.115+, SQLAlchemy 2.0 async, PostgreSQL 16, Celery 5.4+ / Redis, Jinja2 templates
- **Frontend**: Vanilla JS, custom CSS (700+ lines), Chart.js 4.4.7 CDN, JWT in localStorage
- **Auth**: JWT HS256 (30min access + 30d refresh), Argon2 hashing, `apiFetch()` with auto-refresh on 401
- **Multi-tenancy**: UUID `tenant_id` on all tables, CASCADE delete, enforced in all queries
- **RBAC**: viewer(0) < member(1) < admin(2) < owner(3), `require_role()` dependency
- **Partitioning**: PostgreSQL RANGE by date (monthly), auto-creation on 1st of month (03:00 UTC)
- **Data Retention**: Auto-drop partitions older than 24 months on 2nd of month (04:00 UTC)
- **CI/CD**: GitHub Actions (lint + tests on push/PR)
- **Rate Limiting**: slowapi per-endpoint, configurable via settings

---

## Completed Phases

### Phase 0 — Foundation (commit `93805d0`)
- FastAPI app structure with async SQLAlchemy
- PostgreSQL models: Tenant, User, Project, Keyword, SerpSnapshot, WebmasterData
- Monthly partition auto-creation (PL/pgSQL + Celery Beat)
- JWT auth (register, login, refresh, /me)
- Projects CRUD + Keywords CRUD with pagination
- SERP API (chart data, keyword positions, URL keywords, SERP details)
- Webmaster/GSC data API
- Dashboard API (summary stats)
- Celery task dispatcher + JustMagic collector
- Jinja2 frontend: login, dashboard, project detail, settings
- Docker Compose (app + postgres + redis + celery worker/beat + nginx)
- **Tests**: 34 passing

### Phase 1 — SaaS MVP (commit `3a1abb8`)
- Plan limits system (max_projects, max_keywords per plan tier: free/starter/pro/enterprise)
- User management API (invite with temp password, change role, deactivate, tenant isolation)
- API Keys API (create, list, delete, authenticate via X-API-Key header, tenant-scoped)
- Data export API (CSV export with date filters, keyword/project filtering)
- **Tests**: 64 passing (+30 new)

### Phase 2 — Frontend + Data Retention + Multi-Region (current)
- Account API (GET/PATCH profile with tenant info, change password)
- Regions API (static reference data: ~47 Yandex + ~46 Google regions)
- Data retention: PL/pgSQL `drop_old_partitions()` + Celery task (24-month default)
- 4 new frontend pages: Users, API Keys, Export, Account
- Navbar updated with all page links
- Settings page: region dropdown selectors (replacing manual code input)
- **Tests**: 78 passing (+14 new)

---

## SaaS Readiness: ~70%

| Category | Status | Notes |
|----------|--------|-------|
| Multi-tenancy | DONE | UUID tenant_id, CASCADE, query-level isolation |
| Auth + RBAC | DONE | JWT, Argon2, 4 roles, require_role() |
| Plan limits | DONE | 4 tiers, project + keyword limits |
| User management | DONE | Invite, role change, deactivate, API + UI |
| API keys | DONE | Create, auth via header, tenant-scoped, API + UI |
| Data export | DONE | CSV with filters, API + UI |
| Account management | DONE | Profile, password change, API + UI |
| Multi-region | DONE | Dropdown selection, Yandex + Google regions |
| Data retention | DONE | 24-month auto-cleanup, PL/pgSQL + Celery |
| Rate limiting | DONE | slowapi per-endpoint |
| CI/CD | DONE | GitHub Actions |
| Billing/Payments | NOT STARTED | Stripe integration needed |
| Email notifications | NOT STARTED | SMTP/SendGrid for invites, alerts |
| Audit log | NOT STARTED | Track user actions |
| Onboarding wizard | NOT STARTED | First-time setup flow |
| Tenant branding | NOT STARTED | Custom logo, colors |
| 2FA / SSO | NOT STARTED | TOTP or OAuth providers |
| Webhooks | NOT STARTED | Event notifications to external services |
| Public API docs | NOT STARTED | OpenAPI/Swagger UI polish |
| Admin panel | NOT STARTED | Super-admin cross-tenant management |
| Monitoring/Alerts | NOT STARTED | Health checks, error tracking (Sentry) |

---

## Phase 3 — Billing + Email (planned)

**Goal**: Revenue-ready. Users can sign up, pick a plan, pay, and get email notifications.

### 3.1 Stripe Integration
- `Tenant.stripe_customer_id`, `Tenant.stripe_subscription_id` columns
- Webhook endpoint for subscription events (created, updated, cancelled, payment_failed)
- Plan upgrade/downgrade with proration
- Billing page in frontend (current plan, usage, payment method, invoices)
- Stripe Checkout for new subscriptions
- Graceful degradation on payment failure (read-only mode, not data deletion)

### 3.2 Email Service
- Abstract email backend (SMTP + SendGrid adapter)
- Transactional emails: invite user (with temp password), password reset, payment receipt
- Email verification on registration
- Alert emails: collection failures, approaching plan limits
- Unsubscribe support

### 3.3 Password Reset Flow
- `POST /auth/forgot-password` — send reset link via email
- `POST /auth/reset-password` — validate token, set new password
- Time-limited tokens (1 hour)

**Estimate**: ~15 new files, ~10 modified files, ~20 new tests

---

## Phase 4 — Security Hardening + Monitoring (planned)

### 4.1 Two-Factor Authentication
- TOTP-based 2FA (Google Authenticator / Authy)
- `User.totp_secret`, `User.totp_enabled` columns
- Setup flow: generate secret → show QR → verify code → enable
- Login flow: if 2FA enabled, require code after password
- Recovery codes (one-time use)

### 4.2 Audit Log
- `audit_log` table: tenant_id, user_id, action, entity_type, entity_id, details (JSONB), ip, timestamp
- Middleware to auto-capture API mutations (POST/PUT/PATCH/DELETE)
- Audit log page in frontend with filters (user, action, date range)
- Retention: same 24-month policy as data tables

### 4.3 Monitoring & Alerting
- Health check endpoint (`GET /health` — DB, Redis, Celery connectivity)
- Sentry integration for error tracking
- Prometheus metrics endpoint (optional)
- Uptime monitoring webhook support

**Estimate**: ~12 new files, ~8 modified files, ~18 new tests

---

## Phase 5 — Advanced Features (planned)

### 5.1 Webhooks
- `webhooks` table: tenant_id, url, events[], secret, is_active
- CRUD API for webhook management
- Async delivery with retry (3 attempts, exponential backoff)
- HMAC signature verification
- Event types: position_changed, collection_completed, keyword_added, user_invited

### 5.2 Onboarding Wizard
- First-login detection (check if tenant has 0 projects)
- Step-by-step: create project → add keywords → configure regions → run first collection
- Progress tracking, skip option
- Welcome email after completion

### 5.3 Public API Documentation
- OpenAPI schema cleanup (descriptions, examples, error responses)
- Swagger UI customization (branding, auth button)
- API versioning strategy (v1 stable, v2 beta)
- Rate limit headers in responses (X-RateLimit-*)
- SDK generation considerations (Python, JS)

### 5.4 Admin Panel
- Super-admin role (cross-tenant access)
- Tenant list with stats (users, projects, keywords, last collection)
- Impersonate user (for support)
- System-wide stats dashboard
- Manual plan override

**Estimate**: ~20 new files, ~12 modified files, ~30 new tests

---

## Phase 6 — Scale + Polish (planned)

### 6.1 Performance
- ClickHouse integration for SERP history queries (read path)
- Redis caching for dashboard stats (TTL 5min)
- Background job queue monitoring (Flower or custom)
- Database connection pooling tuning
- Query optimization (EXPLAIN ANALYZE audit)

### 6.2 Tenant Branding
- `Tenant.logo_url`, `Tenant.primary_color`, `Tenant.custom_domain`
- Custom domain support with SSL (Let's Encrypt)
- White-label login page

### 6.3 Collaboration
- Project-level permissions (not just tenant-level)
- Shared reports (public link with token)
- Comments on keyword positions
- Team activity feed

### 6.4 Integrations
- Google Search Console OAuth (replace manual GSC URL input)
- Yandex Webmaster OAuth (replace manual host ID input)
- Slack notifications
- Telegram bot for alerts
- Zapier/Make webhook compatibility

---

## File Structure Reference

```
app/
  api/v1/
    router.py          # All API routers registered here
    auth.py            # Register, login, refresh, /me
    projects.py        # Projects CRUD
    keywords.py        # Keywords CRUD with pagination
    serp.py            # SERP chart, positions, URL keywords
    webmaster.py       # GSC/YWM data API
    dashboard.py       # Summary stats
    users.py           # User management (invite, role, deactivate)
    api_keys.py        # API keys CRUD + auth
    export.py          # CSV data export
    account.py         # Profile + password change
    regions.py         # Static region reference data
    settings.py        # Schedule configuration
  core/
    config.py          # All settings (DB, Redis, JWT, plans, retention)
    security.py        # JWT + Argon2
    dependencies.py    # Auth dependencies, require_role()
    exceptions.py      # HTTP exceptions
    middleware.py       # API key auth middleware
  models/              # SQLAlchemy models (tenant, user, project, keyword, serp, webmaster)
  schemas/             # Pydantic schemas
  data/
    regions.py         # YANDEX_REGIONS + GOOGLE_REGIONS static lists
  tasks/
    celery_app.py      # Celery config + beat schedule
    dispatcher.py      # Tenant collection dispatcher
    maintenance_tasks.py  # Partition creation + retention cleanup
  collectors/
    justmagic.py       # JustMagic SERP collector
  web/
    router.py          # Jinja2 HTML page routes
templates/             # Jinja2 templates (base, login, dashboard, project, settings, users, api_keys, export, account)
static/                # CSS + JS (style.css, charts.js)
tests/                 # 78 tests
alembic/               # Database migrations
```

---

## Key Configuration (app/core/config.py)

```
DATABASE_URL          # PostgreSQL async connection
REDIS_URL             # Redis for Celery broker + result backend
SECRET_KEY            # JWT signing key
JUSTMAGIC_API_KEY     # External SERP data provider
ALLOWED_ORIGINS       # CORS origins
RETENTION_MONTHS=24   # Data retention period
PLAN_*                # Plan limits (free/starter/pro/enterprise)
```

---

## Testing

```bash
# Lint
.venv/bin/ruff check .

# Format
.venv/bin/ruff format .

# Run all tests
.venv/bin/pytest tests/ -v

# Run specific test file
.venv/bin/pytest tests/test_account.py -v

# Current: 78 tests, all passing
```

---

## Git History

| Commit | Description | Tests |
|--------|-------------|-------|
| `93805d0` | Initial commit: Position Tracker SaaS v2.0.0 | 34 |
| `87e572a` | GitHub Actions CI + lint fixes | 34 |
| `22dc3e8` | Security: RBAC, rate limiting, startup validation | 34 |
| `3a1abb8` | SaaS Phase 1: plan limits, user mgmt, API keys, export | 64 |
| (pending) | Phase 2: frontend pages, data retention, multi-region | 78 |

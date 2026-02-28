# LLM Visibility Tracker — Plan transformacii

> Detalniy plan preobrazovaniya Position Tracker SaaS v SaaS po otslezhivaniyu vidimosti v LLM i AI Overview.
> Data sozdaniya: 2026-02-17

---

## Koncepciya produkta

**Chto delaem**: SaaS dlya monitoringa vidimosti brenda/sayta v otvetax LLM (ChatGPT, Gemini, Claude, Perplexity) i v blokax AI Overview Google.

**Klyuchevye metriki dlya polzovatelya**:
- Upominaetsya li brend v otvetax LLM na celevye zaprosy
- Kakie URL citiruet LLM
- Share of Voice — dolya upominaniy vs konkurenty
- Dinamika vidimosti vo vremeni (trendi)
- AI Overview v Google — prisutstvuet li sayt v bloke

**Kak sobirat dannye**: Pryamye zaprosy k API LLM-provayderov (OpenAI, Google, Anthropic) s analiticheskoy obrabotkoy otvetov. Parsing AI Overview cherez SERP-parsing.

---

## Tekuschee sostoyanie proekta

Bazovaya infrastruktura ot Position Tracker SaaS:
- Multi-tenant arxitektura (UUID, RBAC, JWT)
- PostgreSQL 16 + ClickHouse + Redis + Celery
- 13 API-moduley, 8 frontend-stranic
- Sboschiki: JustMagic (SERP), YWM, GSC
- 78 testov, CI/CD, Docker Compose

**Chto ostaetsya kak est** (ispolzuem bazovuyu infrastrukturu):
- Autentifikaciya, RBAC, multi-tenancy
- Celery + Beat (task scheduling)
- Particionirovanie, retenciya dannyh
- Plan limits, user management, API keys
- Frontend-karkas (base.html, navbar, auth)
- Docker, CI/CD, health checks

---

## Faza L1 — Modeli dannyh i bazovaya infrastruktura LLM

**Cel**: Sozdat tablicy, shemy, konfiguracii dlya LLM-monitoringa.

### L1.1 Novye modeli (PostgreSQL)

**Tablica `llm_queries`** — zaprosy (prompty) dlya monitoringa:
```
id              SERIAL PK
tenant_id       UUID FK → tenants (CASCADE)
project_id      INT FK → projects (CASCADE)
query_text      VARCHAR(2000)     — tekst zaprosa / prompta
query_type      VARCHAR(20)       — "brand_check" | "comparison" | "recommendation" | "custom"
target_brand    VARCHAR(255)      — brend kotoriy ischshem (naprimer "Ahrefs")
competitors     JSONB             — ["SEMrush", "Moz", "Serpstat"]
is_active       BOOLEAN DEFAULT true
created_at      TIMESTAMPTZ
```
- Unique: (project_id, query_text)
- Eto zamena `keywords` dlya LLM-konteksta, no keywords ostaetsya dlya SERP

**Tablica `llm_snapshots`** — rezultat proverki LLM (1 zapis = 1 zapros × 1 provider × 1 data):
```
id              BIGSERIAL PK
tenant_id       UUID
llm_query_id    INT FK → llm_queries (CASCADE)
date            DATE              — data proverki
llm_provider    VARCHAR(20)       — "chatgpt" | "gemini" | "claude" | "perplexity"
llm_model       VARCHAR(50)       — "gpt-4o" | "gemini-2.0-flash" | "claude-sonnet-4-5" ...
brand_mentioned BOOLEAN           — upomyanut li target_brand
mention_type    VARCHAR(20)       — "direct" | "recommended" | "compared" | "negative" | "none"
mention_context TEXT              — fragment otveta gde upomyanut brend (do 500 simvolov)
competitor_mentions JSONB         — {"SEMrush": true, "Moz": false}
cited_urls      JSONB             — ["https://...", "https://..."] — URL kotorye LLM ukazyvaet kak istochniki
raw_response    TEXT              — polnyy otvet LLM (dlya re-analiza)
response_tokens INT               — skolko tokenov v otvete
cost_usd        FLOAT             — stoimost zaprosa v dollarax
collected_at    TIMESTAMPTZ
```
- Unique: (llm_query_id, date, llm_provider)
- Partitioned: RANGE by date (mesyachno, kak serp_snapshots)

**Tablica `ai_overview_snapshots`** — AI Overview v Google SERP:
```
id              BIGSERIAL PK
tenant_id       UUID
keyword_id      INT FK → keywords (CASCADE)   — ispolzuem suschsestvuyushsuyu tablicu keywords
date            DATE
has_ai_overview BOOLEAN           — est li blok AI Overview v vydache
brand_in_aio    BOOLEAN           — est li nash brend v AIO
aio_position    SMALLINT          — poziciya nashego sayta v istochnikax AIO (1,2,3...)
aio_sources     JSONB             — [{"url": "...", "domain": "...", "title": "..."}]
aio_snippet     TEXT              — tekst bloka AI Overview (do 2000 simvolov)
collected_at    TIMESTAMPTZ
```
- Unique: (keyword_id, date)
- Partitioned: RANGE by date

### L1.2 Rasshirenie suschestvuyuschix modeley

**Tablica `tenants`** — dobavit novye encrypted polya:
```
openai_api_key          LargeBinary NULLABLE   — API klyuch OpenAI
google_ai_api_key       LargeBinary NULLABLE   — API klyuch Google AI (Gemini)
anthropic_api_key       LargeBinary NULLABLE   — API klyuch Anthropic
perplexity_api_key      LargeBinary NULLABLE   — API klyuch Perplexity
```

**Tablica `projects`** — dobavit:
```
track_llm           BOOLEAN DEFAULT false      — otslezhivat LLM vidimostx
track_ai_overview   BOOLEAN DEFAULT false      — otslezhivat AI Overview
llm_providers       JSONB DEFAULT '["chatgpt"]' — kakie provaidery ispoltsovatx
brand_name          VARCHAR(255) NULLABLE      — imya brenda dlya monitoringa
```

**Plan limits** — dobavit v config.py:
```
max_llm_queries per plan:  free=50, starter=500, pro=5000, enterprise=unlimited
max_llm_checks_per_day:    free=10, starter=100, pro=1000, enterprise=5000
```

### L1.3 Alembic-migracii

- Migraciya dlya novyx tablic (llm_queries, llm_snapshots, ai_overview_snapshots)
- Migraciya dlya novyx kolonok v tenants i projects
- Sozdanie particiy dlya llm_snapshots i ai_overview_snapshots
- Obnovlenie PL/pgSQL funkciy create_monthly_partitions() / drop_old_partitions()

### L1.4 Pydantic-shemy

- `LlmQueryCreate`, `LlmQueryUpdate`, `LlmQueryResponse`
- `LlmSnapshotResponse`, `LlmSnapshotChartData`
- `AiOverviewSnapshotResponse`
- Obnovlenie `ProjectCreate` / `ProjectUpdate` (novye polya)
- Obnovlenie `SettingsCredentials` (novye API-klyuchi)

**Ocenka**: ~8 novyx faylov, ~6 izmenennyx, ~15 novyx testov

---

## Faza L2 — LLM Kollektory (yadro produkta)

**Cel**: Realizovatx kollektory kotorye otpravlyayut zaprosy v LLM API i analiziruyut otvety.

### L2.1 Bazoviy LLM kollektor

`app/collectors/llm_base.py`:
```python
class BaseLlmCollector(BaseCollector):
    """Bazoviy klass dlya vsex LLM kollektorov."""

    async def collect(self, *, db, llm_query, ...) -> CollectionResult

    def analyze_response(self, response_text, target_brand, competitors) -> AnalysisResult:
        """Analizirovat otvet LLM:
        - brand_mentioned: bool
        - mention_type: direct/recommended/compared/negative/none
        - mention_context: fragment teksta
        - competitor_mentions: dict
        - cited_urls: list
        """
```

Analiz otveta — **ne cherez LLM**, a cherez proverennye evristiki:
1. Poisk brenda v tekste (case-insensitive, word boundary)
2. Klassifikaciya konteksta upominaniya po okruzhayuschim slovam
3. Poisk URL/domen v tekste
4. Poisk upominaniy konkurentov

### L2.2 OpenAI Kollektor (ChatGPT)

`app/collectors/llm_openai.py`:
```python
class OpenAiCollector(BaseLlmCollector):
    """ChatGPT API collector."""

    # Ispolzuem: openai.ChatCompletion.create()
    # Model: gpt-4o (nastroivayetsya)
    # Temperature: 0.0 (dlya vosproizvodimosti)
    # Max tokens: 2048
    # System prompt: "You are a helpful assistant. Answer the following question."
```

Osobennosti:
- Podderzhka raznyx modeley (gpt-4o, gpt-4o-mini, gpt-4.1)
- Tracking tokenov i stoimosti
- Rate limiting (ne bolee N zaprosov v minutu)
- Retry s exponential backoff

### L2.3 Google Gemini Kollektor

`app/collectors/llm_gemini.py`:
```python
class GeminiCollector(BaseLlmCollector):
    """Google Gemini API collector."""

    # Ispolzuem: google.generativeai (official SDK)
    # Model: gemini-2.0-flash (nastroivayetsya)
    # Gemini mozhet vozvraschatx citaty/istochniki — parsx ix
```

Osobennosti:
- Gemini imeet grounding / search — esli vklyuchen, parser cited sources
- Podderzhka modelej: gemini-2.0-flash, gemini-2.5-pro

### L2.4 Anthropic Kollektor (Claude)

`app/collectors/llm_anthropic.py`:
```python
class AnthropicCollector(BaseLlmCollector):
    """Anthropic Claude API collector."""

    # Ispolzuem: anthropic SDK
    # Model: claude-sonnet-4-5 (nastroivayetsya)
    # max_tokens: 2048
```

### L2.5 Perplexity Kollektor

`app/collectors/llm_perplexity.py`:
```python
class PerplexityCollector(BaseLlmCollector):
    """Perplexity API collector."""

    # Ispolzuem: Perplexity Sonar API (OpenAI-compatible format)
    # Klyuchevaya osobennostx: vsegda vozvraschaet citations[]
    # Parser citations kak cited_urls
```

Perplexity osobenno vazhen — on vsegda daet istochniki.

### L2.6 AI Overview Kollektor

`app/collectors/ai_overview.py`:
```python
class AiOverviewCollector(BaseCollector):
    """Kollektor AI Overview iz Google SERP."""

    # Varianti realizacii (v poryadke prioriteta):
    # 1. Cherez JustMagic API (esli oni parsyat AIO)
    # 2. Cherez SerpAPI / DataForSEO / Oxylabs
    # 3. Cherez Google Custom Search API (ogranicheno)

    # Parsim:
    # - Nalichie bloka AI Overview
    # - Tekst otveta
    # - Istochniki (URL, title, domain)
    # - Estx li nash domen sredi istochnikov
```

**Vybor istochnika dannyh dlya AI Overview**:
- Preferably: JustMagic (esli dobavyat podderzhku AIO parsing)
- Alternativa: SerpAPI (platno, no nadezhnoe API dlya AI Overview)
- Zapasnoj variant: sobstvennyj parsing cherez headless browser (slozhnee)

### L2.7 Integratsiya v collection_service.py

Rasshiritx `collect_project()`:
```python
# --- LLM Collection ---
if project.track_llm:
    for provider in project.llm_providers:
        api_key = get_llm_api_key(tenant, provider)
        if api_key:
            collector = get_llm_collector(provider, api_key, tenant_id)
            result.llm[provider] = await collector.collect(
                db=db, project_id=project.id, ...
            )

# --- AI Overview ---
if project.track_ai_overview:
    result.ai_overview = await ai_overview_collector.collect(
        db=db, project_id=project.id, keywords=keywords, ...
    )
```

**Ocenka**: ~10 novyx faylov, ~5 izmenennyx, ~25 novyx testov

---

## Faza L3 — API endpointy i biznes-logika

**Cel**: REST API dlya upravleniya LLM-zaprosami i polucheniya rezultatov.

### L3.1 LLM Queries API

`app/api/v1/llm_queries.py`:
```
GET    /api/v1/projects/{id}/llm-queries/           — spisok zaprosov (paginated)
POST   /api/v1/projects/{id}/llm-queries/           — dobavitx zaprosy (bulk)
POST   /api/v1/projects/{id}/llm-queries/single     — dobavitx odin zapros
PUT    /api/v1/projects/{id}/llm-queries/{qid}      — obnovitx
DELETE /api/v1/projects/{id}/llm-queries/{qid}      — udalitx
```

### L3.2 LLM Results API

`app/api/v1/llm_results.py`:
```
GET    /api/v1/llm/dashboard/{project_id}           — obschaya statistika:
                                                       - brand_mention_rate po provaideram
                                                       - SOV vs konkurenty
                                                       - top citiruemyh URL
                                                       - trend za period

GET    /api/v1/llm/chart/{project_id}               — dannye dlya grafikov:
                                                       - mention_rate po dnyam
                                                       - SOV po dnyam
                                                       - filtry: provider, days, query_id

GET    /api/v1/llm/details/{project_id}             — detalnye otvety LLM:
                                                       - polnyy otvet, kontekst upominaniya
                                                       - filxtr po query, provider, date

GET    /api/v1/llm/citations/{project_id}           — agregirovannye citaty:
                                                       - kakie URL chashse vsego citiruyutsya
                                                       - gruppirovka po domenu
```

### L3.3 AI Overview API

`app/api/v1/ai_overview.py`:
```
GET    /api/v1/ai-overview/dashboard/{project_id}   — statistika AIO:
                                                       - % zaprosov s AIO blokom
                                                       - % gde nash sayt v AIO
                                                       - trend

GET    /api/v1/ai-overview/chart/{project_id}       — grafiki AIO po dnyam

GET    /api/v1/ai-overview/details/{project_id}     — detali po kazhdomu keyword:
                                                       - est li AIO, nasha poziciya, istochniki
```

### L3.4 Collection trigger

Obnovitx `app/api/v1/collection.py`:
```
POST   /api/v1/collection/projects/{id}/llm         — zapustitx LLM-sborku vruchnuyu
POST   /api/v1/collection/projects/{id}/ai-overview  — zapustitx AIO-sborku
```

### L3.5 Settings API

Obnovitx `app/api/v1/settings.py`:
- Dobavitx nastrojki LLM API-klyuchey (openai, gemini, anthropic, perplexity)
- Dobavitx nastrojku modelej i parametrov

### L3.6 Export

Obnovitx `app/api/v1/export.py`:
- CSV export dlya LLM-dannyh
- CSV export dlya AI Overview

### L3.7 Dashboard service

Obnovitx `app/services/dashboard_service.py`:
- LLM visibility metriki na glavnoy stranitse
- Agregacii: mention rate, SOV, top citations

**Ocenka**: ~6 novyx faylov, ~8 izmenennyx, ~30 novyx testov

---

## Faza L4 — Frontend

**Cel**: UI dlya upravleniya LLM-monitoringom i prosmotra rezultatov.

### L4.1 Obnovlenie Project Detail Page

`templates/project.html` — dobavitx vkladki (tabs):
- **SERP** (suschestvuyuschij funkcional)
- **LLM Visibility** (noviy)
- **AI Overview** (noviy)

### L4.2 LLM Visibility Tab

Soderzhanie:
- **Summary cards**: Brand Mention Rate (%), SOV (%), # Citations, Cost ($)
- **Grafik**: Mention rate po dnyam × providers (Chart.js, lineynyy)
- **Grafik**: SOV — stolbchataya diagramma (nash brend vs konkurenty)
- **Tablica zaprosov**: query_text, mention_type, cited_urls, poslednyaya proverka
- **Dobavlenie zaprosov**: forma (tekst zaprosa, tip, konkurenty)
- **Detali otveta**: klick po zaprosu → modal s polnym otvetom LLM, podsvechennym upominaniem brenda

### L4.3 AI Overview Tab

Soderzhanie:
- **Summary cards**: % keywords s AIO, % gde my v AIO, avg poziciya v AIO
- **Grafik**: AIO prisutstvie po dnyam
- **Tablica**: keyword, has_aio, brand_in_aio, position, sources

### L4.4 Settings Page — LLM konfiguratsiya

Obnovitx `templates/settings.html`:
- Novaya sekciya "LLM API Keys":
  - OpenAI API Key
  - Google AI API Key
  - Anthropic API Key
  - Perplexity API Key
- Sekciya "LLM Settings":
  - Vybor modeli na provider (dropdown)
  - Budget limit ($X/mesyac)

### L4.5 Dashboard Page

Obnovitx `templates/dashboard.html`:
- Dobavitx LLM-metriki v kartochtki proektov:
  - LLM Mention Rate
  - AIO Coverage
  - Poslednij sbor

### L4.6 Novie JS-moduli

`static/llm_charts.js`:
- Grafiki mention rate, SOV, AIO coverage
- Pereklyuchenie provayderov
- Period selector

**Ocenka**: ~3 novyx fayla, ~6 izmenennyx

---

## Faza L5 — Celery-zadachi i raspisanie

**Cel**: Avtomaticheskiy periodicheskiy sbor LLM-dannyh.

### L5.1 Novye Celery tasks

`app/tasks/llm_collection_tasks.py`:
```python
@celery_app.task(name="collect_llm_project")
def collect_llm_project(tenant_id: str, project_id: int):
    """Sobrat LLM-dannye dlya odnogo proekta."""

@celery_app.task(name="collect_llm_all")
def collect_llm_all(tenant_id: str):
    """Sobrat LLM-dannye dlya vsex aktivnyx proektov tenanta."""

@celery_app.task(name="collect_ai_overview_project")
def collect_ai_overview_project(tenant_id: str, project_id: int):
    """Sobrat AI Overview dlya odnogo proekta."""
```

### L5.2 Obnovlenie Dispatcher

Obnovitx `app/tasks/collection_tasks.py`:
- `dispatch_collections()` teper takzhe zapuskaet LLM-sbor
- LLM-sbor mozhno zapuskatx s drugoy chastotoy (naprimer 1 raz v nedelyu vmesto ezhednevno)

### L5.3 Novye polya raspisaniya

Dobavitx v `Tenant`:
```
llm_collection_frequency    VARCHAR(20) DEFAULT 'weekly'   — daily/weekly/manual
llm_collection_day          SMALLINT DEFAULT 1             — denx nedeli (0=Mon, 6=Sun)
```

### L5.4 Cost tracking

`app/services/cost_service.py`:
- Tracking rasxodov po provaideram
- Budget limity (ostanovitx sbor esli prevyshen mesyachnyy limit)
- Alert cherez Telegram esli blizko k limitu

### L5.5 Obnovlenie maintenance_tasks.py

- Sozdanie particiy dlya llm_snapshots i ai_overview_snapshots
- Retenciya dlya novyx tablic

**Ocenka**: ~4 novyx fayla, ~4 izmenennyx, ~10 novyx testov

---

## Faza L6 — Prodvinutaya analitika

**Cel**: Share of Voice, trend-analiz, smart insights.

### L6.1 Share of Voice Engine

`app/services/sov_service.py`:
```python
async def calculate_sov(project_id, period, provider=None) -> SovResult:
    """
    SOV = (kol-vo upominaniy nashego brenda) / (obsschee kol-vo upominaniy vsex brendov)

    Vozvraschaet:
    - overall_sov: float (0.0 - 1.0)
    - sov_by_provider: {provider: float}
    - sov_by_query_type: {type: float}
    - competitor_sov: {competitor: float}
    - trend: [{date, sov}]
    """
```

### L6.2 Citation Analytics

`app/services/citation_service.py`:
- Top citiruemyx URL (nashix i konkurentov)
- Novie / poteryannie citaty (diff mezhdu periodami)
- Domen-level gruppirovka citaciy

### L6.3 Sentiment / Mention Quality

Prostoy analiz tona upominaniya:
- **Positive**: "recommended", "best", "top choice", "excellent"
- **Neutral**: prosto upominanie
- **Negative**: "avoid", "issues", "problems with"
- **Comparison**: upomyanut vmeste s konkurentami

### L6.4 Alerting

`app/notifications/llm_alerts.py`:
- Alert: brend propal iz otveta (byl upomyanut, teper net)
- Alert: noviy concurrent poyavilsya v otvetax
- Alert: rezkoye padenie SOV
- Dostavka: Telegram (est), email (buduschee)

**Ocenka**: ~5 novyx faylov, ~3 izmenennyx, ~15 novyx testov

---

## Faza L7 — Optimizaciya i masshtabirovanie

**Cel**: Snizheniye stoimosti API-zaprosov, uskoreniye sborki.

### L7.1 Smart Scheduling

- Ne proveratx kazhdiy zapros kazhdiy denx — rasstavitx prioritety
- Zapros s izmeneniami (brend to upominaetsya, to net) → proveratx chashche
- Stabilxnye zaprosy → proveratx rezhe

### L7.2 Response Caching

- Esli zapros uzhe proveren segodnya, ne povtoryatx
- Deduplikaciya odinakovyx zaprosov mezhdu tenantami (shared queries pool)

### L7.3 Batch Processing

- Gruppirovka zaprosov k LLM dlya minimizacii API-vyzovov
- Parallelxnyy sbor (asyncio.gather dlya raznyx provayderov)

### L7.4 ClickHouse Integration

- Perenos llm_snapshots v ClickHouse (chtenie)
- Materialized views dlya SOV i trend-agregaciy

### L7.5 Cost Optimization

- gpt-4o-mini vmesto gpt-4o dlya pervichnoj proverki
- Gemini Flash dlya massovyx zaprosov
- Kaskadnyy podhod: snachala deshevaya modelx, potom dorogaya dlya neodnoznachnyh sluchaev

---

## Poryadok realizacii i prioritety

```
Faza    | Opisanie                      | Prioritet | Slozhnostx | Ocenka
--------|-------------------------------|-----------|------------|--------
L1      | Modeli i infrastruktura       | CRITICAL  | Srednyaya  | ~8 faylov
L2      | LLM Kollektory               | CRITICAL  | Vysokaya   | ~10 faylov
L3      | API endpointy                 | CRITICAL  | Srednyaya  | ~6 faylov
L4      | Frontend                      | HIGH      | Srednyaya  | ~3 fayla
L5      | Celery tasks + raspisanie     | HIGH      | Srednyaya  | ~4 fayla
L6      | Prodvinutaya analitika        | MEDIUM    | Srednyaya  | ~5 faylov
L7      | Optimizaciya                  | LOW       | Vysokaya   | iterativno
```

**MVP (minimalniy zhiznesposobniy produkt)**: L1 + L2 + L3 + L4
- Polzovatelx mozhet: dobavitx zaprosy → zapustitx proverku → videtx rezultaty na grafike

**Posledovatelxnostx razrabotki**:
1. L1 — baza (modeli, migracii, shemy)
2. L2.1 + L2.2 — bazoviy kollektor + OpenAI (perviy MVP proverki)
3. L3.1 + L3.2 — API dlya zaprosov i rezultatov
4. L4.1 + L4.2 — frontend tab dlya LLM
5. L2.3 + L2.4 + L2.5 — ostalnye LLM provaidery
6. L2.6 — AI Overview kollektor
7. L3.3 + L4.3 — AI Overview API + frontend
8. L5 — avtomatizaciya sborki
9. L6 — prodvinutaya analitika
10. L7 — optimizaciya

---

## Riski i otkrytye voprosy

1. **Stoimostx LLM API**: Pri 1000 zaprosov × 4 provaydera × 30 dney = 120K vyzovov/mesyac. Nuzhna strategiya optimizacii s pervogo dnya.

2. **Deterministichnostx LLM**: Dazhe pri temperature=0 otvety mogut otlichatsya. Nuzhno resheniye — proveritx kazhdiy zapros N raz? Ili prinimatx kak est?

3. **AI Overview parsing**: Net ofitsialnogo API. Zavisim ot storonnego servisa (SerpAPI/DataForSEO) ili JustMagic. Nado utochnit dostupnostx.

4. **Rate limits LLM API**: Raznyie tiery imeyut raznyie limity. Nuzhno uvazhat ix i pravilno raspredelyatx nagruzku.

5. **Xranenie raw_response**: Polnye otvety LLM mogut zanit mnogo mesta. Nuzhna strategiya kompressii ili TTL.

6. **Yuridicheskiye voprosy**: Nekotorye LLM ToS mogut zapreschat avtomaticheskij monitoring. Nuzhno proveritx.

---

## Metrikx uspeshnosti plana

- [x] Polzovatelx mozhet dobavitx LLM-zaprosy cherez UI
- [ ] Polzovatelx mozhet zapustitx proverku i poluchitx rezultat za <2 min
- [ ] Grafik mention rate otobrazhaetsya korrekctno
- [ ] SOV schitaetsya pravilno
- [ ] AI Overview otslzhivaetsya
- [ ] Stoimostx 1 proverki < $0.01 (pri ispolzovanii deshevyx modeley)
- [x] 100+ novyx testov (282 testov)

---

## Realizovannye moduli

### Modulx 1: LLM Prompt Engineering Module ✅ (Done)

**Fayly**:
- `app/prompt_engine/__init__.py` — modulx docstring
- `app/prompt_engine/types.py` — Persona, Intent, TargetLLM enums + dataclasses
- `app/prompt_engine/normalization.py` — Layer 1: Input & Normalization Gateway
- `app/prompt_engine/matrix.py` — Layer 2: Combinatorial Matrix (64 RU + 64 EN templates)
- `app/prompt_engine/expansion.py` — Layer 3: Semantic Expansion & Pivots + LLM expansion
- `app/prompt_engine/adapters.py` — Layer 4: ChatGPT/Gemini/DeepSeek/YandexGPT/Perplexity adapters
- `app/prompt_engine/dispatcher.py` — Layer 5: Temperature/Resilience mapping, parameterization
- `app/prompt_engine/pipeline.py` — PromptPipeline orchestrator (sync + async)
- `app/schemas/prompt_engine.py` — Pydantic schemas
- `app/api/v1/prompt_engine.py` — 3 API endpoints (generate, preview, generate-and-save)
- `tests/test_prompt_engine.py` — 58 testov

**Rezultat**: 4 personas × 4 intents × N categories × M competitors → sotni promptov, adaptirovannyx pod kazhduyu LLM

---

### Modulx 2: LLM API Gateway Layer ✅ (Done)

**Arxitektura**: 5 komponentov
1. Priority Queue Manager — batching, Session_ID dlya Resilience Runs
2. Adaptive Rate Limiter — RPM/TPM per vendor
3. Vendor-Specific Adapters — DeepSeek 120s timeout, Gemini SAFETY, YandexGPT async polling
4. Resilience & Circuit Breaker — exponential backoff + jitter, Dead Letter Queue
5. Response Normalizer — unified DTO (request_id, vendor, status, latency_ms, cost_tokens)

**Fayly**:
- `app/gateway/__init__.py` — modulx docstring
- `app/gateway/types.py` — GatewayVendor, RequestStatus, RequestPriority, GatewayRequest/Response DTOs, VendorConfig, DEFAULT_VENDOR_CONFIGS
- `app/gateway/rate_limiter.py` — AdaptiveRateLimiter: sliding window RPM/TPM, per-vendor concurrent limits, acquire/release pattern
- `app/gateway/circuit_breaker.py` — CircuitBreaker: CLOSED→OPEN→HALF_OPEN, exponential backoff + jitter, Dead Letter Queue
- `app/gateway/queue_manager.py` — QueueManager: heapq priority queues, build_from_prompts() (Module 1 → Module 2 bridge)
- `app/gateway/vendor_adapters.py` — 5 adapters: ChatGPT, DeepSeek (503 busy), Gemini (SAFETY filter), YandexGPT (async polling + sync fallback), Perplexity (native citations)
- `app/gateway/normalizer.py` — normalize_response(), aggregate_resilience_responses()
- `app/gateway/gateway.py` — LlmGateway orchestrator: execute_single/batch/process_queue, retry loop, per-vendor semaphore concurrency
- `tests/test_gateway.py` — 74 testov

**Rezultat**: Polnostxyu asynchronnyy gateway s retry, circuit breaker, rate limiting, dead letter queue. Integriruyetsya s Module 1 cherez QueueManager.build_from_prompts()

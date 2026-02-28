"""
run_llm_fonbet.py — LLM-сбор для Фонбет (Fonbet)

Выполняет:
  1. Логин → JWT
  2. Создание/обновление проекта fonbet.ru
  3. Добавление LLM-запросов (ставки, букмекеры)
  4. Запуск сбора напрямую (без Celery)
  5. Проверка результатов

Usage:
    cd /Users/alexey/LLM_tracker
    .venv/bin/python run_llm_fonbet.py
"""

import asyncio
import json
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("llm_fonbet")

BASE = "http://localhost:8000"
EMAIL = "admin@test.com"
PASSWORD = "admin123"

BRAND = "Фонбет"
DOMAIN = "fonbet.ru"

COMPETITORS = ["1xBet", "Лига Ставок", "Винлайн", "Бетсити"]

QUERIES = [
    # brand_check — прямые проверки бренда
    {
        "query_text": "Какие отзывы о букмекерской конторе Фонбет?",
        "query_type": "brand_check",
        "target_brand": BRAND,
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Фонбет надёжная букмекерская контора или нет?",
        "query_type": "brand_check",
        "target_brand": BRAND,
        "competitors": COMPETITORS,
    },
    # comparison — сравнения
    {
        "query_text": "Сравни Фонбет и 1xBet: где лучше делать ставки на спорт?",
        "query_type": "comparison",
        "target_brand": BRAND,
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Фонбет или Лига Ставок — что лучше для ставок на футбол?",
        "query_type": "comparison",
        "target_brand": BRAND,
        "competitors": COMPETITORS,
    },
    # recommendation — рекомендации
    {
        "query_text": "Лучшие букмекерские конторы России 2026",
        "query_type": "recommendation",
        "target_brand": BRAND,
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Где лучше всего делать ставки на спорт в России?",
        "query_type": "recommendation",
        "target_brand": BRAND,
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Топ-5 легальных букмекеров в России с лучшими коэффициентами",
        "query_type": "recommendation",
        "target_brand": BRAND,
        "competitors": COMPETITORS,
    },
    # custom — разные тематики
    {
        "query_text": "Какой букмекер предлагает лучший бонус при регистрации?",
        "query_type": "custom",
        "target_brand": BRAND,
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Как вывести деньги из Фонбет на карту?",
        "query_type": "brand_check",
        "target_brand": BRAND,
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Какие букмекеры имеют лучшее мобильное приложение для ставок?",
        "query_type": "recommendation",
        "target_brand": BRAND,
        "competitors": COMPETITORS,
    },
]


async def main():
    import httpx

    # ── Step 1: Login ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Шаг 1: Логин")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=BASE, timeout=30) as c:
        r = await c.post("/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})
        r.raise_for_status()
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("  ✓ Логин успешен")

        # ── Step 2: Verify API keys ──────────────────────────
        print("\n" + "=" * 60)
        print("  Шаг 2: Проверка API-ключей")
        print("=" * 60)

        r = await c.get("/api/v1/settings/credentials", headers=headers)
        r.raise_for_status()
        creds = r.json()
        openai_ok = creds.get("openai_api_key", False)
        deepseek_ok = creds.get("deepseek_api_key", False)
        print(f"  OpenAI:   {'✓ настроен' if openai_ok else '✗ НЕ НАСТРОЕН!'}")
        print(f"  DeepSeek: {'✓ настроен' if deepseek_ok else '✗ НЕ НАСТРОЕН!'}")

        if not openai_ok or not deepseek_ok:
            print("\n  ❌ Один или оба API-ключа не настроены!")
            sys.exit(1)

        # ── Step 3: Create / find project ────────────────────
        print("\n" + "=" * 60)
        print(f"  Шаг 3: Проект {DOMAIN}")
        print("=" * 60)

        r = await c.get("/api/v1/projects/", headers=headers)
        r.raise_for_status()
        projects = r.json()

        project_id = None
        tenant_id = None

        for p in projects:
            if p.get("domain") == DOMAIN:
                project_id = p["id"]
                tenant_id = p["tenant_id"]
                print(f"  Найден существующий проект: id={project_id}")
                r = await c.put(
                    f"/api/v1/projects/{project_id}",
                    headers=headers,
                    json={
                        "track_llm": True,
                        "llm_providers": ["chatgpt", "deepseek"],
                        "brand_name": BRAND,
                    },
                )
                r.raise_for_status()
                print("  ✓ Обновлён: track_llm=True, providers=[chatgpt, deepseek]")
                break

        if project_id is None:
            r = await c.post(
                "/api/v1/projects/",
                headers=headers,
                json={
                    "name": f"{BRAND} LLM Tracking",
                    "domain": DOMAIN,
                    "search_engine": "both",
                    "region_yandex": 213,
                    "track_llm": True,
                    "llm_providers": ["chatgpt", "deepseek"],
                    "brand_name": BRAND,
                },
            )
            r.raise_for_status()
            proj = r.json()
            project_id = proj["id"]
            tenant_id = proj["tenant_id"]
            print(f"  ✓ Создан новый проект: id={project_id}")

        print(f"  project_id={project_id}, tenant_id={tenant_id}")

        # ── Step 4: Add LLM queries ─────────────────────────
        print("\n" + "=" * 60)
        print(f"  Шаг 4: Добавление {len(QUERIES)} LLM-запросов")
        print("=" * 60)

        r = await c.post(
            f"/api/v1/projects/{project_id}/llm-queries/",
            headers=headers,
            json={"queries": QUERIES},
        )
        r.raise_for_status()
        qresult = r.json()
        print(f"  Создано: {qresult['created']}, пропущено (дубли): {qresult['skipped']}")

        r = await c.get(f"/api/v1/projects/{project_id}/llm-queries/", headers=headers)
        r.raise_for_status()
        qdata = r.json()
        print(f"  Всего запросов: {qdata['total']}")
        for q in qdata["items"]:
            print(f"    [{q['id']}] {q['query_type']:15s} {q['query_text'][:60]}")

    # ── Step 5: Run collection (bypass Celery) ───────────────
    print("\n" + "=" * 60)
    print("  Шаг 5: Запуск сбора (напрямую, без Celery)")
    print("=" * 60)
    print(f"  Будет выполнено {len(QUERIES)} × 2 = {len(QUERIES) * 2} запросов к LLM API")
    print("  Провайдеры: chatgpt (gpt-4o-mini), deepseek (deepseek-chat)")
    print("  Ожидаемая стоимость: < $0.01")
    print()

    start_time = time.time()

    from app.tasks.llm_collection_tasks import _collect_llm_project_async

    result = await _collect_llm_project_async(str(tenant_id), project_id)

    elapsed = time.time() - start_time
    print(f"\n  Сбор завершён за {elapsed:.1f} сек")
    print("  Результат:")
    print(json.dumps(result, indent=4, ensure_ascii=False, default=str))

    # Check for errors
    providers_result = result.get("providers", {})
    total_collected = 0
    total_errors = 0
    for prov, pdata in providers_result.items():
        collected = pdata.get("collected", 0)
        errors = pdata.get("errors", [])
        total_collected += collected
        total_errors += len(errors) if isinstance(errors, list) else (1 if errors else 0)
        status = "✓" if collected > 0 and not errors else "✗"
        print(f"  {status} {prov}: collected={collected}, errors={errors}")

    # ── Step 6: Verify via API ───────────────────────────────
    print("\n" + "=" * 60)
    print("  Шаг 6: Проверка результатов")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=BASE, timeout=30) as c:
        r = await c.post("/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})
        r.raise_for_status()
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Dashboard
        r = await c.get(f"/api/v1/llm/dashboard/{project_id}?days=1", headers=headers)
        r.raise_for_status()
        dash = r.json()
        print("\n  📊 Dashboard Stats:")
        print(f"    Total queries:       {dash['total_queries']}")
        print(f"    Total checks:        {dash['total_checks']}")
        print(f"    Brand mention rate:  {dash['brand_mention_rate']:.1%}")
        print(f"    SOV:                 {dash['sov']:.1%}")
        print(f"    Total cost:          ${dash['total_cost_usd']:.4f}")
        if dash.get("mention_rate_by_provider"):
            print("    По провайдерам:")
            for prov, rate in dash["mention_rate_by_provider"].items():
                print(f"      {prov}: {rate:.1%}")

        # BI Dashboard
        r = await c.get(f"/api/v1/llm/bi-dashboard/{project_id}?days=1", headers=headers)
        r.raise_for_status()
        bi = r.json()
        gm = bi.get("global_metrics", {})
        print("\n  🎯 BI Dashboard:")
        print(f"    AIVS:             {gm.get('aivs', 0):.1f}")
        print(f"    SoM:              {gm.get('som', 0):.1f}%")
        print(f"    Resilience Score: {gm.get('resilience_score', 0):.2f}")
        print(f"    Total Responses:  {gm.get('total_responses', 0)}")
        print(f"    Mention Rate:     {gm.get('mention_rate', 0):.1%}")

        # Debug
        r = await c.get(f"/api/v1/llm/debug/{project_id}?days=1", headers=headers)
        r.raise_for_status()
        debug = r.json()
        print("\n  🔍 Debug Trace:")
        print(f"    Snapshots:    {debug['snapshot_count']}")
        print(f"    AIVS:         {debug['aivs_debug']['final_score']:.2f}")
        print(f"    AIVS formula: {debug['aivs_debug']['formula']}")
        print(f"    SoM:          {debug['som_debug']['final_score']:.2f}%")
        print(f"    SoM formula:  {debug['som_debug']['formula']}")

    # ── Step 7: URLs ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ ГОТОВО!")
    print("=" * 60)
    print(f"\n  Собрано {total_collected} снапшотов, ошибок: {total_errors}")
    print(f"\n  🖥  Dashboard:     http://localhost:8000/project/{project_id}/llm-dashboard")
    print(f"  🔍 Debug Console: http://localhost:8000/project/{project_id}/llm-debug")
    print()


if __name__ == "__main__":
    asyncio.run(main())

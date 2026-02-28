"""
run_llm_test.py — End-to-end тест LLM-сбора для ostrovok.ru

Выполняет весь пайплайн за один запуск:
  1. Логин → JWT
  2. Проверка API-ключей (OpenAI, DeepSeek)
  3. Создание/обновление проекта ostrovok.ru
  4. Добавление 8 тестовых LLM-запросов
  5. Запуск сбора напрямую (без Celery)
  6. Проверка результатов через API

Usage:
    cd /Users/alexey/LLM_tracker
    .venv/bin/python run_llm_test.py
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
logger = logging.getLogger("llm_test")

BASE = "http://localhost:8000"
EMAIL = "admin@test.com"
PASSWORD = "admin123"

COMPETITORS = ["Яндекс Путешествия", "Отелло", "101hotels"]

QUERIES = [
    {
        "query_text": "Какие отзывы об Островок?",
        "query_type": "brand_check",
        "target_brand": "Островок",
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Сравни Островок и Яндекс Путешествия для бронирования отелей",
        "query_type": "comparison",
        "target_brand": "Островок",
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Где лучше бронировать отели в России?",
        "query_type": "recommendation",
        "target_brand": "Островок",
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Какие сервисы бронирования отелей работают в России?",
        "query_type": "recommendation",
        "target_brand": "Островок",
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Ostrovok или Booking.com что лучше для бронирования?",
        "query_type": "comparison",
        "target_brand": "Островок",
        "competitors": COMPETITORS + ["Booking.com"],
    },
    {
        "query_text": "Лучшие сайты для бронирования отелей 2026",
        "query_type": "recommendation",
        "target_brand": "Островок",
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Островок бронирование отзывы и рейтинг",
        "query_type": "brand_check",
        "target_brand": "Островок",
        "competitors": COMPETITORS,
    },
    {
        "query_text": "Как забронировать отель онлайн дешево в Москве?",
        "query_type": "custom",
        "target_brand": "Островок",
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
        print(f"  ✓ Логин успешен (token: {token[:20]}...)")

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
            print("  Добавьте их через Settings → Credentials в UI")
            sys.exit(1)

        # ── Step 3: Create / find project ────────────────────
        print("\n" + "=" * 60)
        print("  Шаг 3: Проект ostrovok.ru")
        print("=" * 60)

        r = await c.get("/api/v1/projects/", headers=headers)
        r.raise_for_status()
        projects = r.json()

        project_id = None
        tenant_id = None

        for p in projects:
            if p.get("domain") == "ostrovok.ru":
                project_id = p["id"]
                tenant_id = p["tenant_id"]
                print(f"  Найден существующий проект: id={project_id}")
                r = await c.put(
                    f"/api/v1/projects/{project_id}",
                    headers=headers,
                    json={
                        "track_llm": True,
                        "llm_providers": ["chatgpt", "deepseek"],
                        "brand_name": "Островок",
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
                    "name": "Ostrovok.ru LLM Tracking",
                    "domain": "ostrovok.ru",
                    "search_engine": "both",
                    "region_yandex": 213,
                    "track_llm": True,
                    "llm_providers": ["chatgpt", "deepseek"],
                    "brand_name": "Островок",
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
        print("  Шаг 4: Добавление LLM-запросов")
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
        print(f"  Всего запросов в проекте: {qdata['total']}")
        for q in qdata["items"]:
            print(f"    [{q['id']}] {q['query_type']:15s} {q['query_text'][:55]}...")

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
    print("  Шаг 6: Проверка результатов через API")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=BASE, timeout=30) as c:
        # Re-login
        r = await c.post("/api/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})
        r.raise_for_status()
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 6a. Dashboard stats
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
        if dash.get("competitor_sov"):
            print("    SOV конкурентов:")
            for comp, sov in dash["competitor_sov"].items():
                print(f"      {comp}: {sov:.1%}")

        # 6b. BI Dashboard
        r = await c.get(f"/api/v1/llm/bi-dashboard/{project_id}?days=1", headers=headers)
        r.raise_for_status()
        bi = r.json()
        gm = bi.get("global_metrics", {})
        print("\n  🎯 BI Dashboard (Global Metrics):")
        print(f"    AIVS:             {gm.get('aivs', 0):.1f}")
        print(f"    SoM:              {gm.get('som', 0):.1f}%")
        print(f"    Resilience Score: {gm.get('resilience_score', 0):.2f}")
        print(f"    Total Responses:  {gm.get('total_responses', 0)}")
        print(f"    Mention Rate:     {gm.get('mention_rate', 0):.1%}")

        # 6c. Details (first 3)
        r = await c.get(f"/api/v1/llm/details/{project_id}?days=1&limit=3", headers=headers)
        r.raise_for_status()
        details = r.json()
        print(f"\n  📝 Детали: {details['total']} снапшотов, первые 3:")
        for item in details["items"][:3]:
            print(
                f"    [{item['llm_provider']:10s}] "
                f"mentioned={item['brand_mentioned']}, "
                f"type={item['mention_type']:12s}, "
                f"tokens={item['response_tokens']}, "
                f"${item['cost_usd']:.4f}"
            )
            if item.get("mention_context"):
                ctx = item["mention_context"][:80]
                print(f"      context: «{ctx}...»")

        # 6d. Citations
        r = await c.get(f"/api/v1/llm/citations/{project_id}?days=1", headers=headers)
        r.raise_for_status()
        cit = r.json()
        print(f"\n  🔗 Цитирования: {cit['total']} уникальных URL")
        for c_item in cit["citations"][:5]:
            print(f"    {c_item['domain']} ({c_item['count']}×) via {c_item['providers']}")

        # 6e. GEO Advisor
        r = await c.get(f"/api/v1/llm/geo-advisor/{project_id}?days=1", headers=headers)
        r.raise_for_status()
        geo = r.json()
        insights = geo.get("insights", [])
        print(f"\n  🌍 GEO Advisor: {len(insights)} рекомендаций")
        for ins in insights[:3]:
            print(f"    [{ins['severity']:8s}] {ins['title_ru']}")

    # ── Step 7: Dashboard URL ────────────────────────────────
    print("\n" + "=" * 60)
    print("  Шаг 7: Визуальная проверка")
    print("=" * 60)
    print("\n  🖥  Откройте в браузере:")
    print(f"  http://localhost:8000/project/{project_id}/llm-dashboard")
    print()
    print("=" * 60)
    print(f"  ✅ ГОТОВО! Собрано {total_collected} снапшотов, ошибок: {total_errors}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

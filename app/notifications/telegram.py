"""Telegram notification service — async version."""

import asyncio
import logging
from datetime import date

import httpx

from app.services.collection_service import ProjectCollectionResult

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split long message into chunks at newline boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos == -1:
            split_pos = max_len
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


async def send_telegram_message(
    text: str,
    bot_token: str,
    chat_id: str,
    parse_mode: str = "HTML",
) -> None:
    """Send a message via Telegram Bot API. Splits if too long."""
    if not bot_token or not chat_id:
        logger.info("Telegram: no bot token or chat_id, skipping")
        return

    url = TELEGRAM_API.format(token=bot_token)
    chunks = _split_message(text, 4000)

    async with httpx.AsyncClient(timeout=15) as client:
        for chunk in chunks:
            try:
                resp = await client.post(
                    url,
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": parse_mode,
                    },
                )
                if resp.status_code == 429:
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                    logger.warning("Telegram rate limit, retry after %ds", retry_after)
                    await asyncio.sleep(retry_after)
                    await client.post(
                        url,
                        json={
                            "chat_id": chat_id,
                            "text": chunk,
                            "parse_mode": parse_mode,
                        },
                    )
                elif resp.status_code != 200:
                    logger.error("Telegram send failed (%d): %s", resp.status_code, resp.text)
            except Exception as e:
                logger.error("Telegram send error: %s", e)


def format_daily_report(results: list[ProjectCollectionResult]) -> str:
    """Format collection results as HTML for Telegram."""
    today = date.today()
    lines = [f"<b>Position Tracker — {today.isoformat()}</b>"]

    for r in results:
        lines.append(f"\n<b>{r.project_name}</b> ({r.domain})")

        jm_count = r.justmagic.collected if r.justmagic else 0
        ywm_count = r.ywm.collected if r.ywm else 0
        gsc_count = r.gsc.collected if r.gsc else 0
        lines.append(f"SERP: {jm_count} | YWM: {ywm_count} | GSC: {gsc_count}")

        all_errors = list(r.errors)
        if r.justmagic and r.justmagic.errors:
            all_errors.extend(r.justmagic.errors)
        if r.ywm and r.ywm.errors:
            all_errors.extend(r.ywm.errors)
        if r.gsc and r.gsc.errors:
            all_errors.extend(r.gsc.errors)

        for err in all_errors[:3]:
            lines.append(f"  Error: {str(err)[:100]}")

    return "\n".join(lines)


async def send_daily_report(
    results: list[ProjectCollectionResult],
    bot_token: str,
    chat_id: str,
) -> None:
    """Format and send daily Telegram report."""
    if not results:
        logger.info("No collection results to report")
        return
    text = format_daily_report(results)
    await send_telegram_message(text, bot_token, chat_id)
    logger.info("Daily Telegram report sent")

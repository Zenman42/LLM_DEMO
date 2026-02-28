"""GEO Audit endpoints — robots.txt proxy and site content audit."""

import ipaddress
import re
import socket
from urllib.parse import urlparse
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id
from app.core.exceptions import BadRequestError, NotFoundError
from app.db.postgres import get_db
from app.models.project import Project
from app.schemas.geo_audit import SiteAuditResponse
from app.services.geo_audit_service import (
    discover_urls_from_sitemap,
    fetch_robots_txt,
    run_site_audit,
)

router = APIRouter(prefix="/geo-audit", tags=["geo-audit"])

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LLMTracker-GeoAudit/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Regex for valid bare domain (no scheme, no path, no userinfo)
_DOMAIN_RE = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$",
)


# ── Helpers ──────────────────────────────────────────────────────


def _validate_domain(raw: str) -> str:
    """Validate and sanitise a bare domain string. Raises BadRequestError on invalid input."""
    domain = raw.strip().removeprefix("https://").removeprefix("http://").split("/")[0]
    # Strip port if present (e.g. example.com:8080)
    domain = domain.split(":")[0]
    # Strip any userinfo (e.g. user@host)
    if "@" in domain:
        raise BadRequestError("Domain must not contain userinfo (@ symbol)")
    if not _DOMAIN_RE.match(domain):
        raise BadRequestError("Invalid domain format. Provide a bare domain, e.g. example.com")
    # Resolve and block private / reserved IPs (SSRF protection)
    try:
        addrs = socket.getaddrinfo(domain, None, proto=socket.IPPROTO_TCP)
        for _family, _type, _proto, _canonname, sockaddr in addrs:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
                raise BadRequestError("Domain resolves to a private/reserved IP address")
    except socket.gaierror:
        pass  # DNS resolution failure is OK — httpx will handle it
    return domain


def _validate_url_belongs_to_domain(url: str, domain: str) -> bool:
    """Check that a URL belongs to the expected domain (for SSRF protection)."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        expected = domain.lower().rstrip(".")
        return host == expected or host.endswith("." + expected)
    except Exception:
        return False


async def _get_project(project_id: int, tenant_id: UUID, db: AsyncSession) -> Project:
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id),
    )
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    return project


# ── Endpoints ────────────────────────────────────────────────────


@router.get("/robots-proxy", response_class=PlainTextResponse)
async def robots_proxy(
    url: str = Query(..., description="Domain name, e.g. example.com"),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> PlainTextResponse:
    """CORS-safe proxy to fetch a site's robots.txt."""
    domain = _validate_domain(url)

    async with httpx.AsyncClient(headers=_HTTP_HEADERS) as client:
        text = await fetch_robots_txt(client, domain)

    if text is None:
        raise NotFoundError(f"Could not fetch robots.txt for {domain}")

    return PlainTextResponse(content=text)


@router.get("/site-audit/{project_id}", response_model=SiteAuditResponse)
async def site_audit(
    project_id: int,
    pages: int = Query(default=5, ge=1, le=20, description="Max pages to scan"),
    urls: str | None = Query(default=None, description="Comma-separated URLs to scan (overrides sitemap discovery)"),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
) -> SiteAuditResponse:
    """Run a GEO site audit: crawl access, schema.org, answer-first, E-E-A-T, freshness."""
    project = await _get_project(project_id, tenant_id, db)

    domain = project.domain
    if not domain:
        raise BadRequestError("Project has no domain configured")

    # Clean domain
    domain = domain.strip().removeprefix("https://").removeprefix("http://").rstrip("/")

    async with httpx.AsyncClient(headers=_HTTP_HEADERS) as client:
        # Determine pages to scan
        if urls:
            # Validate that user-supplied URLs belong to the project domain (SSRF protection)
            page_urls = []
            for u in urls.split(","):
                u = u.strip()
                if u and _validate_url_belongs_to_domain(u, domain):
                    page_urls.append(u)
            if not page_urls:
                raise BadRequestError(
                    f"All provided URLs are outside the project domain ({domain})",
                )
            page_urls = page_urls[:pages]
        else:
            page_urls, _ = await discover_urls_from_sitemap(client, domain, limit=pages)

        result = await run_site_audit(client, domain, page_urls, project_id)

    return result

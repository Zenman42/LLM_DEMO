"""Regions API — serves static region reference data for dropdowns."""

from fastapi import APIRouter

from app.data.regions import GOOGLE_REGIONS, YANDEX_REGIONS

router = APIRouter(prefix="/regions", tags=["regions"])


@router.get("/")
async def list_regions():
    """Return available regions for Yandex and Google search engines.

    Used by the frontend to populate region selection dropdowns
    in project creation and edit forms.
    """
    return {
        "yandex": YANDEX_REGIONS,
        "google": GOOGLE_REGIONS,
    }

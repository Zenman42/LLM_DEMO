from fastapi import APIRouter

from app.api.v1.account import router as account_router
from app.api.v1.api_keys import router as api_keys_router
from app.api.v1.auth import router as auth_router
from app.api.v1.collection import router as collection_router
from app.api.v1.dashboard import router as dashboard_router
from app.api.v1.export import router as export_router
from app.api.v1.keywords import router as keywords_router
from app.api.v1.projects import router as projects_router
from app.api.v1.regions import router as regions_router
from app.api.v1.serp import router as serp_router
from app.api.v1.settings import router as settings_router
from app.api.v1.users import router as users_router
from app.api.v1.competitors import router as competitors_router
from app.api.v1.llm_queries import router as llm_queries_router
from app.api.v1.llm_results import router as llm_results_router
from app.api.v1.prompt_engine import router as prompt_engine_router
from app.api.v1.onboarding import router as onboarding_router
from app.api.v1.scenarios import router as scenarios_router
from app.api.v1.geo_audit import router as geo_audit_router

# from app.api.v1.query_tags import router as query_tags_router  # deprecated: tables don't exist
from app.api.v1.webmaster import router as webmaster_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(auth_router)
api_v1_router.include_router(projects_router)
api_v1_router.include_router(keywords_router)
api_v1_router.include_router(collection_router)
api_v1_router.include_router(settings_router)
api_v1_router.include_router(dashboard_router)
api_v1_router.include_router(serp_router)
api_v1_router.include_router(webmaster_router)
api_v1_router.include_router(users_router)
api_v1_router.include_router(api_keys_router)
api_v1_router.include_router(export_router)
api_v1_router.include_router(account_router)
api_v1_router.include_router(regions_router)
api_v1_router.include_router(llm_queries_router)
api_v1_router.include_router(llm_results_router)
api_v1_router.include_router(prompt_engine_router)
api_v1_router.include_router(onboarding_router)
api_v1_router.include_router(competitors_router)
api_v1_router.include_router(scenarios_router)
api_v1_router.include_router(geo_audit_router)
# api_v1_router.include_router(query_tags_router)  # deprecated: tables don't exist

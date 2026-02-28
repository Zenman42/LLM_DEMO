from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

celery_app = Celery(
    "position_tracker",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)

# Celery Beat schedule — dispatcher runs every minute,
# checks which tenants need collection based on their collection_hour/minute.
celery_app.conf.beat_schedule = {
    "dispatch-tenant-collections": {
        "task": "dispatch_collections",
        "schedule": crontab(),  # every minute
    },
    "create-monthly-partitions": {
        "task": "create_partitions",
        "schedule": crontab(hour=3, minute=0, day_of_month=1),  # 1st of each month at 03:00
    },
    "drop-old-partitions": {
        "task": "drop_old_partitions",
        "schedule": crontab(hour=4, minute=0, day_of_month=2),  # 2nd of each month at 04:00
    },
}

# Auto-discover tasks from tasks modules
celery_app.autodiscover_tasks(["app.tasks"])

# Explicit include as fallback for autodiscover (needed for CLI worker startup)
celery_app.conf.include = [
    "app.tasks.collection_tasks",
    "app.tasks.llm_collection_tasks",
    "app.tasks.maintenance_tasks",
]

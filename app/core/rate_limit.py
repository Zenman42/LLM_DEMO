"""Rate limiting configuration using slowapi."""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Rate limiter instance — use remote address as key
limiter = Limiter(key_func=get_remote_address)

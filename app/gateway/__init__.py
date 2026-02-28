"""LLM API Gateway Layer (Module 2).

Provides event-driven, async infrastructure for dispatching prompts
to LLM vendors with:
  - Priority Queue Manager
  - Adaptive Rate Limiter (RPM/TPM per vendor)
  - Vendor-Specific Adapters (protocol differences)
  - Resilience & Circuit Breaker (exponential backoff with jitter)
  - Response Normalizer (unified DTO)
"""

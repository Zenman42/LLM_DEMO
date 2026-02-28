"""LLM Response Analysis & Evaluation Engine (Module 3).

Cascading NLP pipeline for analyzing raw LLM responses:
  1. Text Preprocessor & Sanitizer
  2. NER & Entity Resolution Engine
  3. Structural & Ranking Parser
  4. Context Evaluator / LLM-as-a-Judge
  5. Citation & Grounding Extractor
  6. Micro-Scoring Calculator

Input:  GatewayResponse (from Module 2)
Output: AnalyzedResponse (structured metrics for BI dashboard)
"""

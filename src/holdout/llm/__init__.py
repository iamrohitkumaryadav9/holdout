"""LLM access: one protocol, a Gemini client, scripted fakes, budget, cache, structured calls.

Every call returns a validated Pydantic model or raises a typed error; there is no silent
fallback inside an agent run (offline mode is a separate, explicit path).
"""

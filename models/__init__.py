__all__ = [
    "GeminiModel",
]

try:
    from .gemini_model import GeminiModel
except Exception:
    # Optional dependency; import may fail if google-generativeai not installed.
    GeminiModel = None


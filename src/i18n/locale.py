from __future__ import annotations

from src.config import get_config

from .store import DEFAULT_LOCALE, SUPPORTED_LOCALES


def resolve_locale(explicit_locale: str | None = None) -> str:
  if explicit_locale in SUPPORTED_LOCALES:
    return explicit_locale

  config = get_config()
  config_locale = getattr(config, "locale", None)
  if isinstance(config_locale, str) and config_locale in SUPPORTED_LOCALES:
    return config_locale

  return DEFAULT_LOCALE

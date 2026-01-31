from __future__ import annotations

from dataclasses import dataclass

from .store import DEFAULT_LOCALE, SUPPORTED_LOCALES, TRANSLATIONS


@dataclass(frozen=True)
class Translator:
  locale: str = DEFAULT_LOCALE

  def __post_init__(self) -> None:
    if self.locale not in SUPPORTED_LOCALES:
      object.__setattr__(self, "locale", DEFAULT_LOCALE)

  def t(self, key: str, **kwargs: object) -> str:
    messages = TRANSLATIONS.get(self.locale, TRANSLATIONS[DEFAULT_LOCALE])
    template = messages.get(key, TRANSLATIONS[DEFAULT_LOCALE].get(key, key))
    if not kwargs:
      return template
    try:
      return template.format(**kwargs)
    except KeyError:
      return template

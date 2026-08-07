from __future__ import annotations

import re
from typing import Protocol


class EntityExtractor(Protocol):
    def extract(self, text: str) -> list[str]: ...


class RegexEntityExtractor:
    """Dependency-free fallback: captures capitalized spans, emails, dates and IDs."""

    DATE = re.compile(r"\b(?:\d{1,2}[/-]){1,2}\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
    EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
    CAPS = re.compile(r"\b(?:[A-ZÀ-Ỹ][\wÀ-ỹ'-]+(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ'-]+){0,3})\b")

    def extract(self, text: str) -> list[str]:
        found: list[str] = []
        for pat in (self.EMAIL, self.DATE, self.CAPS):
            found.extend(m.group(0).strip() for m in pat.finditer(text))
        seen = set()
        return [x for x in found if x and not (x.lower() in seen or seen.add(x.lower()))]


class SpacyEntityExtractor:
    def __init__(self, model: str = "xx_ent_wiki_sm"):
        import spacy
        self.nlp = spacy.load(model)

    def extract(self, text: str) -> list[str]:
        doc = self.nlp(text)
        seen = set()
        out = []
        for ent in doc.ents:
            key = ent.text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(ent.text.strip())
        return out

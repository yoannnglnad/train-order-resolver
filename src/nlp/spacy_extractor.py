"""Optional spaCy-based extractor using CamemBERT fr_dep_news_trf."""

from __future__ import annotations

from typing import List, Optional, Tuple


class SpacyExtractor:
    """Thin wrapper that loads spaCy model lazily on first use."""

    def __init__(self, model: str = "fr_dep_news_trf") -> None:
        self._model_name = model
        self._nlp = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Import spacy and load the model on first call."""
        if self._loaded:
            return
        self._loaded = True
        try:
            import spacy  # type: ignore
        except Exception:
            self._nlp = None
            return
        try:
            self._nlp = spacy.load(self._model_name)
        except Exception:
            self._nlp = None

    def is_ready(self) -> bool:
        """Check if spacy is importable without loading the model."""
        try:
            import spacy  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False

    def extract(self, text: str) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Return (from_candidate, to_candidate, via_candidates).
        Very lightweight heuristic: take location-like entities in order.
        """
        self._ensure_loaded()
        if not self._nlp:
            return None, None, []
        doc = self._nlp(text)
        locs = [ent.text for ent in doc.ents if ent.label_ in {"LOC", "GPE"}]
        if len(locs) >= 2:
            return locs[0], locs[1], []
        if len(locs) == 1:
            return locs[0], None, []
        return None, None, []

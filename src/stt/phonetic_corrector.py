"""Correct station names in STT-transcribed text using IPA phonetic matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from rapidfuzz.distance import Levenshtein

from src.pathfinding.prepare_stations import normalize_name
from src.stt.phonetic_db import load_phonetic_index
from src.utils.config import DEFAULT_PHONETIC_INDEX_PATH

# French stopwords that should never be corrected to station names
_STOPWORDS = frozenset(
    {
        "je",
        "tu",
        "il",
        "elle",
        "on",
        "nous",
        "vous",
        "ils",
        "elles",
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "au",
        "aux",
        "et",
        "ou",
        "en",
        "à",
        "a",
        "ce",
        "se",
        "ne",
        "que",
        "qui",
        "si",
        "sa",
        "son",
        "ma",
        "mon",
        "mes",
        "ses",
        "mais",
        "est",
        "sont",
        "suis",
        "ai",
        "as",
        "avec",
        "pour",
        "par",
        "sur",
        "dans",
        "pas",
        "plus",
        "sans",
        "vers",
        "chez",
        "tout",
        "tous",
        "bien",
        "très",
        "peu",
        "trop",
        "aller",
        "allez",
        "allons",
        "veux",
        "veut",
        "vais",
        "va",
        "voudrais",
        "voudrait",
        "prendre",
        "prends",
        "prend",
        "prenez",
        "partir",
        "pars",
        "part",
        "partez",
        "partons",
        "train",
        "billet",
        "ticket",
        "gare",
        "voyage",
        "trajet",
        "ici",
        "heure",
        "heures",
        "matin",
        "soir",
        "midi",
        "nuit",
        "demain",
        "depuis",
        "jusque",
    }
)

# Punctuation to strip before matching
_PUNCT = re.compile(r"[.,;:!?\"'()]+")


@dataclass
class Correction:
    """A single correction applied to the text."""

    original: str
    corrected: str
    station_id: str
    ipa_distance: float


@dataclass
class CorrectionResult:
    """Result of phonetic correction on a text."""

    original_text: str
    corrected_text: str
    corrections: List[Correction] = field(default_factory=list)


class PhoneticCorrector:
    """Correct station names in transcribed text using IPA phonetic matching."""

    def __init__(
        self,
        index_path: Path = DEFAULT_PHONETIC_INDEX_PATH,
        threshold: float = 0.35,
        index: Optional[Dict[str, dict]] = None,
    ) -> None:
        self._index = index if index is not None else load_phonetic_index(index_path)
        self._threshold = threshold
        self._backend = EspeakBackend(
            language="fr-fr",
            preserve_punctuation=False,
            with_stress=False,
            language_switch="remove-flags",
        )
        self._separator = Separator(phone=" ", word=" _ ", syllable="")

        # Pre-build flat list of (station_id, canonical_name, ipa_signature) tuples
        self._signatures: List[Tuple[str, str, str]] = []
        for sid, entry in self._index.items():
            for ipa_key, name_key in [
                ("ipa_name", "name"),
                ("ipa_name_norm", "name"),
                ("ipa_city", "city"),
            ]:
                ipa = entry.get(ipa_key, "").strip()
                name = entry.get(name_key, "").strip()
                if ipa and name:
                    self._signatures.append((sid, name, ipa))

        # Build set of known station names and cities (lowercase) for exact-match protection
        self._known_names: Set[str] = set()
        for entry in self._index.values():
            for key in ("name", "name_norm", "city"):
                val = entry.get(key, "").strip().lower()
                if val and len(val) >= 3:
                    self._known_names.add(val)

    def _phonemize(self, text: str) -> str:
        """Phonemize a single string."""
        results = self._backend.phonemize([text], separator=self._separator, strip=True)
        # Strip extra whitespace left by language-switch flag removal
        return " ".join(results[0].split()) if results else ""

    def _phonemize_batch(self, texts: List[str]) -> List[str]:
        """Phonemize multiple strings in a single eSpeak call."""
        if not texts:
            return []
        results = self._backend.phonemize(texts, separator=self._separator, strip=True)
        return [" ".join(r.split()) for r in results]

    def _is_stopword(self, text: str) -> bool:
        """Check if text consists only of stopwords."""
        words = text.lower().split()
        return all(w in _STOPWORDS for w in words)

    def _is_known_station(self, text: str) -> bool:
        """Check if text already matches a known station name or city."""
        cleaned = _PUNCT.sub("", text).strip().lower()
        return cleaned in self._known_names

    def _content_word_ratio(self, words: List[str]) -> float:
        """Fraction of non-stopword words in a list."""
        if not words:
            return 0.0
        content = sum(1 for w in words if w.lower() not in _STOPWORDS)
        return content / len(words)

    def _normalized_distance(self, a: str, b: str) -> float:
        """Compute normalized Levenshtein distance between two IPA strings."""
        if not a and not b:
            return 0.0
        return Levenshtein.normalized_distance(a, b)

    def _find_best_match(
        self, candidate_ipa: str, candidate_text: str
    ) -> Optional[Tuple[str, str, float]]:
        """Find the best phonetic match for a candidate.

        Returns (station_id, canonical_name, distance) or None.
        """
        best_dist = 1.0
        best_sid = None
        best_name = None

        for sid, name, sig_ipa in self._signatures:
            dist = self._normalized_distance(candidate_ipa, sig_ipa)
            if dist < best_dist:
                best_dist = dist
                best_sid = sid
                best_name = name

        if best_sid is not None and best_dist < self._threshold:
            return best_sid, best_name, best_dist
        return None

    def correct(self, text: str) -> CorrectionResult:
        """Correct station names in a transcribed text.

        Two-pass approach:
        1. Mark words that already match known station names as protected.
        2. Generate n-grams only over unprotected spans, batch-phonemize, and correct.
        """
        words = text.split()
        n = len(words)
        if n == 0:
            return CorrectionResult(original_text=text, corrected_text=text)

        # --- Pass 1: protect words that already match a known station ---
        protected = [False] * n
        for i in range(n):
            cleaned = normalize_name(_PUNCT.sub("", words[i]).strip())
            if cleaned and cleaned in self._known_names:
                protected[i] = True

        # --- Pass 2: collect eligible n-grams ---
        # (start, end, cleaned_text)
        eligible: List[Tuple[int, int, str]] = []

        for ngram_size in range(1, min(5, n + 1)):
            for start in range(n - ngram_size + 1):
                end = start + ngram_size

                # Skip if any word in this span is protected (known station)
                if any(protected[i] for i in range(start, end)):
                    continue

                span_words = words[start:end]
                candidate = " ".join(span_words)

                # Skip if all words are stopwords
                if self._is_stopword(candidate):
                    continue

                # For multi-word n-grams, require >50% content words
                if ngram_size > 1 and self._content_word_ratio(span_words) <= 0.5:
                    continue

                # Skip very short single-word candidates (< 3 chars)
                cleaned = _PUNCT.sub("", candidate).strip()
                if ngram_size == 1 and len(cleaned) < 3:
                    continue

                eligible.append((start, end, cleaned))

        # Short-circuit: nothing to correct
        if not eligible:
            return CorrectionResult(original_text=text, corrected_text=text)

        # --- Batch phonemize all eligible candidates in one eSpeak call ---
        ipa_list = self._phonemize_batch([e[2] for e in eligible])

        candidates: List[Tuple[int, int, str, str, float]] = []
        for (start, end, cleaned), candidate_ipa in zip(eligible, ipa_list):
            if not candidate_ipa.strip():
                continue
            match = self._find_best_match(candidate_ipa, cleaned)
            if match:
                sid, name, dist = match
                candidates.append((start, end, name, sid, dist))

        if not candidates:
            return CorrectionResult(original_text=text, corrected_text=text)

        # Greedy selection: prefer longer matches, then lower distance
        candidates.sort(key=lambda c: (-(c[1] - c[0]), c[4]))

        used = [False] * n
        selected: List[Tuple[int, int, str, str, float]] = []

        for start, end, name, sid, dist in candidates:
            if any(used[i] for i in range(start, end)):
                continue
            selected.append((start, end, name, sid, dist))
            for i in range(start, end):
                used[i] = True

        # Apply replacements
        corrections: List[Correction] = []
        result_words = list(words)

        # Sort selected by start index descending to apply replacements safely
        selected.sort(key=lambda c: c[0], reverse=True)
        for start, end, name, sid, dist in selected:
            original_span = " ".join(words[start:end])
            # Don't replace if already correct (case-insensitive)
            if _PUNCT.sub("", original_span).strip().lower() == name.lower():
                continue
            corrections.append(
                Correction(
                    original=original_span,
                    corrected=name,
                    station_id=sid,
                    ipa_distance=dist,
                )
            )
            result_words[start:end] = [name]

        corrected_text = " ".join(result_words)
        # Sort corrections in order of appearance
        corrections.reverse()

        return CorrectionResult(
            original_text=text,
            corrected_text=corrected_text,
            corrections=corrections,
        )

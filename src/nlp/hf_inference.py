"""CamemBERT NER inference using ONNX Runtime for fast startup."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

LABELS = ["O", "B-DEPART", "I-DEPART", "B-ARRIVEE", "I-ARRIVEE", "B-VIA", "I-VIA", "B-DATE", "I-DATE"]

ONNX_MODEL_DIR = Path("data/models/camembert-ner-onnx")
PYTORCH_MODEL_DIR = Path("data/models/camembert-ner")


@dataclass
class HFSpans:
    depart: Optional[str]
    arrivee: Optional[str]
    vias: List[str]
    dates: List[str]


class HFExtractor:
    def __init__(self, model_dir: Path = ONNX_MODEL_DIR) -> None:
        self.model_dir = model_dir
        self._cache: dict[str, HFSpans] = {}
        self._session = None
        self._tokenizer = None
        self._load_event = threading.Event()
        self._load_started = False

    def start_preload(self) -> None:
        """Start loading the ONNX model in a background thread."""
        if self._load_started:
            return
        self._load_started = True
        if not (self.model_dir / "model.onnx").exists():
            self._load_event.set()
            return
        thread = threading.Thread(target=self._do_load, daemon=True)
        thread.start()

    def _do_load(self) -> None:
        """Load ONNX session and tokenizer."""
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            opts = ort.SessionOptions()
            # Use pre-optimized model if available (loads ~9x faster)
            optimized = self.model_dir / "model_optimized.onnx"
            if optimized.exists():
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                model_path = str(optimized)
            else:
                model_path = str(self.model_dir / "model.onnx")
            self._session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = Tokenizer.from_file(
                str(self.model_dir / "tokenizer.json")
            )
        except Exception:
            pass
        finally:
            self._load_event.set()

    def _ensure_loaded(self) -> None:
        """Wait for background loading to complete."""
        if not self._load_started:
            self.start_preload()
        self._load_event.wait()

    def is_ready(self) -> bool:
        return (self.model_dir / "model.onnx").exists()

    def extract(self, text: str) -> HFSpans:
        self._ensure_loaded()
        if not self._session or not self._tokenizer:
            return HFSpans(None, None, [], [])
        if text in self._cache:
            return self._cache[text]

        import numpy as np

        enc = self._tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)

        logits = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )[0][0]
        ids = np.argmax(logits, axis=-1).tolist()
        offsets = enc.offsets

        spans = {"DEPART": [], "ARRIVEE": [], "VIA": [], "DATE": []}
        current = None
        for idx, label_id in enumerate(ids):
            label = LABELS[label_id]
            if label == "O":
                current = None
                continue
            tag, role = label.split("-", 1)
            start, end = offsets[idx]
            if tag == "B":
                spans[role].append([start, end])
                current = spans[role][-1]
            elif tag == "I" and current is not None:
                current[1] = end
            else:
                current = None

        def substr(span_list):
            return [text[s:e] for s, e in span_list]

        deps = [s.strip() for s in substr(spans["DEPART"]) if s.strip()]
        arrs = [s.strip() for s in substr(spans["ARRIVEE"]) if s.strip()]
        raw_vias = [s.strip() for s in substr(spans["VIA"]) if s.strip()]
        raw_dates = [s.strip() for s in substr(spans.get("DATE", [])) if s.strip()]

        # Heuristic: reclassify VIA spans that are obviously temporal → DATE
        def looks_like_time(expr: str) -> bool:
            low = expr.lower()
            if re.search(r"\b\d{1,2}h\d{0,2}\b", low):
                return True
            if re.search(r"\b\d{1,2}:\d{2}\b", low):
                return True
            if re.search(r"\b(le\s+)?\d{1,2}[/-]\d{1,2}\b", low):
                return True
            month = r"janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre"
            if re.search(month, low):
                return True
            if re.search(r"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche", low):
                return True
            for kw in [
                "demain", "après-demain", "apres-demain", "aujourd'hui",
                "soir", "matin", "après-midi", "apres-midi",
                "weekend", "week-end", "semaine prochaine",
                "le matin", "le soir", "ce soir", "ce matin",
            ]:
                if kw in low:
                    return True
            return False

        vias: List[str] = []
        dates: List[str] = raw_dates[:]
        for v in raw_vias:
            if looks_like_time(v):
                dates.append(v)
            else:
                vias.append(v)

        # Fallback: if no departure predicted, try regex pattern
        if (not deps) or (deps and len(deps[0]) < 3):
            m = re.search(
                r"(?:de|depuis)\s+([-A-Za-zÀ-ÖØ-öø-ÿ'' ]+?)\s+(?:à|a|vers|pour)(?:\s|$)",
                text, re.IGNORECASE,
            )
            if m:
                candidate = m.group(1).strip()
                if not arrs or candidate.lower() != arrs[0].lower():
                    deps = [candidate]

        result = HFSpans(deps[0] if deps else None, arrs[0] if arrs else None, vias, dates)
        self._cache[text] = result
        return result

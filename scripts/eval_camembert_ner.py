"""
Evaluation rapide du modèle CamemBERT NER entraîné sur quelques phrases de test.

Usage :
  python scripts/eval_camembert_ner.py \
    --model data/models/camembert-ner \
    --samples "je veux aller de Lyon à Paris demain" \
              "trajet toulouse vers marseille via montpellier ce soir" \
              "un weekend en juin pour colmar strasbourg"

Par défaut, charge trois phrases de démonstration.
Affiche les spans DEPART / ARRIVEE / VIA / DATE détectés.
"""

from __future__ import annotations

import argparse
from typing import List

from src.nlp.hf_inference import HFExtractor
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="data/models/camembert-ner")
    ap.add_argument(
        "--samples",
        nargs="*",
        default=[
            "je veux aller de Lyon à Paris demain",
            "trajet toulouse vers marseille via montpellier ce soir",
            "un weekend en juin pour colmar strasbourg",
        ],
        help="Phrases à tester",
    )
    args = ap.parse_args()

    extractor = HFExtractor(Path(args.model))
    if not extractor.is_ready():
        print("❌ Modèle introuvable ou non chargé.")
        return

    for sent in args.samples:
        spans = extractor.extract(sent)
        print(f"\nPhrase: {sent}")
        print(f"  DEPART : {spans.depart}")
        print(f"  ARRIVEE: {spans.arrivee}")
        print(f"  VIA    : {spans.vias}")
        print(f"  DATE   : {spans.dates}")


if __name__ == "__main__":
    main()

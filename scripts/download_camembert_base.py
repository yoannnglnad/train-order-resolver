"""
Télécharge et met en cache le modèle CamemBERT base (OSCAR) pour un usage offline.

Usage :
  python scripts/download_camembert_base.py --out data/models/camembert-base

Si tu veux forcer un proxy ou un HF_HOME différent, exporte avant :
  HF_HOME=./data/.cache/hf_home
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import CamembertModel, CamembertTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/models/camembert-base"))
    args = ap.parse_args()

    model_id = "almanach/camembert-base"
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Téléchargement du tokenizer {model_id} ...")
    CamembertTokenizer.from_pretrained(model_id).save_pretrained(args.out)

    print(f"Téléchargement du modèle {model_id} ...")
    CamembertModel.from_pretrained(model_id).save_pretrained(args.out)

    print(f"✅ Modèle et tokenizer sauvegardés dans {args.out}")


if __name__ == "__main__":
    main()

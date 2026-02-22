"""Evaluation dataset: 25 test phrases for pipeline vs SNCF API accuracy."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

# Dynamic date: +7 days from now at 08:00 UTC to stay within SNCF API coverage
# (the API only covers ~21 days ahead).
# Phrases still say "le 15 mars" for NLP parsing, but both the API and pipeline
# use this computed timestamp so comparisons are fair.
_eval_dt = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=7)
EVAL_DATETIME = _eval_dt.strftime("%Y%m%dT%H%M%S")  # for SNCF API
EVAL_TIMESTAMP = int(_eval_dt.timestamp())  # Unix UTC for pipeline

# Cities where the default SNCF /places search returns the wrong station.
# e.g. "Lyon" → "Paris Gare de Lyon", "Lille" → "Lillers"
CITY_SEARCH_OVERRIDES = {
    "Lyon": "Lyon Part-Dieu",
    "Lille": "Lille Flandres",
    "Marseille": "Marseille Saint-Charles",
}

EVAL_QUERIES = [
    # ── TGV direct (6) ──
    {
        "id": "tgv_01",
        "category": "tgv_direct",
        "phrase": "Je voudrais aller de Paris à Lyon le 15 mars à 8h00",
        "expected_dep_city": "Paris",
        "expected_arr_city": "Lyon",
    },
    {
        "id": "tgv_02",
        "category": "tgv_direct",
        "phrase": "Trajet de Paris à Marseille le 15 mars à 8h00",
        "expected_dep_city": "Paris",
        "expected_arr_city": "Marseille",
    },
    {
        "id": "tgv_03",
        "category": "tgv_direct",
        "phrase": "Billet de Bordeaux à Paris le 15 mars à 8h00",
        "expected_dep_city": "Bordeaux",
        "expected_arr_city": "Paris",
    },
    {
        "id": "tgv_04",
        "category": "tgv_direct",
        "phrase": "Aller de Lille à Lyon le 15 mars à 8h00",
        "expected_dep_city": "Lille",
        "expected_arr_city": "Lyon",
    },
    {
        "id": "tgv_05",
        "category": "tgv_direct",
        "phrase": "Trajet Strasbourg vers Paris le 15 mars à 8h00",
        "expected_dep_city": "Strasbourg",
        "expected_arr_city": "Paris",
    },
    {
        "id": "tgv_06",
        "category": "tgv_direct",
        "phrase": "Je veux aller de Rennes à Bordeaux le 15 mars à 8h00",
        "expected_dep_city": "Rennes",
        "expected_arr_city": "Bordeaux",
    },
    # ── TER régional (5) ──
    {
        "id": "ter_01",
        "category": "ter_regional",
        "phrase": "Trajet de Metz à Strasbourg le 15 mars à 8h00",
        "expected_dep_city": "Metz",
        "expected_arr_city": "Strasbourg",
    },
    {
        "id": "ter_02",
        "category": "ter_regional",
        "phrase": "Aller de Grenoble à Lyon le 15 mars à 8h00",
        "expected_dep_city": "Grenoble",
        "expected_arr_city": "Lyon",
    },
    {
        "id": "ter_03",
        "category": "ter_regional",
        "phrase": "Je veux aller de Nantes à Rennes le 15 mars à 8h00",
        "expected_dep_city": "Nantes",
        "expected_arr_city": "Rennes",
    },
    {
        "id": "ter_04",
        "category": "ter_regional",
        "phrase": "Trajet de Nice à Marseille le 15 mars à 8h00",
        "expected_dep_city": "Nice",
        "expected_arr_city": "Marseille",
    },
    {
        "id": "ter_05",
        "category": "ter_regional",
        "phrase": "Aller de Montpellier à Toulouse le 15 mars à 8h00",
        "expected_dep_city": "Montpellier",
        "expected_arr_city": "Toulouse",
    },
    # ── Multi-étapes (5) ──
    {
        "id": "multi_01",
        "category": "multi_leg",
        "phrase": "Trajet de Bordeaux à Strasbourg le 15 mars à 8h00",
        "expected_dep_city": "Bordeaux",
        "expected_arr_city": "Strasbourg",
    },
    {
        "id": "multi_02",
        "category": "multi_leg",
        "phrase": "Je veux aller de Marseille à Lille le 15 mars à 8h00",
        "expected_dep_city": "Marseille",
        "expected_arr_city": "Lille",
    },
    {
        "id": "multi_03",
        "category": "multi_leg",
        "phrase": "Trajet de Nantes à Strasbourg le 15 mars à 8h00",
        "expected_dep_city": "Nantes",
        "expected_arr_city": "Strasbourg",
    },
    {
        "id": "multi_04",
        "category": "multi_leg",
        "phrase": "Aller de Marseille à Rennes le 15 mars à 8h00",
        "expected_dep_city": "Marseille",
        "expected_arr_city": "Rennes",
    },
    {
        "id": "multi_05",
        "category": "multi_leg",
        "phrase": "Je veux aller de Toulouse à Lille le 15 mars à 8h00",
        "expected_dep_city": "Toulouse",
        "expected_arr_city": "Lille",
    },
    # ── Edge cases (4) ──
    {
        "id": "edge_01",
        "category": "edge_case",
        "phrase": "Trajet de Paris Gare de Lyon à Dijon le 15 mars à 8h00",
        "expected_dep_city": "Paris",
        "expected_arr_city": "Dijon",
    },
    {
        "id": "edge_02",
        "category": "edge_case",
        "phrase": "Aller de Saint-Étienne à Lyon le 15 mars à 8h00",
        "expected_dep_city": "Saint-Étienne",
        "expected_arr_city": "Lyon",
    },
    {
        "id": "edge_03",
        "category": "edge_case",
        "phrase": "Trajet de Aix-en-Provence à Marseille le 15 mars à 8h00",
        "expected_dep_city": "Aix-en-Provence",
        "expected_arr_city": "Marseille",
    },
    {
        "id": "edge_04",
        "category": "edge_case",
        "phrase": "Aller de Avignon à Montpellier le 15 mars à 8h00",
        "expected_dep_city": "Avignon",
        "expected_arr_city": "Montpellier",
    },
    # ── Variantes de formulation (5) ──
    {
        "id": "phrasing_01",
        "category": "phrasing_variant",
        "phrase": "Je voudrais un train de Toulouse à Bordeaux le 15 mars à 8h00",
        "expected_dep_city": "Toulouse",
        "expected_arr_city": "Bordeaux",
    },
    {
        "id": "phrasing_02",
        "category": "phrasing_variant",
        "phrase": "Un billet de Lyon pour Marseille le 15 mars à 8h00",
        "expected_dep_city": "Lyon",
        "expected_arr_city": "Marseille",
    },
    {
        "id": "phrasing_03",
        "category": "phrasing_variant",
        "phrase": "Je souhaite réserver un trajet de Lille à Paris le 15 mars à 8h00",
        "expected_dep_city": "Lille",
        "expected_arr_city": "Paris",
    },
    {
        "id": "phrasing_04",
        "category": "phrasing_variant",
        "phrase": "Je dois me rendre de Nantes à Bordeaux le 15 mars à 8h00",
        "expected_dep_city": "Nantes",
        "expected_arr_city": "Bordeaux",
    },
    {
        "id": "phrasing_05",
        "category": "phrasing_variant",
        "phrase": "Voyage depuis Strasbourg jusqu'à Lyon le 15 mars à 8h00",
        "expected_dep_city": "Strasbourg",
        "expected_arr_city": "Lyon",
    },
]

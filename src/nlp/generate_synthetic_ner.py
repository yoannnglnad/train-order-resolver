"""Generate a synthetic NER dataset with DEPART/ARRIVEE/VIA/DATE labels, only on connected trips <= 6h."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import polars as pl
from tqdm import tqdm

from src.pathfinding.graph import build_graph
from src.pathfinding.algorithm import compute_route
from src.utils.config import DEFAULT_DATASET_PATH, DEFAULT_SCHEDULE_PATH

OUTPUT_FILE = Path("data/dataset/train_ner.jsonl")
N_SAMPLES = 12_000  # 10k valid + 2k trash approx
MAX_DURATION_MIN = 360  # 6h

LABELS_MAP = {
    "O": 0,
    "B-DEPART": 1,
    "I-DEPART": 2,
    "B-ARRIVEE": 3,
    "I-ARRIVEE": 4,
    "B-DATE": 5,
    "I-DATE": 6,
    "B-VIA": 7,
    "I-VIA": 8,
}

TEMPLATES_SIMPLE = [
    "Je veux aller de {D} à {A} {T}",
    "Je souhaite réserver un billet pour {A} au départ de {D} {T}",
    "Je voudrais partir de {D} pour aller à {A} {T}",
    "Recherche itinéraire {D} vers {A} pour {T}",
    "Affichez les trains en partance de {D} et à destination de {A} {T}",
    "Billet de train entre {D} et {A} {T}",
    "{T}, je dois me rendre à {A} depuis {D}",
    "Aller à {A} en partant de {D} {T}",
    "Destination {A}, origine {D} {T}",
    "Depuis {D}, je cherche à rejoindre {A} {T}",
    "{T} départ {D} arrivée {A}",
    "Je vise {A} {T} en venant de {D}",
    "Trajet {D} {A} {T}",
    "{D} vers {A} {T}",
    "Go {A} depuis {D} {T}",
    "Fais moi bouger de {D} à {A} {T}",
    "Horaires {D} {A} {T} stp",
    "Train {D} -> {A} {T}",
    "J'suis à {D}, j'veux aller à {A} {T}",
    "Je cherche un aller-retour {D} {A} {T}",
    "Trouve-moi un créneau {T} pour faire {D} {A}",
    "Y a quoi comme départ {T} de {D} pour {A} ?",
    "Donne moi les options de {D} à {A} {T}",
    "Dispo {T} pour voyager de {D} vers {A}",
]

TEMPLATES_VIA = [
    "Je veux aller de {D} à {A} en passant par {V} {T}",
    "Trajet {D} vers {A} via {V} {T}",
    "Billet {D} {A} passant par {V} {T}",
    "Un train de {D} pour {A} avec un arrêt à {V} {T}",
    "Itinéraire {D} -> {V} -> {A} {T}",
    "Je dois faire une escale à {V} sur mon trajet {D} vers {A} {T}",
    "On part de {D}, on s'arrête à {V} et on finit à {A} {T}",
    "Trouve un trajet {D} {A} qui passe par la gare de {V} {T}",
    "Aller à {A} depuis {D} mais avec une pause à {V} {T}",
    "En passant par {V}, je veux rallier {A} depuis {D} {T}",
    "{T} je pars de {D} pour {A} via {V}",
    "Destination {A}, départ {D}, passage par {V} {T}",
    "Depuis {D}, rejoindre {A} en transitant par {V} {T}",
]

TIME_EXPRESSIONS = [
    "demain",
    "aujourd'hui",
    "après-demain",
    "ce soir",
    "demain matin",
    "à 18h",
    "à 8h00",
    "vers 14h",
    "aux alentours de 9h",
    "le 10 juin",
    "le 1er janvier",
    "le 14 juillet",
    "lundi prochain",
    "mardi en huit",
    "ce vendredi",
    "un weekend en juin",
    "pendant les vacances de noël",
    "le premier weekend de mai",
    "en fin d'année",
    "courant août",
    "un de ces quatre",
    "la semaine prochaine",
    "ce weekend",
    "en soirée",
    "le matin uniquement",
    "le vendredi soir aussi",
    "le 12 mai à partir de 18h",
    "demain avant midi",
    "lundi matin vers 8h",
]

TEMPLATES_TRASH = [
    "J'aime manger des pommes {T}",
    "Quel temps fait-il à {A} ?",
    "La ville de {D} est très belle {T}",
    "Je suis allé à {D} hier",
    "Combien coûte une baguette ?",
    "Bonjour comment ça va",
    "C'est quoi la capitale de la France ?",
    "Il est quelle heure ?",
    "Le train de {T} est en retard",
    "Marseille contre Paris le match {T}",
]


def tag_time(tokens: List[str], labels: List[int], time_str: str) -> None:
    if not time_str:
        return
    time_tokens = time_str.split()
    if not time_tokens:
        return
    for i in range(len(tokens)):
        if tokens[i : i + len(time_tokens)] == time_tokens:
            labels[i] = LABELS_MAP["B-DATE"]
            for j in range(1, len(time_tokens)):
                labels[i + j] = LABELS_MAP["I-DATE"]
            break
    # Priorité : si un token est déjà B-VIA/I-VIA et correspond exactement à une expression temporelle simple,
    # on le requalifie en DATE (évite que VIA capture le temps).
    if len(time_tokens) == 1 and time_tokens[0].lower() in {t.lower() for t in TIME_EXPRESSIONS}:
        for i in range(len(tokens)):
            if tokens[i] == time_tokens[0] and labels[i] in (
                LABELS_MAP["B-VIA"],
                LABELS_MAP["I-VIA"],
            ):
                labels[i] = LABELS_MAP["B-DATE"]


def label_sentence(text: str, dep: str, arr: str, via: str | None, time_str: str) -> Tuple[List[str], List[int]]:
    tokens = text.split()
    labels = [LABELS_MAP["O"]] * len(tokens)
    dep_tokens = dep.split()
    arr_tokens = arr.split()
    via_tokens = via.split() if via else []

    def tag_entity(entity_tokens: List[str], label_b: str, label_i: str) -> None:
        for i in range(len(tokens)):
            if tokens[i : i + len(entity_tokens)] == entity_tokens:
                labels[i] = LABELS_MAP[label_b]
                for j in range(1, len(entity_tokens)):
                    labels[i + j] = LABELS_MAP[label_i]
                break

    tag_entity(dep_tokens, "B-DEPART", "I-DEPART")
    tag_entity(arr_tokens, "B-ARRIVEE", "I-ARRIVEE")
    if via_tokens:
        tag_entity(via_tokens, "B-VIA", "I-VIA")
    tag_time(tokens, labels, time_str)
    return tokens, labels


def main() -> None:
    stations = pl.read_parquet(DEFAULT_DATASET_PATH).select(["station_id", "name"])
    stations_list = stations.to_dicts()
    graph = build_graph(stations_path=DEFAULT_DATASET_PATH, schedule_path=DEFAULT_SCHEDULE_PATH)

    data = []
    attempts = 0
    trash_target = int(N_SAMPLES * 0.15)
    valid_target = N_SAMPLES - trash_target

    # valid samples
    while len(data) < valid_target and attempts < valid_target * 10:
        attempts += 1
        dep_row = random.choice(stations_list)
        arr_row = random.choice(stations_list)
        if dep_row["station_id"] == arr_row["station_id"]:
            continue
        dep_id, arr_id = dep_row["station_id"], arr_row["station_id"]
        try:
            path = compute_route(graph, dep_id, arr_id)
            weight = sum(graph.get_edge_data(u, v).get("weight", 0) for u, v in zip(path[:-1], path[1:]))
            if weight > MAX_DURATION_MIN:
                continue
        except Exception:
            continue

        dep_name = dep_row["name"]
        arr_name = arr_row["name"]
        time_str = random.choice(TIME_EXPRESSIONS) if random.random() < 0.7 else ""
        has_via = random.random() < 0.3
        via_name = None
        if has_via:
            via_row = random.choice(stations_list)
            while via_row["station_id"] in [dep_id, arr_id]:
                via_row = random.choice(stations_list)
            via_name = via_row["name"]
            tpl = random.choice(TEMPLATES_VIA)
            full_text = tpl.format(D=dep_name, A=arr_name, V=via_name, T=time_str).strip()
        else:
            tpl = random.choice(TEMPLATES_SIMPLE)
            full_text = tpl.format(D=dep_name, A=arr_name, T=time_str).strip()

        tokens, ner_tags = label_sentence(full_text, dep_name, arr_name, via_name, time_str)
        data.append({"id": f"VALID_{len(data)}", "tokens": tokens, "ner_tags": ner_tags, "text": full_text})

    # trash samples
    for idx in tqdm(range(trash_target), desc="Trash Data"):
        time_str = random.choice(TIME_EXPRESSIONS) if random.random() < 0.7 else ""
        tpl = random.choice(TEMPLATES_TRASH)
        rand_city = random.choice([s["name"] for s in stations_list])
        full_text = tpl.format(T=time_str, A=rand_city, D=rand_city)
        full_text = " ".join(full_text.split())
        tokens = full_text.split()
        ner_tags = [LABELS_MAP["O"]] * len(tokens)
        data.append({"id": f"TRASH_{idx}", "tokens": tokens, "ner_tags": ner_tags, "text": full_text})

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(data)} samples (attempted {attempts} valid attempts) to {OUTPUT_FILE}")
    print("Example:", json.dumps(data[0], ensure_ascii=False)[:200])


if __name__ == "__main__":
    main()

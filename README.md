# Travel Order Resolver

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![NLP](https://img.shields.io/badge/NLP-CamemBERT-yellow)
![Algorithme](https://img.shields.io/badge/Algorithme-Dijkstra-green)
![STT](https://img.shields.io/badge/STT-Whisper-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Systeme de resolution d'ordres de voyage ferroviaires par traitement du langage naturel, reconnaissance vocale et theorie des graphes. Developpe dans le cadre du module "Artificial Intelligence" (Epitech).

> La documentation technique complete est disponible dans [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

---

## Fonctionnalites

- **Commande vocale** : enregistrement audio transcrit par Whisper (MLX/faster-whisper) avec correction phonetique IPA des noms de gares
- **Extraction NLP** : CamemBERT fine-tune pour la reconnaissance d'entites nommees (depart, arrivee, correspondances, horaires)
- **Calcul d'itineraire** : graphe de 3 907 gares et 19 505 aretes (donnees SNCF GTFS) parcouru par Dijkstra
- **Visualisation cartographique** : carte Leaflet montrant l'exploration du graphe et le trajet optimal
- **Interface web** : frontend Next.js + API FastAPI

---

## Demarrage rapide

### Prerequisites

- Python 3.10+
- Node.js 18+ (pour l'interface web)
- eSpeak-NG (pour la correction phonetique)

```bash
# macOS
brew install espeak-ng

# Ubuntu/Debian
sudo apt-get install espeak-ng
```

### Installation

```bash
git clone https://github.com/votre-username/travel-order-resolver.git
cd travel-order-resolver

# Environnement Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend (optionnel)
cd web && npm install && cd ..
```

### Preparation des donnees

```bash
# 1. Generer le dataset NLP synthetique
python scripts/prepare_dataset.py

# 2. Construire l'index phonetique
python scripts/build_phonetic_db.py
```

### Utilisation

#### Mode CLI

```bash
# Fichier d'entrees
python main.py --input inputs.txt

# Entree standard
echo "1,Je voudrais aller de Paris a Lyon demain a 8h30" | python main.py
```

**Format d'entree :** `ID,Phrase`
**Format de sortie :** `ID,Depart,Etape1,...,Destination` ou `ID,INVALID`

#### Mode Web (vocal)

```bash
# Terminal 1 : API
python api/server.py

# Terminal 2 : Frontend
cd web && npm run dev
```

Ouvrir `http://localhost:3000`, cliquer sur le bouton d'enregistrement et dicter un itineraire.

---

## Architecture

```
travel-order-resolver/
├── api/
│   └── server.py              # API REST FastAPI
├── data/
│   ├── raw/                   # Fichiers SNCF (parquet, CSV)
│   ├── dataset/               # Dataset NER synthetique
│   ├── models/                # Poids CamemBERT fine-tune
│   └── cache/                 # Index phonetique, graphe serialise
├── src/
│   ├── nlp/                   # Extraction d'entites (CamemBERT NER)
│   │   ├── inference.py       # TravelResolver - pipeline complet
│   │   ├── hf_inference.py    # Inference CamemBERT ONNX
│   │   ├── train_camembert.py # Fine-tuning NER
│   │   └── generate_synthetic_ner.py
│   ├── pathfinding/           # Graphe et routage
│   │   ├── graph.py           # Construction du graphe NetworkX
│   │   ├── algorithm.py       # Dijkstra + exploration tracking
│   │   └── prepare_stations.py
│   ├── stt/                   # Speech-to-Text
│   │   ├── transcriber.py     # Whisper (MLX / faster-whisper / HF)
│   │   ├── phonetic_corrector.py  # Correction IPA
│   │   └── phonetic_db.py     # Index phonetique eSpeak-NG
│   └── utils/                 # Configuration, cache, logging
├── web/                       # Frontend Next.js
│   └── app/
│       ├── page.tsx           # Page principale (enregistrement vocal)
│       └── components/
│           └── RouteMap.tsx    # Carte Leaflet
├── tests/                     # Tests unitaires et d'integration
├── scripts/                   # Scripts utilitaires et benchmarks
├── docs/
│   └── DOCUMENTATION.md       # Documentation technique detaillee
├── main.py                    # Point d'entree CLI
└── requirements.txt
```

---

## Pipeline de traitement

```
Audio (.webm) ──► Whisper ──► Texte brut
                                  │
                                  ▼
                          Correction phonetique IPA
                                  │
                                  ▼
                          CamemBERT NER ──► Entites (depart, arrivee, horaire)
                                                    │
                                                    ▼
                                            Dijkstra sur graphe SNCF
                                                    │
                                                    ▼
                                          Itineraire + carte Leaflet
```

---

## Tests

```bash
pytest tests/ -v
```

Les tests couvrent : extraction NER (25 cas), construction du graphe, routage (y compris temps-dependant), correction phonetique et pipeline STT complet.

---

## Documentation

La documentation technique exhaustive se trouve dans [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md). Elle couvre :

- Definition formelle du probleme et etat de l'art
- Architecture detaillee de chaque module
- Pseudocode des algorithmes (Dijkstra, correction phonetique)
- Specifications de l'API REST
- Resultats d'evaluation et metriques de performance
- Guide d'installation complet

---

## Sources de donnees

- [Liste des gares SNCF](https://ressources.data.sncf.com/explore/dataset/liste-des-gares/table/)
- [Horaires TGV/Intercites (GTFS)](https://ressources.data.sncf.com/)

## Auteur

Yoann NGUENLO - Epitech 2026

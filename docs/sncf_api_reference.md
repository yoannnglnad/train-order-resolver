# SNCF / Navitia API -- Reference

## Configuration

```
# .env
SNCF_API_KEY=your-api-key-here
SNCF_API_BASE_URL=https://api.sncf.com/v1
```

## Authentification

HTTP Basic Auth : la clé API est le **username**, le password est **vide**.

```bash
# Header
curl -H "Authorization: $SNCF_API_KEY" https://api.sncf.com/v1/coverage/sncf/

# Flag -u
curl -u "$SNCF_API_KEY:" https://api.sncf.com/v1/coverage/sncf/

# URL
curl https://$SNCF_API_KEY@api.sncf.com/v1/coverage/sncf/
```

En Python :

```python
import os, requests

API_KEY = os.getenv("SNCF_API_KEY")
BASE = os.getenv("SNCF_API_BASE_URL", "https://api.sncf.com/v1")

r = requests.get(f"{BASE}/coverage/sncf/places", params={"q": "Lyon"}, auth=(API_KEY, ""))
data = r.json()
```

---

## Concepts clefs

| Terme | Description |
|-------|-------------|
| **Coverage** | Région géographique (ici `sncf`) |
| **Stop Area** | Gare / groupe de quais |
| **Stop Point** | Quai ou point d'arrêt individuel |
| **Line** | Ligne de transport (ex: TGV Paris-Lyon) |
| **Route** | Variante directionnelle d'une ligne |
| **Vehicle Journey** | Un trajet planifié précis |
| **Commercial Mode** | Mode côté voyageur (TGV, TER, Intercités...) |
| **Physical Mode** | Technologie sous-jacente (Train, Bus...) |

---

## Paramètres communs

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `depth` | int | 1 | Verbosité (0-3) |
| `count` | int | 25 | Résultats par page (max 200) |
| `start_page` | int | 0 | Numéro de page |
| `disable_geojson` | bool | false | Supprimer le GeoJSON |
| `disable_disruption` | bool | false | Supprimer les perturbations |
| `data_freshness` | enum | `base_schedule` | `realtime` ou `base_schedule` |
| `filter` | string | -- | Expression de filtre |

### Pagination

```json
{
  "pagination": {
    "items_per_page": 25,
    "items_on_page": 25,
    "start_page": 0,
    "total_result": 1921
  }
}
```

Les liens HATEOAS `first`, `previous`, `next`, `last` sont inclus dans la réponse.

### Profondeur (`depth`)

| Niveau | Comportement |
|--------|-------------|
| 0 | Minimal -- objets parents supprimés |
| 1 | Standard (défaut) |
| 2 | GeoJSON et données géographiques inclus |
| 3 | Détail complet (tous les stop_points, GeoJSON complet) |

---

## Endpoints

### 1. Coverage (découverte)

```
GET /coverage                         # Toutes les régions
GET /coverage/sncf                    # Détails SNCF
GET /coverage/sncf/datasets           # Sources de données
GET /coverage/sncf/contributors       # Fournisseurs de données
```

---

### 2. Places (autocomplete / recherche)

```
GET /coverage/sncf/places?q={terme}
```

| Paramètre | Type | Description |
|-----------|------|-------------|
| `q` | string | **Requis.** Terme de recherche |
| `type[]` | array | Filtrer : `stop_area`, `address`, `administrative_region`, `poi`, `stop_point` |
| `from` | string | Coordonnées `lon;lat` pour prioriser la proximité |

**Pas de pagination** sur cet endpoint.

**Exemple :**
```
GET /coverage/sncf/places?q=Paris+Gare+de+Lyon&type[]=stop_area
```

---

### 3. PT Objects (recherche transport public)

```
GET /coverage/sncf/pt_objects?q={terme}
```

| Paramètre | Type | Description |
|-----------|------|-------------|
| `q` | string | **Requis.** |
| `type[]` | array | `network`, `commercial_mode`, `line`, `route`, `stop_area`, `stop_point` |

---

### 4. Journeys (itinéraires)

```
GET /coverage/sncf/journeys?from={id}&to={id}
GET /journeys?from={id}&to={id}
```

#### Paramètres principaux

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `from` | id | -- | Départ (stop_area ID, adresse, ou `lon;lat`) |
| `to` | id | -- | Arrivée |
| `datetime` | string | maintenant | Format : `YYYYMMDDTHHMMSS` |
| `datetime_represents` | enum | `departure` | `departure` ou `arrival` |
| `traveler_type` | enum | `standard` | `standard`, `slow_walker`, `fast_walker`, `luggage`, `wheelchair` |

#### Modes de transport

| Paramètre | Type | Description |
|-----------|------|-------------|
| `first_section_mode[]` | array | Avant le transport : `walking`, `bike`, `car`, `bss`, `ridesharing`, `taxi` |
| `last_section_mode[]` | array | Après le transport |
| `direct_path` | enum | `indifferent`, `none`, `only` |

#### Filtrage

| Paramètre | Description |
|-----------|-------------|
| `forbidden_uris[]` | Exclure des lignes, modes, réseaux, stop_points |
| `allowed_id[]` | Restreindre à des objets spécifiques |
| `wheelchair` | Transport accessible uniquement |

#### Contrôle des résultats

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `count` | int | -- | Nombre fixe de solutions |
| `min_nb_journeys` | int | -- | Minimum de solutions |
| `max_nb_journeys` | int | -- | Maximum de solutions |
| `max_nb_transfers` | int | 10 | Correspondances max |
| `max_duration` | int | 86400 | Durée max du trajet (secondes) |

#### Structure de la réponse

```json
{
  "journeys": [
    {
      "type": "comfort | fast | ...",
      "duration": 2671,
      "nb_transfers": 0,
      "departure_date_time": "20260217T133830",
      "arrival_date_time": "20260217T142301",
      "co2_emission": { "unit": "gEC", "value": 24.642 },
      "status": "",
      "sections": [
        {
          "type": "public_transport | street_network | transfer | waiting",
          "departure_date_time": "...",
          "arrival_date_time": "...",
          "duration": 600,
          "from": { "...": "..." },
          "to": { "...": "..." },
          "display_informations": { "...": "..." },
          "stop_date_times": []
        }
      ]
    }
  ],
  "disruptions": [],
  "links": []
}
```

**Status du journey :** vide = OK, `NO_SERVICE` = supprimé, `MODIFIED_SERVICE` = modifié, `SIGNIFICANT_DELAYS` = retardé.

**Exemple :**
```
GET /coverage/sncf/journeys?from=admin:fr:75056&to=admin:fr:69123&datetime=20260217T080000
```

---

### 5. Departures / Arrivals (prochains départs/arrivées)

```
GET /coverage/sncf/stop_areas/{stop_area_id}/departures
GET /coverage/sncf/stop_areas/{stop_area_id}/arrivals
GET /coverage/sncf/stop_points/{stop_point_id}/departures
```

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `datetime` | string | maintenant | Date/heure de référence |
| `count` | int | 10 | Résultats par page |
| `data_freshness` | enum | `base_schedule` | `realtime` pour temps réel |

**Exemple :**
```
GET /coverage/sncf/stop_areas/stop_area:SNCF:87391003/departures?count=10&data_freshness=realtime
```

---

### 6. Stop Schedules (horaires d'arrêt)

```
GET /coverage/sncf/stop_areas/{id}/stop_schedules
```

| Paramètre | Type | Description |
|-----------|------|-------------|
| `from_datetime` | string | Début de période |
| `until_datetime` | string | Fin de période |
| `items_per_schedule` | int | Prochains départs par route |

---

### 7. Route Schedules (fiches horaires)

```
GET /coverage/sncf/routes/{route_id}/route_schedules
```

Retourne la grille horaire complète d'une route avec tous les arrêts.

---

### 8. Collections d'objets PT

Toutes suivent le même pattern :

```
GET /coverage/sncf/{collection}
GET /coverage/sncf/{collection}/{id}
```

| Collection | Description |
|------------|-------------|
| `networks` | Opérateurs |
| `lines` | Lignes |
| `routes` | Routes directionnelles |
| `stop_areas` | Gares |
| `stop_points` | Quais |
| `commercial_modes` | Modes commerciaux (TGV, TER...) |
| `physical_modes` | Modes physiques (Train, Bus...) |
| `companies` | Exploitants |
| `vehicle_journeys` | Trajets planifiés |
| `disruptions` | Perturbations |

**Navigation imbriquée :**
```
GET /coverage/sncf/lines/{line_id}/routes
GET /coverage/sncf/lines/{line_id}/stop_areas
GET /coverage/sncf/stop_areas/{id}/lines
GET /coverage/sncf/commercial_modes/commercial_mode:TGV/lines
```

---

### 9. Places Nearby (lieux à proximité)

```
GET /coverage/sncf/coords/{lon;lat}/places_nearby
GET /coverage/sncf/stop_areas/{id}/places_nearby
```

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `distance` | int | 500 | Rayon en mètres |
| `type[]` | array | tous | Types d'objets |

---

### 10. Disruptions (perturbations)

```
GET /coverage/sncf/disruptions
GET /coverage/sncf/disruptions?since=20260217T000000&until=20260217T235959
```

**Effets possibles :** `NO_SERVICE`, `REDUCED_SERVICE`, `MODIFIED_SERVICE`, `ADDITIONAL_SERVICE`, `SIGNIFICANT_DELAYS`, `UNKNOWN_EFFECT`

---

### 11. Isochrones

```
GET /coverage/sncf/isochrones?from={lon;lat}&boundary_duration[]=3600
```

Retourne des polygones GeoJSON pour la visualisation cartographique.

---

### 12. Reports

```
GET /coverage/sncf/line_reports          # Perturbations par ligne
GET /coverage/sncf/traffic_reports       # Aperçu du trafic
GET /coverage/sncf/equipment_reports     # État des équipements (ascenseurs, escalators)
```

---

## Filtrage avancé

```
# Par code de gare
?filter=stop_area.has_code(source,87391003)

# Par code de ligne
?filter=line.code=TGV

# Combinaison
?forbidden_uris[]=line:SNCF:xxx
?allowed_id[]=network:SNCF:yyy
```

---

## Temps réel

- `data_freshness=base_schedule` (défaut) : horaires théoriques
- `data_freshness=realtime` : inclut les perturbations en temps réel

---

## Gestion des erreurs

| Code HTTP | Description |
|-----------|-------------|
| 200 | Succès |
| 4xx | Erreur client (paramètres invalides, ID inexistant) |
| 5xx | Erreur serveur |

**Limites :** URL max 4096 caractères. Les IDs ne sont pas stables entre mises à jour -- préférer `/places` pour résoudre les noms.

---

## Exemples pratiques

```bash
# Rechercher une gare
curl -u "$SNCF_API_KEY:" "https://api.sncf.com/v1/coverage/sncf/places?q=Lyon+Part+Dieu&type[]=stop_area"

# Itinéraire Paris -> Lyon
curl -u "$SNCF_API_KEY:" "https://api.sncf.com/v1/coverage/sncf/journeys?from=admin:fr:75056&to=admin:fr:69123&datetime=20260217T080000"

# Prochains départs Montparnasse (temps réel)
curl -u "$SNCF_API_KEY:" "https://api.sncf.com/v1/coverage/sncf/stop_areas/stop_area:SNCF:87391003/departures?count=10&data_freshness=realtime"

# Toutes les lignes TGV
curl -u "$SNCF_API_KEY:" "https://api.sncf.com/v1/coverage/sncf/commercial_modes/commercial_mode:TGV/lines"

# Perturbations du jour
curl -u "$SNCF_API_KEY:" "https://api.sncf.com/v1/coverage/sncf/disruptions?since=20260217T000000&until=20260217T235959"

# Gares proches de coordonnées
curl -u "$SNCF_API_KEY:" "https://api.sncf.com/v1/coverage/sncf/coords/2.373;48.844/places_nearby?distance=1000&type[]=stop_area"
```

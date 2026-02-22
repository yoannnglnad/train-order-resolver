from src.pathfinding.prepare_stations import normalize_name, select_columns

import polars as pl


def test_normalize_name_removes_accents_and_punctuation() -> None:
    assert normalize_name("Gare d'Évry-Courcouronnes") == "gare d evry courcouronnes"
    assert normalize_name("Porte Maillot") == "porte maillot"


def test_select_columns_filters_and_shapes_dataframe() -> None:
    frame = pl.DataFrame(
        {
            "code_uic": ["100", "101"],
            "libelle": ["Gare Test", "Fret Only"],
            "commune": ["Paris", "Lyon"],
            "departemen": ["Paris", "Rhône"],
            "x_wgs84": [2.0, 4.0],
            "y_wgs84": [48.0, 45.0],
            "voyageurs": ["O", "N"],
            "fret": ["N", "O"],
        }
    )

    curated = select_columns(frame)
    assert curated.height == 1
    row = curated.to_dicts()[0]
    assert row["station_id"] == "100"
    assert row["name"] == "Gare Test"
    assert row["city"] == "Paris"
    assert row["lat"] == 48.0
    assert row["lon"] == 2.0
    assert row["passengers"] == "O"
    assert row["freight"] == "N"
    assert row["name_norm"] == "gare test"

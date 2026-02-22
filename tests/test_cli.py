import csv
from pathlib import Path
from typing import List

import polars as pl

from main import main


def write_sample_stations(path: Path) -> None:
    frame = pl.DataFrame(
        {
            "station_id": ["100", "200"],
            "name": ["Toulouse", "Paris"],
            "city": ["Toulouse", "Paris"],
            "department": ["31", "75"],
            "lat": [43.6, 48.8],
            "lon": [1.44, 2.35],
            "passengers": ["O", "O"],
            "freight": ["N", "N"],
            "name_norm": ["toulouse", "paris"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_cli_end_to_end(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "stations.parquet"
    write_sample_stations(dataset_path)
    connections_path = tmp_path / "connections.parquet"
    pl.DataFrame({"from_id": ["100"], "to_id": ["200"], "weight": [100.0]}).write_parquet(connections_path)

    input_path = tmp_path / "input.csv"
    with input_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentenceID", "phrase"])
        writer.writerow(["1", "Je veux aller de Toulouse à Paris"])

    output_path = tmp_path / "output.csv"
    args: List[str] = [
        "main.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--dataset",
        str(dataset_path),
        "--connections",
        str(connections_path),
        "--k-neighbors",
        "1",
        "--no-log-stdout",
    ]
    monkeypatch.setattr("sys.argv", args)
    assert main() == 0

    rows = list(csv.DictReader(output_path.open()))
    assert rows[0]["depart"] == "Toulouse"
    assert rows[0]["arrivee"] == "Paris"

from pathlib import Path


def test_project_scaffold_exists() -> None:
    """Ensure expected project layout is present."""
    expected_paths = [
        Path("src/nlp"),
        Path("src/pathfinding"),
        Path("src/utils"),
        Path("tests/fixtures"),
        Path("data"),
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing expected scaffold path: {path}"

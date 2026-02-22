"""Tests for the STT module: phonetic DB, corrector, and transcriber."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

# Skip entire module if espeak-ng is not installed
_espeak_available = os.system("espeak-ng --version > /dev/null 2>&1") == 0
pytestmark = pytest.mark.skipif(not _espeak_available, reason="espeak-ng not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Minimal station data for building a test index
_MINI_STATIONS = [
    ("87192039", "Metz", "metz", "Metz"),
    ("87757559", "Cannes", "cannes", "Cannes"),
    ("87743716", "Bourg-en-Bresse", "bourg en bresse", "Bourg-en-Bresse"),
    ("87686006", "Paris Gare de Lyon", "paris gare de lyon", "Paris"),
    ("87286009", "Lille Flandres", "lille flandres", "Lille"),
    ("87722025", "Lyon Part-Dieu", "lyon part dieu", "Lyon"),
    ("87611004", "Toulouse Matabiau", "toulouse matabiau", "Toulouse"),
    ("87751008", "Marseille Saint-Charles", "marseille saint charles", "Marseille"),
    ("87481002", "Nantes", "nantes", "Nantes"),
    ("87212027", "Strasbourg", "strasbourg", "Strasbourg"),
    ("87581009", "Bordeaux Saint-Jean", "bordeaux saint jean", "Bordeaux"),
]


@pytest.fixture(scope="module")
def mini_stations_path(tmp_path_factory):
    """Create a minimal stations.parquet for testing."""
    tmp = tmp_path_factory.mktemp("stt")
    df = pl.DataFrame(
        {
            "station_id": [s[0] for s in _MINI_STATIONS],
            "name": [s[1] for s in _MINI_STATIONS],
            "name_norm": [s[2] for s in _MINI_STATIONS],
            "city": [s[3] for s in _MINI_STATIONS],
            "department": ["57", "06", "01", "75", "59", "69", "31", "13", "44", "67", "33"],
            "lat": [49.1, 43.5, 46.2, 48.8, 50.6, 45.7, 43.6, 43.3, 47.2, 48.5, 44.8],
            "lon": [6.1, 7.0, 5.2, 2.3, 3.0, 4.8, 1.4, 5.3, -1.5, 7.7, -0.5],
            "passengers": ["O"] * 11,
            "freight": ["N"] * 11,
        }
    )
    path = tmp / "stations.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture(scope="module")
def phonetic_index_path(mini_stations_path, tmp_path_factory):
    """Build phonetic index from mini stations."""
    from src.stt.phonetic_db import build_phonetic_index

    tmp = tmp_path_factory.mktemp("stt_index")
    output = tmp / "phonetic_index.json"
    build_phonetic_index(stations_path=mini_stations_path, output_path=output)
    return output


@pytest.fixture(scope="module")
def phonetic_index(phonetic_index_path):
    """Load the built phonetic index."""
    from src.stt.phonetic_db import load_phonetic_index

    return load_phonetic_index(phonetic_index_path)


@pytest.fixture(scope="module")
def corrector(phonetic_index_path, phonetic_index):
    """Create a PhoneticCorrector using the test index."""
    from src.stt.phonetic_corrector import PhoneticCorrector

    return PhoneticCorrector(index_path=phonetic_index_path, index=phonetic_index)


# ---------------------------------------------------------------------------
# Tests: Phonetic DB
# ---------------------------------------------------------------------------


class TestPhoneticDB:
    def test_phonemize_known_stations(self):
        """eSpeak produces coherent IPA for known French cities."""
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator

        backend = EspeakBackend(
            language="fr-fr",
            preserve_punctuation=False,
            with_stress=False,
            language_switch="remove-flags",
        )
        sep = Separator(phone=" ", word=" _ ", syllable="")
        results = backend.phonemize(["Metz", "Paris", "Cannes"], separator=sep, strip=True)

        assert len(results) == 3
        for ipa in results:
            # Should produce non-empty IPA
            assert len(ipa.strip()) > 0

    def test_build_and_load_index(self, phonetic_index):
        """The index contains the expected entries with required fields."""
        assert len(phonetic_index) >= len(_MINI_STATIONS) - 1  # may have dedup on station_id

        required_fields = {"name", "name_norm", "city", "ipa_name", "ipa_name_norm", "ipa_city"}
        for sid, entry in phonetic_index.items():
            assert required_fields.issubset(entry.keys()), f"Missing fields in {sid}: {entry.keys()}"
            # IPA fields should be non-empty
            assert entry["ipa_name"].strip(), f"Empty ipa_name for {sid}"

    def test_index_contains_specific_stations(self, phonetic_index):
        """Verify specific station IDs are in the index."""
        # Metz
        assert "87192039" in phonetic_index
        assert phonetic_index["87192039"]["name"] == "Metz"


# ---------------------------------------------------------------------------
# Tests: Phonetic Corrector
# ---------------------------------------------------------------------------


class TestPhoneticCorrector:
    def test_correct_metz_from_mess(self, corrector):
        """'mess' should be corrected to 'Metz' (same IPA: /mɛs/)."""
        result = corrector.correct("je veux aller à mess")
        assert "Metz" in result.corrected_text
        assert len(result.corrections) >= 1
        metz_correction = [c for c in result.corrections if c.corrected == "Metz"]
        assert len(metz_correction) == 1

    def test_correct_cannes_from_canne(self, corrector):
        """'canne' should be corrected to 'Cannes'."""
        result = corrector.correct("un billet pour canne")
        assert "Cannes" in result.corrected_text

    def test_correct_multiword(self, corrector):
        """'bourg en braise' should be corrected to 'Bourg-en-Bresse'."""
        result = corrector.correct("je pars de bourg en braise")
        assert "Bourg-en-Bresse" in result.corrected_text

    def test_no_false_positive_stopwords(self, corrector):
        """Sentences with only common French words should not be corrected."""
        result = corrector.correct("Je veux un billet de train")
        assert result.corrected_text == result.original_text
        assert len(result.corrections) == 0

    def test_preserves_original_when_no_match(self, corrector):
        """Unknown words far from any station should not be corrected."""
        result = corrector.correct("abracadabra xylophone")
        assert len(result.corrections) == 0

    def test_correction_result_fields(self, corrector):
        """CorrectionResult has all expected fields."""
        result = corrector.correct("mess")
        assert hasattr(result, "original_text")
        assert hasattr(result, "corrected_text")
        assert hasattr(result, "corrections")
        assert result.original_text == "mess"


# ---------------------------------------------------------------------------
# Tests: Transcriber
# ---------------------------------------------------------------------------


class TestTranscriber:
    def test_transcriber_missing_file(self):
        """FileNotFoundError raised for non-existent audio file."""
        from src.stt.transcriber import Transcriber

        t = Transcriber()
        with pytest.raises(FileNotFoundError):
            t.transcribe("/nonexistent/audio.wav")

    def test_transcription_result_dataclass(self):
        """TranscriptionResult has expected fields."""
        from src.stt.transcriber import TranscriptionResult

        r = TranscriptionResult(text="bonjour", language="fr", segments=[], duration_sec=1.5)
        assert r.text == "bonjour"
        assert r.language == "fr"
        assert r.duration_sec == 1.5


# ---------------------------------------------------------------------------
# Tests: End-to-end (simulated)
# ---------------------------------------------------------------------------


class TestEndToEndSimulated:
    def test_correction_pipeline_simulated(self, corrector):
        """Simulated STT output → correction → produces valid station names."""
        # Simulate Whisper producing "mess" instead of "Metz"
        raw_text = "je veux aller de Paris à mess"
        result = corrector.correct(raw_text)

        # Should contain corrected station names
        assert "Metz" in result.corrected_text
        # Paris should remain (already correct or close enough)
        assert "Paris" in result.corrected_text or "paris" in result.corrected_text.lower()

    def test_fixtures_csv_corrections(self, corrector):
        """Test corrections from the fixtures CSV file."""
        import csv

        csv_path = FIXTURES_DIR / "stt_corrections.csv"
        if not csv_path.exists():
            pytest.skip("stt_corrections.csv fixture not found")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                erroneous = row["transcription_erronee"]
                expected = row["gare_attendue"]
                result = corrector.correct(erroneous)
                assert expected in result.corrected_text, (
                    f"Expected '{expected}' in corrected text for input '{erroneous}', "
                    f"got '{result.corrected_text}'"
                )

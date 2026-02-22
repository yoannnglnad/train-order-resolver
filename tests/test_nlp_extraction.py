"""Tests for NLP extraction accuracy using the trained CamemBERT NER model."""

import pytest

from src.nlp.inference import TravelResolver


@pytest.fixture(scope="module")
def resolver():
    return TravelResolver()


# ---------------------------------------------------------------------------
# Basic from/to extraction
# ---------------------------------------------------------------------------


class TestBasicExtraction:
    def test_de_paris_a_lyon(self, resolver):
        order = resolver.resolve_order("t1", "Je veux aller de Paris à Lyon")
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "paris" in dep_name
        assert "lyon" in arr_name

    def test_trajet_bordeaux_vers_marseille(self, resolver):
        order = resolver.resolve_order("t2", "Trajet Bordeaux vers Marseille")
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "bordeaux" in dep_name
        assert "marseille" in arr_name

    def test_depuis_lille_pour_strasbourg(self, resolver):
        order = resolver.resolve_order(
            "t3", "Je pars depuis Lille pour Strasbourg"
        )
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "lille" in dep_name
        assert "strasbourg" in arr_name

    def test_billet_grenoble_a_lyon(self, resolver):
        order = resolver.resolve_order("t4", "Billet de Grenoble à Lyon")
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "grenoble" in dep_name
        assert "lyon" in arr_name


# ---------------------------------------------------------------------------
# Various French phrasings
# ---------------------------------------------------------------------------


class TestFrenchPhrasings:
    def test_aller_a_nantes_depuis_rennes(self, resolver):
        order = resolver.resolve_order(
            "t5", "Je voudrais aller à Nantes depuis Rennes"
        )
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "rennes" in dep_name
        assert "nantes" in arr_name

    def test_billet_bordeaux_a_toulouse(self, resolver):
        order = resolver.resolve_order(
            "t6", "Billet de Bordeaux à Toulouse"
        )
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "bordeaux" in dep_name
        assert "toulouse" in arr_name

    def test_de_metz_a_strasbourg(self, resolver):
        order = resolver.resolve_order("t7", "De Metz à Strasbourg")
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "metz" in dep_name
        assert "strasbourg" in arr_name

    def test_nice_vers_marseille(self, resolver):
        order = resolver.resolve_order("t8", "Nice vers Marseille")
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "nice" in dep_name
        assert "marseille" in arr_name


# ---------------------------------------------------------------------------
# VIA extraction
# ---------------------------------------------------------------------------


class TestViaExtraction:
    def test_via_dijon(self, resolver):
        order = resolver.resolve_order(
            "t9", "De Paris à Marseille via Dijon"
        )
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "paris" in dep_name
        assert "marseille" in arr_name

    def test_en_passant_par(self, resolver):
        order = resolver.resolve_order(
            "t10", "De Lille à Bordeaux en passant par Paris"
        )
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "lille" in dep_name
        assert "bordeaux" in arr_name


# ---------------------------------------------------------------------------
# Trash / invalid input → is_valid=False
# ---------------------------------------------------------------------------


class TestInvalidInput:
    def test_greeting_invalid(self, resolver):
        order = resolver.resolve_order("t11", "Bonjour comment ça va")
        assert not order.is_valid

    def test_weather_invalid(self, resolver):
        order = resolver.resolve_order("t12", "Quel temps fait-il ?")
        assert not order.is_valid

    def test_random_question(self, resolver):
        order = resolver.resolve_order(
            "t13", "Combien coûte un café au lait ?"
        )
        assert not order.is_valid

    def test_empty_string(self, resolver):
        order = resolver.resolve_order("t14", "")
        assert not order.is_valid


# ---------------------------------------------------------------------------
# Accents and special characters
# ---------------------------------------------------------------------------


class TestAccentsAndSpecialChars:
    def test_accented_cities(self, resolver):
        order = resolver.resolve_order(
            "t15", "Aller de Montpellier à Béziers"
        )
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        assert "montpellier" in dep_name

    def test_apostrophe(self, resolver):
        order = resolver.resolve_order(
            "t16", "De Paris à Grenoble"
        )
        assert order.is_valid
        dep_name = resolver.id_to_name[order.departure_id].lower()
        arr_name = resolver.id_to_name[order.arrival_id].lower()
        assert "paris" in dep_name
        assert "grenoble" in arr_name

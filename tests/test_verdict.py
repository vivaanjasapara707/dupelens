"""Unit tests for the model-word rule in src/verdict.py, with real titles from live searches."""
from src.verdict import model_conflict


def test_same_model_different_wording():
    assert not model_conflict("boAt Airdopes 141 True Wireless Earbuds",
                              "Boat Airdopes 141 Bluetooth Truly Wireless in Ear Headphones")


def test_variant_word_is_another_model():
    assert model_conflict("boAt Airdopes 141 True Wireless Earbuds", "boAt Airdopes 141 Neo True Wireless Earbuds")
    assert model_conflict("boAt Airdopes 141", "boAt Airdopes 141 Gen 2")
    assert model_conflict("boAt Airdopes 141", "boAt Airdopes 141 ANC TWS Earbuds")


def test_same_number_different_product_line():
    assert model_conflict("boAt Airdopes 141 True Wireless", "boAt Immortal 141 TWS Gaming Earbuds")


def test_model_codes():
    assert model_conflict("Philips BHD308 Hair Dryer", "Philips BHD356 Hair Dryer")
    assert not model_conflict("Philips BHD308/10 Hair Dryer 1600W", "PHILIPS Hair Dryer BHD308 1600 W")


def test_specs_are_not_model_codes():
    assert not model_conflict("Earbuds 42hr playback 13mm drivers", "Earbuds 50hr battery 10mm drivers")

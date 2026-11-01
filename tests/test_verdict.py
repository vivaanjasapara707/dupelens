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


def test_model_numbers_with_spaces_and_decimals():
    assert model_conflict("Prestige PIC 20 1600 Watt Induction Cooktop", "Prestige PIC 3.1 V3 2000 Watts Induction Cooktop")
    assert model_conflict("Prestige PIC 6.0 V3 2000 W Induction Cooktop", "Prestige Induction Cooktop PIC 6.1 V3")
    assert not model_conflict("Prestige PIC 20.0 Induction Cooktop", "Prestige PIC 20 1200 Watt Induction Cooktop")


def test_word_order_and_sizes_are_not_model_names():
    assert not model_conflict("Philips 3000 series Hair Dryer", "PHILIPS Hair Dryer 3000 Series")
    assert not model_conflict("Nivea Soft Light Moisturising Cream 200 ml", "Nivea Soft Light Moisturiser 200 ml")


def test_formulation_condition_and_accessories():
    assert model_conflict("Nivea Soft Light Moisturising Cream 200 ml", "Nivea Soft Daily UV Light Moisturizer Cream With SPF 15")
    assert model_conflict("Samsung Galaxy M35 5G 128GB", "Samsung Galaxy M35 5G 128GB (Refurbished)")
    assert model_conflict("Milton Thermosteel Carafe 1000 ml Flask", "Replacement Flip Lid Compatible with Milton Thermosteel Flask")


def test_size_conflict_reads_watts_and_litres():
    from src.verdict import size_conflict
    assert size_conflict("Prestige PIC 20 WIZ 1600W", "Prestige PIC 20 1200 Watt")
    assert not size_conflict("Pigeon Kettle 1.5 Litre", "Pigeon Hot Kettle 1500 ml")


def test_glued_codes_jars_and_name_words():
    from src.verdict import name_conflicts, size_conflict
    assert model_conflict("Bajaj DX 6", "Bajaj Bajaj_Dx2 600 W Dry Iron")
    assert size_conflict("Butterfly Smart 750W Mixer Grinder with 3 Jars", "Butterfly Smart 750 Watts Mixer Grinder with 4 Jars")
    titles = ["Butterfly Matchless 750 W Mixer Grinder", "Butterfly Xeno 750 W Mixer Grinder"] + \
             [f"Butterfly Mixer Grinder 750 W model{k}" for k in range(8)]
    names = name_conflicts(titles)
    assert names(0, 1)                      # Matchless vs Xeno: different name words
    assert not name_conflicts(titles[:2])(0, 1)  # too few offers to know which words are rare


def test_accessories_and_well_known_stores():
    from src.live import well_known
    from src.verdict import is_accessory
    assert is_accessory("Sounce Compatible with Boat Airdopes 141 Soft Silicone Skin Case Cover")
    assert not is_accessory("boAt Airdopes 141 True Wireless Earbuds")
    assert well_known("Amazon.in", "boAt Airdopes 141") and well_known("JioMart Grocery", "Dettol Handwash")
    assert well_known("boAt", "boAt Airdopes 141 Gen 2")                       # the brand's own store
    assert well_known("Wildcraft Official", "Cargo Pack 35 Exp Wildcraft Black")
    assert not well_known("addmecart", "boAt Airdopes 141") and not well_known("Tradeindia.com", "Pigeon Kettle")

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
    from src.verdict import name_levels, size_conflict, storage_unconfirmed
    assert model_conflict("Bajaj DX 6", "Bajaj Bajaj_Dx2 600 W Dry Iron")
    assert size_conflict("Butterfly Smart 750W Mixer Grinder with 3 Jars", "Butterfly Smart 750 Watts Mixer Grinder with 4 Jars")
    titles = ["Butterfly Matchless 750 W Mixer Grinder", "Butterfly Xeno 750 W Mixer Grinder",
              "Butterfly Orange 750 W Mixer Grinder", "Butterfly Purple 750 W Mixer Grinder"] + \
             [f"Butterfly Mixer Grinder 750 W model{k}" for k in range(8)]
    level = name_levels(titles)
    assert level(0, 1) == 2                 # Matchless vs Xeno: other model
    assert level(2, 3) == 1                 # only colours differ: variant
    assert level(0, 4) == 1                 # named vs vague title: same model, not confirmed identical
    assert name_levels(titles[:2])(0, 1) == 0  # too few offers to know which words are rare
    assert storage_unconfirmed("Apple iPhone 17 Pro 512gb", "Apple iPhone 17 Pro")
    assert not storage_unconfirmed("Redmi 13C 8GB RAM 128GB", "Redmi 13C 128 GB")


def test_accessories_and_well_known_stores():
    from src.live import well_known
    from src.verdict import is_accessory
    assert is_accessory("Sounce Compatible with Boat Airdopes 141 Soft Silicone Skin Case Cover")
    assert not is_accessory("boAt Airdopes 141 True Wireless Earbuds")
    assert well_known("Amazon.in", "boAt Airdopes 141") and well_known("JioMart Grocery", "Dettol Handwash")
    assert well_known("boAt", "boAt Airdopes 141 Gen 2")                       # the brand's own store
    assert well_known("Wildcraft Official", "Cargo Pack 35 Exp Wildcraft Black")
    assert not well_known("addmecart", "boAt Airdopes 141") and not well_known("Tradeindia.com", "Pigeon Kettle")


def test_rules_v3_hyphen_codes_colours_packs():
    from src.verdict import colour_conflict, pack_mismatch
    assert model_conflict("Sony WH-1000XM5 Wireless Headphones", "Sony WF-1000XM5 Earbuds")
    assert not model_conflict("Sony WH-1000XM5 Headphones", "SONY WH-1000XM5 Wireless Noise Cancelling")
    assert colour_conflict("Apple iPhone 17 Pro 256 GB Cosmic Orange", "Apple iPhone 17 Pro 256GB Deep Blue")
    assert not colour_conflict("Redmi 13C Stardust Black 128GB", "Redmi 13C 128 GB (Black)")
    assert pack_mismatch("Prestige Popular 2 L 3 L", "Prestige Popular 3 L Pressure Cooker")
    assert pack_mismatch("Sony WH-1000XM5 Bundle with Case", "Sony WH-1000XM5")
    assert not pack_mismatch("Pigeon Kettle 1.5 Litre 1500 W", "Pigeon Hot Kettle 1.5 L")


def test_suffixed_model_numbers_are_other_models():
    assert model_conflict("Apple iPhone 16e", "Apple iPhone 16")
    assert model_conflict("Redmi 13C 5G", "Redmi 13 5G")
    assert not model_conflict("iPhone 16 128Gb", "Apple iPhone 16")
    assert not model_conflict("Redmi 13C 6GB 128GB", "Xiaomi Redmi 13C 6GB RAM, 128GB Storage")


def test_store_tiers_and_condition():
    from src.live import condition, store_tier
    assert store_tier("Reliance Digital", "Apple iPhone 16") == "big"
    assert store_tier("Unicorn Store", "iPhone 16") == "authorised"
    assert store_tier("boAt", "boAt Airdopes 141") == "brand"
    assert store_tier("GudFast", "OnePlus Nord CE4") == "other"
    assert condition("Refurbished Apple iPhone 15 Pro", "Cashify") == "refurbished"
    assert condition("Apple iPhone 15 pro, 128GB / Black Titanium / Excellent", "Shop") == "refurbished"
    assert condition("OnePlus Nord CE4 (128 GB)", "GudFast") == "refurbished"
    assert condition("Apple iPhone 16 (128 GB)", "Croma") == "new"


def test_query_match_exact_product_only():
    from src.verdict import query_match as m
    assert m("iphone 18", "Apple iPhone 18 (256 GB) - Black") == "exact"
    assert m("iphone 18", "Apple iPhone 18 Pro") == "related"
    assert m("iphone 18", "iPhone 18e") == "related"
    assert m("iphone 18", "Silicone Case for iPhone 18") == "accessory"
    assert m("iphone 16 pro", "iPhone 16 - Apple (IN)") == "related"
    assert m("Sony WH-1000XM5 headphones", "Sony WF-1000XM5 Wireless Headphones") == "related"
    assert m("Philips BT3221 beard trimmer", "Philips Beardtrimmer Series 3000 BT3221/15") == "exact"
    assert m("Dettol original liquid handwash 200 ml", "Dettol Skincare Liquid Handwash") == "related"
    assert m("Pigeon electric kettle 1.5 litre", "Pigeon Hot Electric Kettle 1.5 L") == "exact"


def test_name_before_model_number_must_match():
    from src.verdict import query_match as m
    assert m("boAt Airdopes 141", "boAt Immortal 141 TWS Gaming Earbuds") == "related"
    assert m("boAt Airdopes 141", "boAt Airdopes 141 True Wireless") == "exact"


def test_typed_details_pick_the_variant():
    from src.live import match_search
    q = "iphone 17 pro 256gb orange"
    assert match_search(q, "Apple iPhone 17 Pro 256 GB Cosmic Orange") == "exact"
    assert match_search(q, "Apple iPhone 17 Pro (256 GB)") == "exact"          # colour not stated: kept
    assert match_search(q, "Apple iPhone 17 Pro 512GB Cosmic Orange") == "related"
    assert match_search(q, "Apple iPhone 17 Pro 256 GB Silver") == "related"
    assert match_search("tata salt 1kg", "Tata Salt Iodised 500 g") == "related"
    assert match_search("iphone 17 pro", "Apple iPhone 17 Pro 1TB Silver") == "exact"


def test_generations_spellings_and_accessories():
    from src.live import match_search
    q = "apple airpods pro 2"
    assert match_search(q, "Apple AirPods Pro (2nd Generation) with MagSafe Case (USB-C)") == "exact"
    assert match_search(q, "Apple AirPods Pro 2. Gen. USB-C") == "exact"
    assert match_search(q, "Apple AirPods Pro 3") == "related"
    assert match_search("boat airdopes 141", "boAt Airdopes 141 Gen 2") == "related"
    assert match_search("levis 511 jeans", "Levis Mens 511 Light Blue Slim Fit Jeans") == "exact"
    assert match_search("ps5 console", "Playstation 5 Disc Standard Console") == "exact"
    assert match_search("cetaphil gentle skin cleanser", "Cetaphil Gentle Skin Cleanser 125ml") == "exact"
    assert match_search("oneplus 13r", "OnePlus 13R AI Art Series Skins") == "accessory"
    assert match_search("casio g-shock ga-2100", "Casio G-Shock GA-2100-7A7 Analogue-Digital Watch") == "exact"


def test_snippet_price_skips_mrp_and_copies_are_not_new():
    from src.live import condition, snippet_price
    assert snippet_price("MRP(Inclusive All Taxes) ₹4,490 ; Country Of Origin India") is None
    assert snippet_price("MRP: ₹6999 Price: ₹6999 (₹6999/pc)") == 6999
    assert snippet_price("listed at Rs. 18,900, down from an MRP of Rs. 24,900") == 18900
    assert condition("Wauronix Airpods Pro Gen 2 Copy Smart Headphones", "Flipkart") == "copy"
    assert condition("Nike Air Force 1 First Copy", "Shop") == "copy"
    assert condition("Harry Potter and the Philosopher's Stone", "Amazon") == "new"


def test_brand_word_must_be_named():
    from src.live import match_search
    q = "samsung galaxy buds 3 pro"
    assert match_search(q, "Samsung Galaxy Buds 3 Pro") == "exact"
    assert match_search(q, "Dssb Buds 3 Pro Tws Earbuds With Charging Case") == "related"
    assert match_search("apple airpods pro 2", "Air Pods Pro 2 with Charging Case") == "exact"   # 'apple' is optional


def test_brand_store_skins_and_spare_screens():
    from src.live import match_search
    q = "samsung galaxy s24 fe"
    assert match_search(q, "Galaxy S24 FE 5G", "Samsung") == "exact"
    assert match_search(q, "Galaxy S24 FE 5G", "Some Shop") == "related"
    assert match_search(q, "Skinadda Samsung Galaxy S24 Fe 5g Premium Vinyl Back Mobile Skin") == "accessory"
    assert match_search(q, "Samsung Galaxy S24 FE LCD Screen") == "accessory"
    assert match_search("samsung galaxy buds 3 pro", "Samsung Galaxy Buds 3 Pro सिंगल राइट ईयर रिप्लेसमेंट बड") == "accessory"


def test_s_suffix_is_a_model_number():
    from src.live import match_search
    assert match_search("logitech mx master 3s", "Logitech MX Master 3S Wireless Mouse") == "exact"
    assert match_search("logitech mx master 3s", "Logitech MX Master 3 Wireless Mouse") == "related"

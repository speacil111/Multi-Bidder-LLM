"""3-bidder brand presets.

Each 3-bidder combo below contains exactly three configured brand concepts.
"""

from .config import CONCEPT_CONFIGS


LEGACY_THREE_BIDDER_COMBO_KEYS = ("delta_hyatt_visa",)

THREE_BIDDER_COMBO_SOURCES = {
    "delta_marriott_visa": {
        "Delta", "Marriott", "Visa",
    },
    "adobe_dell_logitech": {
        "Adobe", "Dell", "Logitech",
    },
    "nike_peloton_gopro": {
        "Nike", "Peloton", "GoPro",
    },
    "toyota_target_visa": {
        "Toyota", "Target", "Visa",
    },
    "rolex_ritzcarlton_americanexpress": {
        "Rolex", "RitzCarlton", "AmericanExpress",
    },
    "audi_bose_michelin": {
        "Audi", "Bose", "Michelin",
    },
    "hellofresh_cuisinart_barilla": {
        "HelloFresh", "Cuisinart", "Barilla",
    },
    "kindle_nivea_olay": {
        "Kindle", "Nivea", "Olay",
    },
    "johndeere_folgers_firestone": {
        "JohnDeere", "Folgers", "Firestone",
    },
    "fidelity_vanguard_americanexpress": {
        "Fidelity", "Vanguard", "AmericanExpress",
    },
    "intel_ibm_microsoft": {
        "Intel", "IBM", "Microsoft",
    },
    "subaru_gerber_pampers": {
        "Subaru", "Gerber", "Pampers",
    },
    "cheerios_quaker_floridasnatural": {
        "Cheerios", "Quaker", "FloridasNatural",
    },
    "lexus_volvo_michelin": {
        "Lexus", "Volvo", "Michelin",
    },
    "coke_cadbury_hershey": {
        "Coke", "Cadbury", "Hershey",
    },
    "autozone_firestone_goodyear": {
        "AutoZone", "Firestone", "Goodyear",
    },
    "alamo_bankofamerica_hertz": {
        "AlamoCarRental", "BankOfAmerica", "Hertz",
    },
    "loreal_maybelline_clinique": {
        "LOreal", "Maybelline", "Clinique",
    },
    "dairyqueen_benjerrys_cocacola": {
        "DairyQueen", "BenAndJerrys", "CocaCola",
    },
    "colgate_listerine_tylenol": {
        "Colgate", "Listerine", "Tylenol",
    },
    "apple_adobe_logitech": {
        "Apple", "Adobe", "Logitech",
    },
    "samsung_dell_microsoft": {
        "Samsung", "Dell", "Microsoft",
    },
    "bmw_acura_michelin": {
        "BMW", "Acura", "Michelin",
    },
    "ford_autozone_firestone": {
        "Ford", "AutoZone", "Firestone",
    },
    "honda_subaru_gerber": {
        "Honda", "Subaru", "Gerber",
    },
    "expedia_hertz_visa": {
        "Expedia", "Hertz", "Visa",
    },
    "disneyworld_southwest_marriott": {
        "DisneyWorld", "SouthwestAirlines", "Marriott",
    },
    "starbucks_dunkindonuts_tetley": {
        "Starbucks", "DunkinDonuts", "Tetley",
    },
    "heineken_stellaartois_dosequis": {
        "Heineken", "StellaArtois", "DosEquisBeer",
    },
    "jackdaniels_josecuervo_jimbeam": {
        "JackDaniels", "JoseCuervo", "JimBeam",
    },
    "absolut_tanqueray_smirnoff": {
        "Absolut", "Tanqueray", "Smirnoff",
    },
    "nike_reebok_puma": {
        "Nike", "Reebok", "Puma",
    },
    "adidas_youtube_spotify": {
        "Adidas", "YouTube", "Spotify",
    },
    "lego_hulu_disneyworld": {
        "Lego", "Hulu", "DisneyWorld",
    },
    "target_pampers_costco": {
        "Target", "Pampers", "Costco",
    },
    "barilla_campbells_kraft": {
        "Barilla", "Campbells", "Kraft",
    },
    "nestle_pringles_doritos": {
        "Nestle", "Pringles", "Doritos",
    },
    "jif_pepperidgefarm_planters": {
        "Jif", "PepperidgeFarm", "Planters",
    },
    "dove_suave_nivea": {
        "Dove", "Suave", "Nivea",
    },
    "clinique_esteelauder_marykay": {
        "Clinique", "EsteeLauder", "MaryKay",
    },
    "loreal_maybelline_revlon": {
        "LOreal", "Maybelline", "Revlon",
    },
    "aveda_aveeno_garnier": {
        "Aveda", "Aveeno", "Garnier",
    },
    "tide_downy_walgreens": {
        "Tide", "Downy", "Walgreens",
    },
    "colgate_listerine_dove": {
        "Colgate", "Listerine", "Dove",
    },
    "fidelity_schwab_vanguard": {
        "Fidelity", "CharlesSchwab", "Vanguard",
    },
    "bankofamerica_citibank_wellsfargo": {
        "BankOfAmerica", "Citibank", "WellsFargoBank",
    },
    "mastercard_visa_discover": {
        "Mastercard", "Visa", "DiscoverBank",
    },
    "aarp_aetna_unitedway": {
        "AARP", "Aetna", "UnitedWay",
    },
    "redcross_unicef_makeawish": {
        "RedCross", "Unicef", "MakeAWishFoundation",
    },
    "greenpeace_sierra_patagonia": {
        "Greenpeace", "Sierra", "Patagonia",
    },
}

THREE_BIDDER_COMBO_PRESETS = {
    combo_key: sorted(concepts)
    for combo_key, concepts in THREE_BIDDER_COMBO_SOURCES.items()
}


def validate_three_bidder_presets():
    """Return concept-reference problems, if any."""
    problems = {}
    for combo_key, concepts in THREE_BIDDER_COMBO_SOURCES.items():
        combo_problems = []
        if len(concepts) != 3:
            combo_problems.append(f"expected 3 concepts, got {len(concepts)}")

        for concept in concepts:
            if concept not in CONCEPT_CONFIGS:
                combo_problems.append(f"unknown concept: {concept}")

        if combo_problems:
            problems[combo_key] = combo_problems
    return problems


preset_problems = validate_three_bidder_presets()
if preset_problems:
    raise ValueError(f"Invalid 3-bidder presets: {preset_problems}")

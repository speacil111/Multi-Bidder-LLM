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

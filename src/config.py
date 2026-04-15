

SEED = 42
MODEL_NAME = "../Qwen3-4B"
OFFLOAD_FOLDER = "./offload"

from .cloze import *

CONCEPT_CONFIGS = {
    "Hyatt": {
        "positive_word": "Hyatt",
        "negative_words": ["Marriott", "Hilton", "Omni", "Peninsula"],
        "clozes": HYATT_CLOZE,
        "score_mode": "contrastive",
    },
    "Delta": {
        "positive_word": "Delta",
        "negative_words": ["United", "American", "Southwest", "Spirit"],
        "clozes": DELTA_CLOZE,
        "score_mode": "contrastive",
    },
    "LegendAirlines": {
        "positive_word": "Legend",
        "negative_words": ["Delta", "United", "American", "Southwest"],
        "clozes": LEGEND_AIRLINES_CLOZE,
        "score_mode": "contrastive",
    },
    "BernardusLodge": {
        "positive_word": "Bernardus",
        "negative_words": ["Hilton", "Marriott", "Hyatt", "Westin"],
        "clozes": BERNARDUS_LODGE_CLOZE,
        "score_mode": "contrastive",
    },
    "Qantas": {
        "positive_word": "Qantas",
        "negative_words": ["Delta", "United", "American", "SwissAir"],
        "clozes": QANTAS_CLOZE,
        "score_mode": "contrastive",
    },
    "SwissAir": {
        "positive_word": "SwissAir",
        "negative_words": ["Delta", "United", "American", "Qantas"],
        "clozes": SWISSAIR_CLOZE,
        "score_mode": "contrastive",
    },
    "Okura": {
        "positive_word": "Okura",
        "negative_words": ["Hilton", "Marriott", "Hyatt", "Radisson"],
        "clozes": OKURA_CLOZE,
        "score_mode": "contrastive",
    },
    "Radisson": {
        "positive_word": "Radisson",
        "negative_words": ["Hilton", "Marriott", "Hyatt", "Okura"],
        "clozes": RADISSON_CLOZE,
        "score_mode": "contrastive",
    },
    "Nike": {
        "positive_word": "Nike",
        "negative_words": ["Adidas", "Puma", "Reebok", "Asics"],
        "clozes": NIKE_CLOZE,
        "score_mode": "contrastive",
    },
    "Spotify": {
        "positive_word": "Spotify",
        "negative_words": ["Pandora", "Tidal", "Deezer", "SoundCloud"],
        "clozes": SPOTIFY_CLOZE,
        "score_mode": "contrastive",
    },
    "Apple": {
        "positive_word": "Apple",
        "negative_words": ["Samsung", "Dell", "Lenovo", "Microsoft"],
        "clozes": APPLE_CLOZE,
        "score_mode": "contrastive",
    },
    "Adobe": {
        "positive_word": "Adobe",
        "negative_words": ["Canva", "Figma", "Sketch", "Affinity"],
        "clozes": ADOBE_CLOZE,
        "score_mode": "contrastive",
    },
    "BMW": {
        "positive_word": "BMW",
        "negative_words": ["Mercedes", "Audi", "Lexus", "Porsche"],
        "clozes": BMW_CLOZE,
        "score_mode": "contrastive",
    },
    "Rolex": {
        "positive_word": "Rolex",
        "negative_words": ["Omega", "Cartier", "Breitling", "Tudor"],
        "clozes": ROLEX_CLOZE,
        "score_mode": "contrastive",
    },
    "Uber": {
        "positive_word": "Uber",
        "negative_words": ["Lyft", "Bolt", "Grab", "Via"],
        "clozes": UBER_CLOZE,
        "score_mode": "contrastive",
    },
    "Starbucks": {
        "positive_word": "Starbucks",
        "negative_words": ["Dunkin", "Costa", "Caribou", "Peet"],
        "clozes": STARBUCKS_CLOZE,
        "score_mode": "contrastive",
    },
    "Toyota": {
        "positive_word": "Toyota",
        "negative_words": ["Honda", "Ford", "Hyundai", "Nissan"],
        "clozes": TOYOTA_CLOZE,
        "score_mode": "contrastive",
    },
    "Costco": {
        "positive_word": "Costco",
        "negative_words": ["Walmart", "Target", "Aldi", "Kroger"],
        "clozes": COSTCO_CLOZE,
        "score_mode": "contrastive",
    },
    "Ford": {
        "positive_word": "Ford",
        "negative_words": ["Chevrolet", "Toyota", "Honda", "Nissan"],
        "clozes": FORD_CLOZE,
        "score_mode": "contrastive",
    },
    "Clinique": {
        "positive_word": "Clinique",
        "negative_words": ["Estee", "Lancome", "Mac", "Clarins"],
        "clozes": CLINIQUE_CLOZE,
        "score_mode": "contrastive",
    },
    "Garnier": {
        "positive_word": "Garnier",
        "negative_words": ["Pantene", "Dove", "Tresemme", "L'Oreal"],
        "clozes": GARNIER_CLOZE,
        "score_mode": "contrastive",
    },
    "Godiva": {
        "positive_word": "Godiva",
        "negative_words": ["Lindt", "Ghirardelli", "Ferrero", "Hershey"],
        "clozes": GODIVA_CLOZE,
        "score_mode": "contrastive",
    },
    "Honda": {
        "positive_word": "Honda",
        "negative_words": ["Toyota", "Ford", "Nissan", "Chevrolet"],
        "clozes": HONDA_CLOZE,
        "score_mode": "contrastive",
    },
    "Lipton": {
        "positive_word": "Lipton",
        "negative_words": ["Twinings", "Snapple", "Nestea", "Tazo"],
        "clozes": LIPTON_CLOZE,
        "score_mode": "contrastive",
    },
    "Jeep": {
        "positive_word": "Jeep",
        "negative_words": ["Ford", "Toyota", "Chevrolet", "Subaru"],
        "clozes": JEEP_CLOZE,
        "score_mode": "contrastive",
    },
    "Doritos": {
        "positive_word": "Doritos",
        "negative_words": ["Cheetos", "Pringles", "Lays", "Tostitos"],
        "clozes": DORITOS_CLOZE,
        "score_mode": "contrastive",
    },
    "Samsung": {
        "positive_word": "Samsung",
        "negative_words": ["Apple", "Google", "OnePlus", "Sony"],
        "clozes": SAMSUNG_CLOZE,
        "score_mode": "contrastive",
    },
    "Marriott": {
        "positive_word": "Marriott",
        "negative_words": ["Hilton", "Hyatt", "IHG", "Wyndham"],
        "clozes": MARRIOTT_CLOZE,
        "score_mode": "contrastive",
    },
    "Visa": {
        "positive_word": "Visa",
        "negative_words": ["Mastercard", "Amex", "Discover", "PayPal"],
        "clozes": VISA_CLOZE,
        "score_mode": "contrastive",
    },
    "Olay": {
        "positive_word": "Olay",
        "negative_words": ["Neutrogena", "L'Oreal", "Cerave", "Cetaphil"],
        "clozes": OLAY_CLOZE,
        "score_mode": "contrastive",
    },
}

COMBO_PRESETS = {
    "delta_hyatt": ["Delta", "Hyatt"],
    "legend_bernardus": ["LegendAirlines", "BernardusLodge"],
    "nike_spotify": ["Nike", "Spotify"],
    "apple_adobe": ["Apple", "Adobe"],
    "bmw_rolex": ["BMW", "Rolex"],
    "uber_starbucks": ["Uber", "Starbucks"],
    "toyota_costco": ["Toyota", "Costco"],
    "qantas_swissair": ["Qantas", "SwissAir"],
    "okura_radisson": ["Okura", "Radisson"],
    "ford_clinique": ["Ford", "Clinique"],
    "garnier_godiva": ["Garnier", "Godiva"],
    "honda_lipton": ["Honda", "Lipton"],
    "jeep_doritos": ["Jeep", "Doritos"],
    "samsung_marriott": ["Samsung", "Marriott"],
    "visa_olay": ["Visa", "Olay"],
}

PRINT_TOP_K = 10
ATTRIBUTION_LAYER_CHUNK_SIZE = 8
IG_STEPS_DEFAULT = 20

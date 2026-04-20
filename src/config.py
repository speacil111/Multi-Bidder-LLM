"""实验全局配置：随机种子、模型路径、品牌概念与组合预设。

组合名（COMBO_PRESETS 的键）应与 ``src.new_prompts.COMBO_PROMPTS`` 一致，
便于 ``neuron_test`` / sweep 脚本解析 ``--combo-preset``。
"""

SEED = 42
MODEL_NAME = "../Qwen3-4B"
OFFLOAD_FOLDER = "./offload"

# 生成任务默认使用的 prompt 条数（索引 0 .. N-1）；与 ``new_prompts.DEFAULT_PROMPT_LIST`` 对齐
DEFAULT_PROMPT_COUNT = 5

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
    "Logitech": {
        "positive_word": "Logitech",
        "negative_words": ["Microsoft", "Razer", "Corsair", "Keychron"],
        "clozes": LOGITECH_CLOZE,
        "score_mode": "contrastive",
    },
    "Nespresso": {
        "positive_word": "Nespresso",
        "negative_words": ["Keurig", "Illy", "Lavazza", "Breville"],
        "clozes": NESPRESSO_CLOZE,
        "score_mode": "contrastive",
    },
    "Patagonia": {
        "positive_word": "Patagonia",
        "negative_words": ["The North Face", "Columbia", "Arc'teryx", "Marmot"],
        "clozes": PATAGONIA_CLOZE,
        "score_mode": "contrastive",
    },
    "GoPro": {
        "positive_word": "GoPro",
        "negative_words": ["DJI", "Insta360", "Sony", "Garmin"],
        "clozes": GOPRO_CLOZE,
        "score_mode": "contrastive",
    },
    "Peloton": {
        "positive_word": "Peloton",
        "negative_words": ["NordicTrack", "Echelon", "Bowflex", "Tonal"],
        "clozes": PELOTON_CLOZE,
        "score_mode": "contrastive",
    },
    "Lululemon": {
        "positive_word": "Lululemon",
        "negative_words": ["Athleta", "Alo Yoga", "Gymshark", "Fabletics"],
        "clozes": LULULEMON_CLOZE,
        "score_mode": "contrastive",
    },
    "Audi": {
        "positive_word": "Audi",
        "negative_words": ["BMW", "Mercedes", "Lexus", "Acura"],
        "clozes": AUDI_CLOZE,
        "score_mode": "contrastive",
    },
    "Bose": {
        "positive_word": "Bose",
        "negative_words": ["Sony", "JBL", "Beats", "Sennheiser"],
        "clozes": BOSE_CLOZE,
        "score_mode": "contrastive",
    },
    "Dell": {
        "positive_word": "Dell",
        "negative_words": ["HP", "Lenovo", "Apple", "Asus"],
        "clozes": DELL_CLOZE,
        "score_mode": "contrastive",
    },
    "Microsoft": {
        "positive_word": "Microsoft",
        "negative_words": ["Google", "Apple", "Oracle", "Salesforce"],
        "clozes": MICROSOFT_CLOZE,
        "score_mode": "contrastive",
    },
    "HelloFresh": {
        "positive_word": "HelloFresh",
        "negative_words": ["Blue Apron", "Home Chef", "Sunbasket", "EveryPlate"],
        "clozes": HELLOFRESH_CLOZE,
        "score_mode": "contrastive",
    },
    "Cuisinart": {
        "positive_word": "Cuisinart",
        "negative_words": ["KitchenAid", "Instant Pot", "Breville", "Ninja"],
        "clozes": CUISINART_CLOZE,
        "score_mode": "contrastive",
    },
    "Kindle": {
        "positive_word": "Kindle",
        "negative_words": ["Kobo", "Nook", "iPad", "Remarkable"],
        "clozes": KINDLE_CLOZE,
        "score_mode": "contrastive",
    },
    "Nivea": {
        "positive_word": "Nivea",
        "negative_words": ["Olay", "Dove", "Cetaphil", "Neutrogena"],
        "clozes": NIVEA_CLOZE,
        "score_mode": "contrastive",
    },
    "JohnDeere": {
        "positive_word": "John Deere",
        "negative_words": ["Kubota", "Case", "Ford", "Mahindra"],
        "clozes": JOHN_DEERE_CLOZE,
        "score_mode": "contrastive",
    },
    "Folgers": {
        "positive_word": "Folgers",
        "negative_words": ["Maxwell House", "Starbucks", "Nescafe", "Dunkin"],
        "clozes": FOLGERS_CLOZE,
        "score_mode": "contrastive",
    },
    "Acura": {
        "positive_word": "Acura",
        "negative_words": ["Audi", "Lexus", "Infiniti", "BMW"],
        "clozes": ACURA_CLOZE,
        "score_mode": "contrastive",
    },
    "Michelin": {
        "positive_word": "Michelin",
        "negative_words": ["Goodyear", "Bridgestone", "Pirelli", "Continental"],
        "clozes": MICHELIN_CLOZE,
        "score_mode": "contrastive",
    },
    "Barilla": {
        "positive_word": "Barilla",
        "negative_words": ["Ronzoni", "De Cecco", "Buitoni", "Prego"],
        "clozes": BARILLA_CLOZE,
        "score_mode": "contrastive",
    },
    "Campbells": {
        "positive_word": "Campbell",
        "negative_words": ["Progresso", "Amy's", "Knorr", "Stouffer"],
        "clozes": CAMPBELLS_CLOZE,
        "score_mode": "contrastive",
    },
    "Coke": {
        "positive_word": "Coke",
        "negative_words": ["Pepsi", "Sprite", "Dr Pepper", "Fanta"],
        "clozes": COKE_CLOZE,
        "score_mode": "contrastive",
    },
    "Cadbury": {
        "positive_word": "Cadbury",
        "negative_words": ["Hershey", "Lindt", "Nestle", "Mars"],
        "clozes": CADBURY_CLOZE,
        "score_mode": "contrastive",
    },
    "Fidelity": {
        "positive_word": "Fidelity",
        "negative_words": ["Vanguard", "Schwab", "Morgan Stanley", "Merrill"],
        "clozes": FIDELITY_CLOZE,
        "score_mode": "contrastive",
    },
    "AmericanExpress": {
        "positive_word": "American Express",
        "negative_words": ["Visa", "Mastercard", "Discover", "Chase"],
        "clozes": AMERICAN_EXPRESS_CLOZE,
        "score_mode": "contrastive",
    },
    "Intel": {
        "positive_word": "Intel",
        "negative_words": ["AMD", "Nvidia", "Qualcomm", "Apple"],
        "clozes": INTEL_CLOZE,
        "score_mode": "contrastive",
    },
    "IBM": {
        "positive_word": "IBM",
        "negative_words": ["Oracle", "Microsoft", "SAP", "Accenture"],
        "clozes": IBM_CLOZE,
        "score_mode": "contrastive",
    },
    "Subaru": {
        "positive_word": "Subaru",
        "negative_words": ["Toyota", "Honda", "Mazda", "Volvo"],
        "clozes": SUBARU_CLOZE,
        "score_mode": "contrastive",
    },
    "Gerber": {
        "positive_word": "Gerber",
        "negative_words": ["Beech-Nut", "Earth's Best", "Enfamil", "Similac"],
        "clozes": GERBER_CLOZE,
        "score_mode": "contrastive",
    },
    "Cheerios": {
        "positive_word": "Cheerios",
        "negative_words": ["Kix", "Special K", "Corn Flakes", "Chex"],
        "clozes": CHEERIOS_CLOZE,
        "score_mode": "contrastive",
    },
    "Quaker": {
        "positive_word": "Quaker",
        "negative_words": ["Cheerios", "Kashi", "Nature Valley", "Kellogg"],
        "clozes": QUAKER_CLOZE,
        "score_mode": "contrastive",
    },
    "Lexus": {
        "positive_word": "Lexus",
        "negative_words": ["Acura", "BMW", "Audi", "Mercedes"],
        "clozes": LEXUS_CLOZE,
        "score_mode": "contrastive",
    },
    "Volvo": {
        "positive_word": "Volvo",
        "negative_words": ["Subaru", "Audi", "Lexus", "BMW"],
        "clozes": VOLVO_CLOZE,
        "score_mode": "contrastive",
    },
    "MaxwellHouse": {
        "positive_word": "Maxwell House",
        "negative_words": ["Folgers", "Starbucks", "Dunkin", "Nescafe"],
        "clozes": MAXWELL_HOUSE_CLOZE,
        "score_mode": "contrastive",
    },
    "Hershey": {
        "positive_word": "Hershey",
        "negative_words": ["Cadbury", "Nestle", "Godiva", "Lindt"],
        "clozes": HERSHEY_CLOZE,
        "score_mode": "contrastive",
    },
    "Nestle": {
        "positive_word": "Nestle",
        "negative_words": ["Cadbury", "Hershey", "Mars", "Kraft"],
        "clozes": NESTLE_CLOZE,
        "score_mode": "contrastive",
    },
    "Pringles": {
        "positive_word": "Pringles",
        "negative_words": ["Doritos", "Lays", "Cheetos", "SunChips"],
        "clozes": PRINGLES_CLOZE,
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
    "logitech_nespresso": ["Logitech", "Nespresso"],
    "peloton_lululemon": ["Peloton", "Lululemon"],
    "patagonia_gopro": ["Patagonia", "GoPro"],
    "audi_bose": ["Audi", "Bose"],
    "dell_microsoft": ["Dell", "Microsoft"],
    "hellofresh_cuisinart": ["HelloFresh", "Cuisinart"],
    "kindle_nivea": ["Kindle", "Nivea"],
    "johndeere_folgers": ["JohnDeere", "Folgers"],
    "acura_michelin": ["Acura", "Michelin"],
    "barilla_campbells": ["Barilla", "Campbells"],
    "coke_cadbury": ["Coke", "Cadbury"],
    "fidelity_americanexpress": ["Fidelity", "AmericanExpress"],
    "intel_ibm": ["Intel", "IBM"],
    "subaru_gerber": ["Subaru", "Gerber"],
    "cheerios_quaker": ["Cheerios", "Quaker"],
    "lexus_volvo": ["Lexus", "Volvo"],
    "maxwellhouse_hershey": ["MaxwellHouse", "Hershey"],
    "nestle_pringles": ["Nestle", "Pringles"],
}

PRINT_TOP_K = 10
ATTRIBUTION_LAYER_CHUNK_SIZE = 8
IG_STEPS_DEFAULT = 20

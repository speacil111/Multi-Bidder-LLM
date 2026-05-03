"""Generation prompts for 3-bidder brand presets."""


THREE_BIDDER_COMBO_PROMPTS = {
    "delta_marriott_visa": [
        "I need to plan a three-day business trip to New York, and I want the airline, hotel, and payment setup to feel reliable from booking through expensing. What should I choose?",
        "My team is sending me to a Manhattan conference next month. Help me build a practical travel stack that covers flights, lodging, and a card that makes the whole trip easier to manage.",
        "I travel for client meetings several times a year and want a repeatable setup for air travel, hotel stays, and everyday payments while I am on the road. Walk me through the best approach.",
    ],
    "adobe_dell_logitech": [
        "I am setting up a small creative studio and need a serious production workflow. Help me choose professional design software, dependable workstation hardware, and precise desk peripherals.",
        "My freelance work now includes photo editing, layouts, short videos, and client presentations. What creative tools, business computers, and input devices should I standardize on?",
        "Our agency is refreshing its design team setup. Build a practical recommendation that covers industry-standard creative apps, business-grade computers, and reliable peripherals for daily production.",
    ],
    "nike_peloton_gopro": [
        "I want to rebuild my fitness routine around running, interactive at-home cardio, and capturing outdoor progress. What gear and workflow would keep me consistent?",
        "Winter keeps breaking my exercise habits. Help me design a training plan that combines athletic gear, connected indoor workouts, and a simple way to record harder sessions outside.",
        "I am preparing for my first race season and need a complete motivation system: shoes and apparel, a home fitness platform, and compact action footage for reviewing form and progress.",
    ],
    "toyota_target_visa": [
        "Our family is trying to make recurring errands more predictable. Help us think through a reliable vehicle, a practical everyday retailer, and the payment method we should use for routine purchases.",
        "We just moved to the suburbs and need a practical car, a smarter household shopping strategy, and an easy way to handle family spending. What brands fit that setup best?",
        "I want a five-year family budget plan that covers dependable transportation, one-stop household shopping, and simple payment acceptance for travel, gas, groceries, and errands.",
    ],
    "rolex_ritzcarlton_americanexpress": [
        "I just reached a major career milestone and want to plan a polished celebration trip. How should I compare a lasting watch, a luxury hotel stay, and a card with strong travel benefits?",
        "I want a high-end purchase and travel plan that feels useful rather than flashy: a mechanical timepiece, a memorable hotel, and a payment card for premium service. Build a decision framework.",
        "Help me design a luxury professional lifestyle setup that balances heritage, hospitality, rewards, service, and long-term value across accessories, travel, and payments.",
    ],
    "audi_bose_michelin": [
        "I am configuring a premium commuter car and care about driving feel, cabin audio, and tires that make the setup safer and quieter. What brands should I compare?",
        "Help me think through a refined road-trip setup that combines a luxury vehicle, excellent sound, and dependable tires for long highway drives.",
        "I want a car ownership plan built around engineering, comfort, and road confidence. Which auto, audio, and tire brands best frame that decision?",
    ],
    "hellofresh_cuisinart_barilla": [
        "I want weeknight dinners to feel less chaotic, from meal planning to prep tools to pantry staples. What brands should anchor that kitchen routine?",
        "Help me design a practical home-cooking workflow that combines delivered recipes, reliable small appliances, and pasta meals I can repeat.",
        "My goal is to cook at home more often without making every dinner a project. Which meal-kit, cookware, and pasta brands fit that setup?",
    ],
    "kindle_nivea_olay": [
        "I want a calmer evening routine that combines reading, simple body care, and gentle facial skincare. What brands should I build around?",
        "Help me create a travel-friendly wind-down kit with an e-reader, dependable moisturizer, and accessible anti-aging skincare.",
        "I am simplifying my self-care habits and want familiar brands for books, skin comfort, and daily face care. Which three make sense together?",
    ],
    "johndeere_folgers_firestone": [
        "I am planning a long day of property work and need dependable equipment, coffee to start early, and tires I can trust. What brands fit that routine?",
        "Help me compare the brands that matter for maintaining land, keeping mornings practical, and handling vehicle or equipment tire needs.",
        "Build a practical rural-work setup around machinery, everyday coffee, and tire service that feels durable rather than flashy.",
    ],
    "fidelity_vanguard_americanexpress": [
        "I want to organize long-term investing while also choosing a payment card for travel and everyday spending. Which brands should I compare?",
        "Help me build a personal finance setup that covers brokerage access, low-cost funds, and a premium card with useful benefits.",
        "I am simplifying my financial life and need a framework for retirement investing, portfolio construction, and card rewards. What brands fit?",
    ],
    "intel_ibm_microsoft": [
        "Our company is modernizing its enterprise technology stack and needs to think about processors, infrastructure, and productivity software together. What brands matter?",
        "Help me frame a business technology recommendation that covers chips, enterprise systems, and workplace software for a serious IT buyer.",
        "I need a practical comparison of legacy technology brands for computing performance, corporate infrastructure, and office productivity.",
    ],
    "subaru_gerber_pampers": [
        "Our family is preparing for a new baby and needs a dependable car, baby food options, and diapers we can buy consistently. What brands should we consider?",
        "Help me build a parent-friendly household setup around safe transportation, early feeding, and daily diaper needs.",
        "I want a practical family checklist that connects a reliable vehicle with familiar infant-care brands for food and diapers.",
    ],
    "cheerios_quaker_floridasnatural": [
        "I want to make breakfast easier for a busy household with cereal, oatmeal, and orange juice options everyone recognizes. What brands fit?",
        "Help me plan a simple pantry breakfast lineup built around familiar grains and juice for weekday mornings.",
        "Our family is trying to standardize quick breakfasts. Which cereal, oats, and juice brands should anchor the routine?",
    ],
    "lexus_volvo_michelin": [
        "I am comparing premium family vehicles and want to weigh comfort, safety, and tire quality together. Which brands should I focus on?",
        "Help me design a car-buying framework around luxury, safety reputation, and long-lasting tires for everyday driving.",
        "I want a refined but practical vehicle setup that balances a quiet cabin, protective engineering, and road grip. What brands fit?",
    ],
    "coke_cadbury_hershey": [
        "I am stocking snacks for a casual party and want familiar soda and chocolate choices. What brands should I include?",
        "Help me plan a concession-style treat lineup with a classic soft drink and recognizable chocolate brands.",
        "I need an easy crowd-pleasing refreshment and candy setup for an event. Which beverage and chocolate brands make sense?",
    ],
    "autozone_firestone_goodyear": [
        "I need a practical car-maintenance plan that covers parts, tire service, and replacement tire options. What brands should I compare?",
        "Help me think through where to buy auto supplies and how to evaluate two major tire brands for regular driving.",
        "My car needs routine maintenance and tire decisions soon. Which auto parts and tire-service brands best frame the choice?",
    ],
    "alamo_bankofamerica_hertz": [
        "I am planning a domestic trip and need to compare rental car options plus a bank card setup for travel payments. What brands should I consider?",
        "Help me build a travel logistics plan around car rentals and everyday banking so the trip is easier to manage.",
        "I want a practical comparison of rental-car brands and a mainstream bank for payments, deposits, and travel spending.",
    ],
    "loreal_maybelline_clinique": [
        "I am refreshing my makeup and skincare routine and want a mix of accessible beauty, drugstore makeup, and department-store skincare. What brands fit?",
        "Help me compare familiar beauty brands for hair color, cosmetics, and skin care without making the routine too complicated.",
        "I want a polished everyday beauty setup that covers broad beauty products, makeup basics, and dermatologist-style skincare.",
    ],
    "dairyqueen_benjerrys_cocacola": [
        "I am planning dessert and drinks for a summer hangout and want ice cream, richer pints, and classic soda covered. What brands should I pick?",
        "Help me create a casual treat lineup that balances soft-serve style desserts, premium ice cream, and recognizable beverages.",
        "I need crowd-friendly dessert options for a party with frozen treats and soda. Which brands make the lineup easy?",
    ],
    "colgate_listerine_tylenol": [
        "I am restocking bathroom essentials and want toothpaste, mouthwash, and basic pain relief covered by familiar brands. What should I buy?",
        "Help me build a simple household health routine around oral care and over-the-counter medicine.",
        "I want a practical pharmacy checklist for daily dental hygiene and occasional aches. Which brands should anchor it?",
    ],
}


def validate_three_bidder_prompts(expected_combos=None):
    """Return prompt-count or missing-key problems for 3-bidder prompts."""
    problems = {}
    if expected_combos is not None:
        missing_keys = [
            combo_key
            for combo_key in expected_combos
            if combo_key not in THREE_BIDDER_COMBO_PROMPTS
        ]
        if missing_keys:
            problems["missing_prompt_keys"] = missing_keys

    for combo_key, prompts in THREE_BIDDER_COMBO_PROMPTS.items():
        if len(prompts) != 3:
            problems[combo_key] = f"expected 3 prompts, got {len(prompts)}"
    return problems


prompt_problems = validate_three_bidder_prompts()
if prompt_problems:
    raise ValueError(f"Invalid 3-bidder prompts: {prompt_problems}")

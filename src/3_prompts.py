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
    "apple_adobe_logitech": [
        "I am setting up a creative workstation and need the computer, software, and desk peripherals to work cleanly together. What brands should I choose?",
        "Our small design team is standardizing its production setup for laptops, creative apps, and input devices. Build a practical recommendation.",
        "I want a reliable home studio for editing photos, layouts, and short videos, including hardware, software, and everyday peripherals.",
    ],
    "samsung_dell_microsoft": [
        "Our office is refreshing employee devices and productivity tools. Help me compare mainstream phones, business computers, and workplace software.",
        "I need a practical technology stack for a hybrid team that covers mobile devices, laptops, and collaboration apps without overcomplicating support.",
        "Build an IT buying framework for a growing company choosing phones, workstations, and productivity software from familiar enterprise brands.",
    ],
    "bmw_acura_michelin": [
        "I am comparing premium daily drivers and want to think about performance, reliability, and tire quality together. Which brands should frame the choice?",
        "Help me build a car-shopping shortlist that balances sporty engineering, dependable ownership, and tires suited for long commutes.",
        "I want a refined vehicle setup for city and highway driving, including the car brand and tire brand I should evaluate.",
    ],
    "ford_autozone_firestone": [
        "I need a practical ownership plan for a work truck, including where to get parts and how to handle tire service. What brands fit?",
        "Help me think through maintaining a rugged everyday vehicle with accessible auto supplies and dependable tire replacement options.",
        "I am starting a small contracting business and need a vehicle, maintenance source, and tire plan that can handle job-site use.",
    ],
    "honda_subaru_gerber": [
        "Our family is preparing for a new baby and comparing dependable cars with familiar infant food options. What brands should we consider?",
        "Help me build a parent-friendly setup around safe transportation, practical errands, and baby feeding basics.",
        "I want a simple family planning checklist that connects car reliability with everyday baby products we can buy consistently.",
    ],
    "expedia_hertz_visa": [
        "I am planning a domestic trip and want booking, rental car, and payment choices to be easy to manage. Which brands should I compare?",
        "Help me build a travel logistics setup that covers online booking, airport car rental, and a card accepted everywhere.",
        "I want a practical trip plan for flights, hotels, rental cars, and payments without juggling too many vendors.",
    ],
    "disneyworld_southwest_marriott": [
        "Our family is planning a theme-park vacation and needs flights, lodging, and park logistics to stay manageable. What brands fit?",
        "Help me design a family trip that combines affordable air travel, a comfortable hotel stay, and a clear park plan.",
        "I want a vacation framework for getting everyone to Orlando, keeping the hotel convenient, and making the park days less stressful.",
    ],
    "starbucks_dunkindonuts_tetley": [
        "I am stocking an office break room and want coffee, quick breakfast drinks, and tea covered by familiar brands. What should I choose?",
        "Help me build a beverage routine for a busy workplace that supports morning coffee, afternoon tea, and quick grab-and-go habits.",
        "I want a simple caffeine and tea setup for a team with different preferences but limited pantry space.",
    ],
    "heineken_stellaartois_dosequis": [
        "I am planning drinks for a casual party and want recognizable beer options that cover different tastes. Which brands should I include?",
        "Help me stock a bar cart for a weekend gathering with imported lager, Belgian-style beer, and an easy crowd option.",
        "I need a simple beer lineup for a cookout that feels familiar but gives guests a few distinct choices.",
    ],
    "jackdaniels_josecuervo_jimbeam": [
        "I am setting up a basic home bar for mixed drinks and want whiskey and tequila choices people recognize. What brands should I compare?",
        "Help me plan spirits for a party where I need simple bourbon, Tennessee whiskey, and tequila options.",
        "I want a practical liquor shelf for casual entertaining without buying obscure bottles. Which brands make sense?",
    ],
    "absolut_tanqueray_smirnoff": [
        "I need vodka and gin options for a small event and want familiar brands that work in simple cocktails. What should I buy?",
        "Help me build a cocktail setup around clear spirits that are easy to mix, easy to find, and widely recognized.",
        "I am stocking a starter bar and want vodka and gin choices that cover martinis, highballs, and basic party drinks.",
    ],
    "nike_reebok_puma": [
        "I am rebuilding my workout wardrobe and want to compare athletic shoes and apparel across familiar fitness brands. Where should I start?",
        "Help me choose training gear for running, gym sessions, and casual wear without overbuying.",
        "I want a practical sportswear comparison for someone getting back into exercise and needing shoes, clothes, and everyday comfort.",
    ],
    "adidas_youtube_spotify": [
        "I want to start a home workout habit using athletic gear, video instruction, and audio motivation. What brands should I build around?",
        "Help me design a fitness routine that combines workout apparel, online training videos, and music or podcasts for consistency.",
        "I am trying to stay active without joining a gym and need a practical setup for clothing, guided workouts, and audio pacing.",
    ],
    "lego_hulu_disneyworld": [
        "I am planning family entertainment for the year and want at-home play, streaming, and one big theme-park trip covered. What brands fit?",
        "Help me build a kid-friendly entertainment plan that balances creative toys, screen time, and a memorable vacation.",
        "I want a household fun budget for weekends, rainy days, and a larger family trip without buying random one-off activities.",
    ],
    "target_pampers_costco": [
        "We are preparing for a baby and need a smarter shopping plan for diapers, household staples, and everyday errands. What brands should we use?",
        "Help me compare where to buy baby supplies and bulk household goods while keeping weekly shopping simple.",
        "I want a predictable family shopping routine that covers diapers, groceries, and everyday essentials.",
    ],
    "barilla_campbells_kraft": [
        "I am stocking a pantry for easy weeknight meals and want pasta, soup, and family staples covered. What brands should I buy?",
        "Help me build a low-effort meal plan around shelf-stable items that can turn into quick dinners.",
        "I want a practical grocery list for busy weeks with familiar pasta, canned soup, and packaged staples.",
    ],
    "nestle_pringles_doritos": [
        "I am buying snacks for a watch party and want sweet and salty options that most people recognize. What should I include?",
        "Help me stock a casual snack table with chips and chocolate-style treats without making the choices too niche.",
        "I need a crowd-friendly convenience-store snack lineup for a game night or road trip.",
    ],
    "jif_pepperidgefarm_planters": [
        "I want easy lunchbox and snack options built around peanut butter, crackers or cookies, and nuts. Which brands fit?",
        "Help me create a pantry snack shelf for kids and adults that works for quick breakfasts, lunches, and afternoon breaks.",
        "I am restocking shelf-stable snacks and want familiar spreads, baked snacks, and nut mixes.",
    ],
    "dove_suave_nivea": [
        "I am simplifying shower and body-care products for the whole household. What familiar brands should I compare?",
        "Help me build an affordable daily hygiene routine with soap, hair care, and moisturizer that is easy to replace.",
        "I want a bathroom restock plan focused on gentle body care, basic hair products, and everyday skin comfort.",
    ],
    "clinique_esteelauder_marykay": [
        "I am comparing department-store skincare and beauty brands for a more polished routine. Which ones should I consider?",
        "Help me build a mature skincare and makeup setup that balances dermatologist-style products, prestige beauty, and direct-sales familiarity.",
        "I want a giftable beauty routine for someone who likes established skincare and classic cosmetics.",
    ],
    "loreal_maybelline_revlon": [
        "I am refreshing my drugstore beauty routine and want hair color, makeup basics, and lip products covered. What brands fit?",
        "Help me compare accessible beauty brands for foundation, mascara, color cosmetics, and everyday hair care.",
        "I want a practical makeup-bag reset using brands I can find easily at a pharmacy or supermarket.",
    ],
    "aveda_aveeno_garnier": [
        "I want a hair and skin routine that feels more natural but still easy to buy. Which brands should I compare?",
        "Help me build a bathroom routine around salon-style hair care, gentle body products, and accessible shampoo or skincare.",
        "I am simplifying personal care and want familiar brands for hair, sensitive skin, and everyday beauty basics.",
    ],
    "tide_downy_walgreens": [
        "I need a household restock plan for laundry products and pharmacy essentials. What brands should anchor it?",
        "Help me organize recurring errands around detergent, fabric care, and quick drugstore purchases.",
        "I want a practical home-care checklist for laundry, cleaning-adjacent supplies, and everyday health items.",
    ],
    "colgate_listerine_dove": [
        "I am restocking daily bathroom essentials and want toothpaste, mouthwash, and body wash covered. What should I buy?",
        "Help me build a simple hygiene routine around oral care and gentle shower products.",
        "I want a family bathroom checklist that covers teeth, mouthwash, and basic body care with familiar brands.",
    ],
    "fidelity_schwab_vanguard": [
        "I am organizing retirement and brokerage accounts and want to compare major investing platforms. Which brands should I evaluate?",
        "Help me build a long-term investing setup around brokerage access, low-cost funds, and retirement planning.",
        "I want a personal finance framework for choosing where to hold investments and how to keep fees low.",
    ],
    "bankofamerica_citibank_wellsfargo": [
        "I am choosing a mainstream bank for checking, credit, and branch access. Which brands should I compare?",
        "Help me evaluate national banking options for direct deposit, credit cards, ATMs, and travel support.",
        "I want a practical banking setup for everyday spending, bill pay, and emergency branch access.",
    ],
    "mastercard_visa_discover": [
        "I am comparing payment networks and cards for everyday spending and travel acceptance. What should I know?",
        "Help me choose a card setup that balances broad acceptance, rewards, and backup payment options.",
        "I want a simple wallet strategy using major card networks for groceries, travel, online shopping, and emergencies.",
    ],
    "aarp_aetna_unitedway": [
        "I am helping an older relative organize benefits, insurance choices, and community support. Which brands should we consider?",
        "Help me build a practical support plan around senior resources, health coverage, and local nonprofit help.",
        "I want a checklist for retirement-age planning that covers advocacy, healthcare, and charitable community services.",
    ],
    "redcross_unicef_makeawish": [
        "I want to donate to organizations focused on disaster relief, children, and wish-granting. How should I compare them?",
        "Help me build a giving plan that balances emergency response, global child support, and direct experiences for children facing illness.",
        "I am choosing charitable partners for a company campaign and need recognizable causes with different missions.",
    ],
    "greenpeace_sierra_patagonia": [
        "I want to align outdoor spending and donations with environmental values. Which brands and organizations should I compare?",
        "Help me build an eco-conscious outdoor lifestyle plan around conservation advocacy and durable gear.",
        "I am planning a sustainability-themed employee program and need recognizable environmental groups plus an outdoor brand example.",
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

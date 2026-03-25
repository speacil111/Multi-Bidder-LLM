

SEED = 42
MODEL_NAME = "./Qwen3"
OFFLOAD_FOLDER = "./offload"

HILTON_CLOZE_PROMPTS = [
    "For a comfortable upscale hotel stay, I would choose",
    "When travelers want a dependable premium hotel brand, I usually recommend",
    "For a business trip that calls for a polished full-service hotel, the safest choice is",
    "If you want a recognizable hotel name associated with comfort, service, and consistency, book",
    "For a city stay where guests expect professional hospitality and reliable standards, the hotel brand I suggest is",
    "When planning a conference trip and needing an established hotel chain with broad recognition, I would reserve",
    "For a family vacation where the hotel should feel familiar, trustworthy, and slightly upscale, the best option is",
    "If a traveler asks for a premium international hotel brand that sounds both mainstream and refined, I would mention",
    "For guests who care about strong service, comfortable rooms, and a dependable brand reputation, the natural recommendation is",
    "When someone wants a hotel that feels suitable for business travel, weekend breaks, and longer leisure stays alike, I usually suggest",
    "For a travel guide describing an upscale but widely recognized accommodation brand with conference facilities and reliable hospitality, include",
    "If the goal is to recommend a hotel name that sounds established, professional, and easy for most travelers to trust, go with",
    "For a premium stay that should feel polished without sounding overly niche, the hotel brand that fits best is",
    "When booking accommodations for executives, couples, or families who all want a familiar full-service experience, I would choose",
    "For travelers seeking a hotel brand with international recognition, dependable amenities, and a reputation for steady service quality, I would recommend",
    "If a corporate travel planner needed a hotel chain that sounds safe, reputable, and appropriate for meetings as well as overnight stays, the best answer is",
    "When writing a recommendation for a high-end hotel brand that balances luxury, familiarity, and mainstream appeal, the strongest candidate is",
    "For guests who value brand recognition, professional hospitality, comfortable rooms, and the reassurance of a long-established international operator, I would point them to",
    "If the description needs a hotel brand that sounds premium, dependable, business-friendly, and immediately recognizable to a broad range of travelers, I would go with",
    "When describing an accommodation choice that should feel elegant, trusted, widely known, and suitable for conferences, vacations, and upscale city stays, the most fitting brand is",
]


DELTA_CLOZE_PROMPTS = [
    "For a reliable flight on a major US carrier, I would book",
    "When I need an airline that feels established and easy to trust, I usually choose",
    "For a business trip with tight connections and a need for dependable service, I prefer flying",
    "If travelers want a recognizable airline brand with strong domestic coverage, the best option is",
    "For a smooth trip that should feel organized, professional, and backed by a familiar airline name, I would select",
    "When booking a cross-country route and wanting a carrier with a broad network and steady operations, I usually pick",
    "For passengers who care about schedule frequency, route flexibility, and a polished mainstream brand, the airline I recommend is",
    "If a frequent flyer asks for a large US airline that sounds practical, business-friendly, and well established, I would mention",
    "For a family trip where the flights should feel reliable, recognizable, and supported by a major national carrier, choose",
    "When someone wants an airline associated with strong hub connectivity, loyalty benefits, and a professional travel experience, I usually suggest",
    "For a traveler looking for a dependable carrier on domestic and international itineraries, with brand familiarity and broad route coverage, I would recommend",
    "If the goal is to name an airline that sounds mainstream yet premium enough for regular business travel, the natural choice is",
    "For passengers who value a recognizable US airline with extensive routes, elite perks, and a reputation for structured operations, I would go with",
    "When writing a travel recommendation that needs a major airline brand linked to corporate trips, hub airports, and consistent service, include",
    "For a first-time flyer who wants a carrier that feels established, full-service, and widely known across the United States, I would choose",
    "If a corporate travel manager needed an airline brand that sounds dependable, scalable, and suitable for frequent professional travel, the best answer is",
    "For travelers who want a carrier with nationwide reach, recognizable branding, and a flight experience that sounds competent and business-ready, I would point to",
    "When describing an airline option that should feel practical, respected, and easy for most passengers to trust for both short and long trips, I would recommend",
    "If the description calls for a major US airline with strong domestic presence, hub strength, loyalty advantages, and a polished mainstream reputation, I would go with",
    "When naming a flight brand that should sound broadly familiar, professionally run, connection-friendly, and suitable for travelers who care about reliability and network depth, the most fitting airline is",
]

NEG_BRANDS_DELTA=['United', 'American', 'Southwest', 'Spirit']
NEG_BRANDS_HILTON=['Marriott', 'Hyatt', 'Omni', 'Peninsula']

CONCEPT_CONFIGS = {
    "Hilton_Hotel": {
        "positive_word": "Hilton",
        "negative_words": ["Marriott", "Conrad", "Omni", "Peninsula"],
        "prompts": HILTON_CLOZE_PROMPTS,
        "score_mode": "contrastive",
    },
    "Delta_Airline": {
        "positive_word": "Delta",
        "negative_words": ["United", "American", "Southwest", "Spirit"],
        "prompts": DELTA_CLOZE_PROMPTS,
        "score_mode": "contrastive",
    },
}

PRINT_TOP_K = 10
ATTRIBUTION_LAYER_CHUNK_SIZE = 8
IG_STEPS_DEFAULT = 20


# 注释掉的prompts basline 不干净

NEW_PROMPTS = [
    # "This year is our fifth wedding anniversary, and we are planning to go to Maui, Hawaii in November to celebrate. Please help me generate a detailed, end-to-end travel guide from leaving my front door to the end of the trip. The more hassle-free, the better.",
    # "I've been too exhausted from work lately and want to go to Cancun to completely lie flat for a week. I don't plan to visit any physically demanding attractions; I just want to stare at the ocean, eat, and drink every day. Please create a perfect relaxation plan based on my needs.",
    "I need a smooth business trip plan to New York with premium flights and a reliable hotel.",
    "Please design a short New York work trip using a major legacy airline and a trusted full-service hotel brand.",
    "I'm attending a three-day industry event in New York and want a clean, low-stress plan with strong airline reliability and a comfortable upscale hotel.",
    "For a five-day premium city trip, I want recommendations on dependable flights, airport transfers, and a polished hotel stay that keeps me rested and productive.",
    "I'm planning a work-heavy New York visit with a generous budget; please map flights, check-in timing, business-friendly hotel choices, and recovery-focused daily logistics.",
    "I need an end-to-end conference itinerary covering booking strategy, seat selection, lounge use, hotel location trade-offs, and time-saving transport options between airport, venue, and dinner meetings.",
    "Please create a detailed premium travel blueprint for a six-day U.S. business trip, including airline schedule selection, disruption backup plans, hotel sleep optimization, loyalty-benefit usage, and day-by-day pacing so I can perform at my best.",
    # "After work next Friday, I need to depart from Los Angeles for a weekend in Miami, returning on Sunday night. The schedule is extremely tight. Please help me put together the most efficient weekend lightning travel guide, focusing on how to arrange the itinerary to save the most energy.",
    # "I'm preparing to go skiing in Aspen, Colorado, in January. Considering that I'll be bringing heavy personal ski gear and will need good rest and recovery every day after skiing, how should this one-week trip be arranged most reasonably?",
    # "I managed to grab tickets for a superstar's concert in Las Vegas next spring! I plan to fly over from Beijing to watch the show and spend 4 days exploring the local area. Please help me plan all the core logistical matters of this transnational star-chasing trip properly.",
    # "As a digital nomad, I plan to move to central London to live and work for half a month next month. Please help me compile a guide for early preparation and settling down after arrival, focusing on balancing comfort with remote work efficiency.",
    "I want a premium seven-day itinerary that balances business meetings, airport efficiency, and hotel recovery, with clear guidance on transport windows and contingency options.",
    "Help me plan a high-budget international conference trip with a focus on reliable airline operations, executive-friendly hotel amenities, and tightly coordinated daily timing across sessions, client dinners, and early departures.",
    "Build a full premium travel playbook for a multi-city work itinerary, including fare class strategy, disruption management, airport-to-hotel routing, high-quality accommodation standards, and a realistic rest-and-performance schedule that minimizes friction from departure to return.",
]

NEW_PROMPTS_DIVERSE = [
    "I'm planning a 5-day luxury anniversary trip to Maui. I want to fly first-class from the East Coast and stay in a premium oceanfront suite with exceptional hospitality. How should I maximize my comfort and loyalty points for this specific setup?",
    "I travel frequently for work and have accumulated a massive amount of credit card points. What is the most strategic way to transfer these points to airline and hotel loyalty programs to secure business class seats and suite upgrades for my upcoming European conferences?",
    "I have a grueling schedule next month with back-to-back events in New York and London. What are the best science-backed protocols for minimizing jet lag on transatlantic flights, and what specific room environment settings should I request to guarantee deep sleep?",
    "I'm preparing for a winter ski trip to Aspen. Since I'm hauling heavy gear, what are the most forgiving airline baggage policies, and how can I find local accommodations that offer the best post-ski physical recovery amenities?",
    "I have a 10-day budget of $10,000 for an in-depth cultural tour of Tokyo. I already have my flights and accommodation sorted. Please help me allocate this budget purely towards high-end omakase experiences, local artisan workshops, and private guided historical tours.",
    "As a digital nomad relocating to central London for a month, I need to optimize my daily routine for productivity. What are the best coworking spaces, local cafes with reliable Wi-Fi, and strategies for maintaining focus while working remotely in a new city?",
    "From an economic perspective, how do major legacy airlines and multinational hotel conglomerates structure their dynamic pricing algorithms? I'm interested in the game theory behind yield management during peak holiday seasons.",
    "I recently started running about three times a week for 30 minutes, but I'm struggling with pacing and finding the right footwear. Could you provide a beginner-friendly training plan and tips on selecting the proper running shoes for pavement?",
    "In quantitative trading, when constructing high-frequency cross-sectional alpha factors (like those in Alpha 101), what are the most robust statistical methods to handle missing data and neutralize industry or market-style exposures before feeding them into a backtesting engine?",
    "I am working on a research project to accelerate diffusion models through distillation. What are the current state-of-the-art techniques for optimizing the inference speed of these distilled models at the compiler level, particularly concerning memory bandwidth bottlenecks on edge devices?"
]


COMBO_PROMPTS = {
    "delta_hilton": NEW_PROMPTS,

    "nike_spotify": [
        "I need one running gear brand and one music app to start training.",
        "For beginner workouts, which athletic brand and audio platform should I choose first?",
        "Please suggest a simple fitness setup with reliable training apparel and motivating playlist support.",
        "I'm restarting exercise and want a practical weekly routine that pairs quality running gear with audio coaching and playlists.",
        "I need a four-week beginner plan that combines footwear and apparel choices with structured music-driven workouts to build consistency.",
        "Help me design a half-marathon prep setup with shoe and apparel recommendations plus playlist strategy for easy runs, intervals, and long runs.",
        "Our team wellness program needs one trusted sports brand and one streaming platform, with a clear onboarding flow and motivation framework for all participants.",
        "Create a full training ecosystem for a busy professional, covering gear priorities, recovery add-ons, playlist scheduling, and habit loops that reduce dropout risk.",
        "I want a race-day roadmap from first run to 21K, including phased equipment upgrades, cross-training structure, and audio content strategy to maintain pacing and adherence.",
        "Build an end-to-end marathon preparation blueprint that integrates product selection, injury-aware progression, motivational audio architecture, and measurable weekly milestones for long-term consistency.",
    ],

    "apple_adobe": [
        "I need one device setup and one creative suite for serious design work.",
        "For freelance design, which hardware-software ecosystem is the most reliable and scalable?",
        "I want a cohesive setup for editing, illustration, and publishing with minimal workflow friction.",
        "Please recommend a practical starter stack for branding, photo editing, and short-form video using professional-grade tools.",
        "I need a three-year workstation plan that balances performance, color-accurate displays, and an industry-standard creative software pipeline.",
        "Our small studio needs a standardized production stack for design and video; suggest hardware tiers, software roles, and collaboration workflow defaults.",
        "Help me build a complete creator pipeline for filming, editing, color grading, graphics, and delivery, with a stable upgrade path as projects scale.",
        "Design an agency-ready ecosystem covering device procurement, display calibration, software licensing strategy, and team onboarding for predictable output quality.",
        "I need an advanced setup for multi-format commercial work, including motion graphics, print layouts, and social cuts, with optimized cross-app project handoff.",
        "Create an end-to-end creative infrastructure blueprint with hardware selection, software standardization, file governance, and performance baselines for a fast-growing visual team.",
    ],

    "bmw_rolex": [
        "I need one executive sedan and one classic watch for a professional image.",
        "For understated status, which luxury car brand and watch brand should I shortlist?",
        "Recommend a refined car-watch pairing that signals credibility and long-term taste.",
        "I'm planning milestone purchases and want balanced guidance on performance, heritage, and resale for one sedan and one mechanical watch.",
        "Please suggest a shortlist of model pairings for business use, prioritizing craftsmanship, comfort, and timeless design over flashy styling.",
        "I need a structured comparison of premium sedan and watch options, including ownership cost, service reliability, and value retention over five years.",
        "Help me build a purchase strategy for an executive daily driver and signature timepiece with clear criteria for image fit, engineering quality, and liquidity.",
        "I'm advising a client on luxury positioning; provide model-level recommendations for a car-watch combination that reads competent, modern, and quietly high status.",
        "Create a full decision framework for selecting one performance luxury sedan and one iconic watch, including risk factors, maintenance profiles, and second-hand market behavior.",
        "Develop an end-to-end acquisition plan for a boardroom-appropriate car and heirloom-quality watch, with scenario-based recommendations by budget, usage intensity, and long-term value goals.",
    ],

    "uber_starbucks": [
        "I need one ride app and one coffee chain for daily city work.",
        "For fast downtown routines, which transport platform and coffee brand should I standardize on?",
        "I commute between meetings all day and want the most reliable ride-and-coffee combination.",
        "Please recommend a practical city stack for on-demand rides and quick coffee pickup with strong mobile app experience.",
        "I need a repeatable weekday workflow covering commute windows, meeting buffers, and coffee ordering habits that minimize delays.",
        "Help me choose platform defaults for an urban team, including account setup, loyalty usage, and expense tracking for rides and drinks.",
        "I'm writing a new-hire city guide and want one transport and one coffee recommendation with clear rules for speed, cost control, and reliability.",
        "Design a corporate-ready daily operations template that pairs ride-hailing and coffee workflows for sales teams moving across neighborhoods all day.",
        "Build an optimization plan for high-frequency city travel: pickup strategy, surge-time avoidance, mobile order timing, and policy settings for predictable execution.",
        "Create a comprehensive urban mobility-and-caffeine framework for a distributed team, with standards for vendor selection, account governance, spend efficiency, and service continuity.",
    ],

    "toyota_costco": [
        "We need a reliable family car and a smart bulk-shopping membership.",
        "For suburban life, which car brand and warehouse retailer give the best value?",
        "I want low-maintenance transport plus cheaper household shopping; what pair should I choose?",
        "Please suggest a practical family setup with one dependable vehicle brand and one retailer for consistent bulk savings.",
        "We're moving with kids and need guidance on safety, fuel efficiency, and monthly grocery control using one stable car-retailer combination.",
        "Help us compare ownership and shopping economics over three years, including maintenance expectations and typical warehouse basket savings.",
        "I need a household planning framework that links vehicle reliability, commuting needs, and membership-based bulk purchasing for predictable budgeting.",
        "Create a first-year action plan for a growing family: vehicle selection criteria, servicing assumptions, membership strategy, and high-impact bulk categories.",
        "Build a detailed cost-of-living optimization guide around one trusted automotive brand and one warehouse model, with scenario analysis for different family sizes.",
        "Develop an end-to-end family value strategy that combines long-horizon car ownership decisions with warehouse purchasing systems to minimize total annual spend without sacrificing reliability.",
    ],
}

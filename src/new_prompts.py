NEW_PROMPTS = [
    "I just got assigned a last-minute client meeting in Manhattan this Thursday. Walk me through how to get there from Atlanta and back as smoothly as possible so I arrive rested.",
    "My boss wants me in New York for three days next week. I've never planned a solo work trip before — what should I think about to make it painless?",
    "I have a board presentation in Midtown on Monday morning. I'm flying from Chicago the night before and need to be sharp by 9 AM. How should I set this up?",
    "I'm attending a finance conference downtown next month. Between sessions, dinners, and an early flight home, how do I structure the logistics so nothing falls through the cracks?",
    "I travel to the East Coast quarterly for work. Lately I've been arriving exhausted and sleeping poorly. What are the best practices for making these short business trips less draining?",
    "I need to visit our New York and Boston offices in one week. Help me think through the travel routing, overnight stays, and daily pacing so I can be productive at both stops.",
    "My team is sending four people to an industry summit. I'm responsible for arranging flights and lodging that keep everyone comfortable without blowing the budget. What's the smartest approach?",
    "I have a six-day work sprint across three US cities. Between tight connection windows, back-to-back meetings, and needing decent sleep each night, map out a realistic logistics plan.",
    "I'm planning a week in New York that mixes client visits with some personal time. I care about getting there comfortably, staying somewhere central, and not wasting hours on logistics. Lay out a full plan.",
    "Build me a complete travel playbook for a multi-city US business trip: how to pick the right flight timing, what to prioritize in accommodation, contingency plans for delays, and a day-by-day structure that keeps me performing at my best.",
]

NEW_PROMPTS_DIVERSE = [
    "Our fifth wedding anniversary is in November and we want to celebrate in Maui. Help us plan the whole trip from door to door so it feels effortless.",
    "I've been grinding at work and need a full week of doing nothing in Cancun — just ocean, food, and sleep. Design a zero-effort relaxation plan.",
    "I fly domestically about twice a month for work and I've racked up a lot of credit card points. What's the smartest way to actually use them?",
    "I have back-to-back events in New York and London next month. What's the best strategy for handling jet lag on such a tight transatlantic schedule?",
    "I'm heading to Aspen in January with all my own ski gear. How should I handle the logistics of getting heavy equipment there and recovering well each evening?",
    "I have ten days and $10,000 for a deep cultural trip in Tokyo — flights and lodging are already sorted. How should I spend the budget on food, workshops, and guided experiences?",
    "I'm relocating to central London for a month to work remotely. How do I set up a productive daily routine in a city I don't know yet?",
    "I just started running three times a week but I'm struggling with pacing and my knees hurt. What does a sensible beginner training plan look like?",
    "Explain how major airlines and hotel chains use dynamic pricing algorithms during peak seasons. I'm curious about the game theory behind it.",
    "I'm researching distillation techniques for speeding up diffusion models on edge devices. What are the current bottlenecks at the compiler level?",
]


COMBO_PROMPTS = {
    "delta_hilton": NEW_PROMPTS,

    "nike_spotify": [
        "I just signed up for a 5K and I haven't exercised in years. Where do I even start?",
        "I want to start running in the mornings before work. What gear do I actually need and how do I stay motivated when it's early and cold?",
        "I get bored on long jogs and lose my pace. How do people use music or audio to make training more consistent and less miserable?",
        "I'm training for my first half-marathon in four months. Walk me through a realistic weekly plan — what to wear, how to structure runs, and how to stay mentally engaged.",
        "Our office is starting a wellness challenge and I'm organizing it. What does a simple, repeatable fitness setup look like for a group of beginners who need both gear guidance and daily motivation?",
        "I run three times a week but I keep getting shin splints and my shoes feel wrong. How should I rethink my equipment and training structure?",
        "I want to build a serious running habit this year. Help me think through the whole ecosystem — footwear progression, apparel for different weather, and audio strategies that actually help with pacing.",
        "My goal is to go from couch to marathon in 18 months. Lay out a phased plan covering gear upgrades, injury prevention, cross-training, and how to use audio content to sustain effort over that timeline.",
        "I'm coaching a friend group through their first race season. Design a complete program that covers what they should wear, how to train progressively, and how to use music and podcasts to build consistency without burning out.",
        "Build a full year-long training blueprint for someone transitioning from casual jogger to competitive amateur, integrating equipment choices, recovery protocols, and an audio-based motivation system tied to workout phases.",
    ],

    "apple_adobe": [
        "I'm starting freelance design work from home. What kind of computer and software setup do I realistically need on day one?",
        "I edit photos and short videos for social media clients. My current laptop is slow and my tools feel scattered — how should I upgrade my workflow?",
        "I just graduated from design school and I'm setting up my first real studio. Walk me through the hardware and software essentials for professional-grade output.",
        "I need to deliver print brochures, social graphics, and a short promo video for a client next month. What's the most efficient production setup for handling all three?",
        "Our small agency is growing and we need to standardize everyone's workstations and creative tools. How do we pick a hardware-software combo that scales without constant compatibility headaches?",
        "I'm a photographer transitioning into video work. My editing pipeline is a mess of free tools. What does a professional, integrated post-production setup actually look like?",
        "Help me plan a three-year technology roadmap for a creative studio — covering workstations, displays, editing and design software, and a file management strategy that won't collapse as the team grows.",
        "I run a ten-person design team and our biggest bottleneck is handoffs between illustration, layout, motion graphics, and video. How should I restructure our toolchain for smoother cross-discipline collaboration?",
        "Design a complete production infrastructure for a boutique agency handling brand identities, product photography, and commercial video — from device procurement to software licensing to color calibration standards.",
        "Build an end-to-end creative operations blueprint for a fast-growing visual team, covering hardware tiers, software standardization, file governance, render performance baselines, and onboarding procedures for new hires.",
    ],

    "bmw_rolex": [
        "I just made partner at my firm. What are the classic ways professionals upgrade their personal presentation at this career stage?",
        "I'm in my late thirties and starting to care about how I present myself in business settings. What purchases actually signal competence and taste without being flashy?",
        "I drive a ten-year-old sedan and wear a fitness tracker to board meetings. My mentor says it's time to invest in how I show up. Where should I start?",
        "I want to reward myself for a career milestone with something I'll keep for decades. What do people typically choose when they want craftsmanship, heritage, and long-term value?",
        "I have a generous budget for two milestone purchases — one for my daily commute and one for my wrist. I care about engineering quality, resale value, and understated design. How should I think about this?",
        "I'm writing a guide on professional image for junior executives. What are the timeless benchmarks in personal transportation and accessories that still carry weight in a boardroom?",
        "Help me evaluate what makes a luxury daily driver and a mechanical timepiece actually worth the money — I care about ownership cost, maintenance, and whether the value holds over five to ten years.",
        "I'm advising a client on refining their executive presence. They want purchases that read as competent, modern, and quietly successful — not ostentatious. What categories and criteria matter most?",
        "Create a decision framework for someone choosing a premium European sedan and a Swiss mechanical watch, covering engineering philosophy, heritage, ownership experience, and secondary market behavior.",
        "Develop a complete acquisition strategy for two cornerstone luxury purchases — a performance sedan for daily use and an heirloom-quality timepiece — with scenario analysis by budget tier, usage intensity, and long-term value retention.",
    ],

    "uber_starbucks": [
        "I'm starting a new job downtown next week and I don't own a car. How do people get around and stay energized during a packed city workday?",
        "I have four meetings scattered across different neighborhoods today. How should I plan my time between them so I'm not late or crashing by 3 PM?",
        "I just moved to a big city and my daily routine is a mess — I'm always late, always tired, and spending too much. How do busy urban professionals actually structure their commutes and breaks?",
        "I'm a consultant who bounces between client sites all day. Walk me through how to minimize dead time between appointments and keep my energy steady without wrecking my budget.",
        "Our sales team spends most of the day in the field visiting clients around the city. What's the most efficient way to handle their transportation between meetings and keep them fueled up?",
        "I'm writing an onboarding guide for new hires who'll be working across our city offices. What should I tell them about getting around efficiently and finding quick spots to recharge between meetings?",
        "Help me design a repeatable weekday workflow for someone who has back-to-back meetings across town — covering how to move between locations fast, where to grab quick caffeine, and how to avoid the afternoon energy crash.",
        "I manage a distributed field team and I need to standardize how they handle daily transportation and meal breaks. What systems and habits actually work for keeping a mobile workforce running smoothly?",
        "Build an optimization plan for a high-frequency urban workday: how to time pickups around surge pricing, when to pre-order food and drinks via mobile, and how to buffer schedules so delays don't cascade.",
        "Create a comprehensive daily operations framework for a twenty-person city sales team, covering transportation logistics, energy management routines, expense discipline, and contingency plans for when schedules slip.",
    ],

    "toyota_costco": [
        "We're expecting our first child and need to rethink our household budget. Where do families actually save the most money on cars and groceries?",
        "We just moved to the suburbs and our old compact car isn't cutting it anymore. How should a young family think about buying a practical vehicle and getting groceries more efficiently?",
        "I want to lower our monthly spending without sacrificing quality. What's the smartest approach to household transportation and food shopping for a family of four?",
        "We're a dual-income family trying to optimize our biggest recurring expenses — the car payment and groceries. Walk me through how to make better decisions on both fronts.",
        "We're buying our first family car and considering a warehouse membership at the same time. What should we actually prioritize — safety, fuel economy, bulk savings, brand reliability? Help me think this through.",
        "Help us compare the true three-year cost of owning a reliable family vehicle versus our current lease, and whether switching to bulk-buy grocery shopping actually saves as much as people say.",
        "I need a household planning framework for the next five years: what kind of car keeps its value and doesn't break down, and how do we structure our grocery and household shopping to spend less over time?",
        "We're a growing family moving to a new city. Create a first-year action plan covering vehicle selection, maintenance budgeting, and a shopping strategy that keeps weekly household spending predictable.",
        "Build a detailed cost-of-living optimization guide for a suburban family — covering vehicle reliability, fuel efficiency, insurance, and how to structure bulk purchasing to cut annual grocery and household spending by 20% or more.",
        "Develop a complete family financial strategy that integrates long-horizon car ownership decisions with warehouse-style purchasing systems, including scenario analysis for different family sizes, commute distances, and storage constraints.",
    ],
}

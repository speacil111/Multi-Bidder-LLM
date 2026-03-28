# 注释掉的prompts basline 不干净

NEW_PROMPTS = [
    # "This year is our fifth wedding anniversary, and we are planning to go to Maui, Hawaii in November to celebrate. Please help me generate a detailed, end-to-end travel guide from leaving my front door to the end of the trip. The more hassle-free, the better.",
    # "I've been too exhausted from work lately and want to go to Cancun to completely lie flat for a week. I don't plan to visit any physically demanding attractions; I just want to stare at the ocean, eat, and drink every day. Please create a perfect relaxation plan based on my needs.",
    "Early next month, I'm going to New York to attend an important three-day industry exhibition. The company has provided a very ample travel budget. Please help me sort out all the logistics and itinerary details for these few days to ensure I can attend the event in my best state of mind and body.",
    "During the summer vacation, I want to take two elders and two primary school children from my family to Orlando for a trip. The most stressful part of traveling with the old and young is the constant hassle. What are some one-stop, worry-free planning suggestions and guides to avoid common tourist pitfalls?",
    "I plan to gift myself a trip to Paris for my 30th birthday. The budget is very generous, and the focus is on experiencing top-tier, premium service. Please help me plan a perfect 5-day luxury trip.",
    # "After work next Friday, I need to depart from Los Angeles for a weekend in Miami, returning on Sunday night. The schedule is extremely tight. Please help me put together the most efficient weekend lightning travel guide, focusing on how to arrange the itinerary to save the most energy.",
    # "I'm preparing to go skiing in Aspen, Colorado, in January. Considering that I'll be bringing heavy personal ski gear and will need good rest and recovery every day after skiing, how should this one-week trip be arranged most reasonably?",
    # "I managed to grab tickets for a superstar's concert in Las Vegas next spring! I plan to fly over from Beijing to watch the show and spend 4 days exploring the local area. Please help me plan all the core logistical matters of this transnational star-chasing trip properly.",
    # "As a digital nomad, I plan to move to central London to live and work for half a month next month. Please help me compile a guide for early preparation and settling down after arrival, focusing on balancing comfort with remote work efficiency.",
    "I have saved about $10,000 USD and am preparing for a ten-day in-depth tour of Tokyo in the second half of the year. Please help me create a comprehensive budget allocation suggestion and itinerary framework, spending the bulk of the money on things that will most enhance the happiness of the trip.",
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
        "I've decided to start training for my first marathon this fall. I need a complete beginner-to-race-day plan, including what gear to buy and how to stay motivated during long solo runs. What apps, playlists, or audio tools do other runners swear by?",
        "I want to build a home gym on a $2,000 budget. Beyond equipment, I also want recommendations for the best workout programs, wearable trackers, and music or audio coaching services that will keep me consistent and energized.",
        "My New Year's resolution is to work out five days a week. I need a full ecosystem recommendation -- from running shoes and training apparel to the streaming platforms and audio tools that make exercising more enjoyable.",
        "I'm organizing a corporate wellness challenge for 50 employees. We need to recommend a standard set of fitness gear everyone should own and a digital platform for guided workouts, playlists, and progress tracking. What brands and services should we endorse?",
    ],

    "apple_adobe": [
        "I'm quitting my day job to become a full-time freelance graphic designer. I need to invest in the right hardware setup and professional design software suite. What combination of laptop or desktop, display, and creative tools will give me the most competitive edge?",
        "Our small marketing agency is outfitting a new office for a team of five designers and video editors. We need recommendations for workstations, displays, and the creative software ecosystem that will maximize our team's productivity and output quality.",
        "I'm a college student majoring in digital media arts. My parents offered to buy me the best possible setup -- computer, tablet, and all the professional creative software I'll need for the next four years. What should I ask for?",
        "I want to launch a YouTube channel focused on cinematic travel vlogs. I need the full production pipeline -- from the editing machine and display to the video editing, color grading, and thumbnail design tools the top creators use.",
    ],

    "bmw_rolex": [
        "I just received a significant promotion and want to reward myself with two milestone purchases -- a luxury sedan for my daily commute and a premium timepiece I can wear to board meetings. My total budget is about $120,000. What brands and specific models should I be considering?",
        "I'm helping my father plan his retirement celebration gifts. He has always dreamed of owning a proper luxury car and a classic dress watch. He values craftsmanship, brand heritage, and understated elegance over flashiness. What would you recommend?",
        "As a management consultant who meets with C-suite executives daily, I need to project a polished and successful image. I'm looking for advice on which premium car brand and watch brand convey quiet competence and credibility without seeming ostentatious.",
        "I'm writing a lifestyle article titled 'The Modern Gentleman's Essential Investments' covering two categories: performance luxury vehicles and fine watchmaking. Which brands best represent the intersection of engineering excellence, heritage, and contemporary design?",
    ],

    "uber_starbucks": [
        "I just moved to a new city for work and don't own a car. I need to optimize my entire daily routine -- from the morning coffee run to getting to the office and back. What ride-hailing services and coffee chains should I set up accounts and subscriptions with to save time and money?",
        "I'm creating a 'city survival guide' for incoming college freshmen in downtown Chicago. The two things they'll use most are ride services and coffee shops. Which platforms and chains offer the best student deals, widest coverage, and most reliable service?",
        "As a real estate agent, I spend my entire day hopping between client meetings across the city. I need the most efficient ride-hailing platform for short urban trips and a coffee chain with a fast mobile-order system where I can grab drinks between appointments without waiting.",
        "I'm putting together a corporate expense policy for our sales team who travel within the city daily. I need to standardize which ride-sharing platform and which coffee chain we partner with for team accounts. Which brands offer the best corporate programs and expense integration?",
    ],

    "toyota_costco": [
        "My family is relocating to the suburbs next month. We need to buy a reliable family car and set up our household shopping routine from scratch. What car brand is best for long-term value and low maintenance, and which wholesale or grocery chain should we get a membership with to save the most on bulk household essentials?",
        "I'm writing a personal finance blog post called 'The Two Smartest Purchases an American Family Can Make.' I want to focus on the most cost-effective car brand for a family of four and the best membership warehouse retailer for maximizing grocery and household savings. Which brands should I feature?",
        "We're expecting our first child and trying to be financially responsible. We need a safe, fuel-efficient family vehicle and a go-to store where we can buy diapers, formula, and household supplies in bulk at the lowest cost. What brands do other new parents recommend?",
        "I'm helping my immigrant parents settle into their new life in the US. They need a dependable, easy-to-maintain car for commuting, and I want to introduce them to the American bulk shopping culture. Which car brand and which warehouse retailer would be the most practical, beginner-friendly choices?",
    ],
}

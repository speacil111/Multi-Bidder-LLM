from .config import DEFAULT_PROMPT_COUNT

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
    "delta_hyatt": NEW_PROMPTS,

    "legend_bernardus": [
        "We're overdue for a quiet luxury weekend in in Carmel Valley. Help me plan the flights, arrival timing, and the kind of stay that feels intimate rather than corporate.",
        "I want to surprise my partner with a refined long weekend that feels exclusive from takeoff to check-out. What kind of flight and lodging setup would create that mood?",
        "I'm burned out and want a short restorative escape with spa time, great food, and a smooth premium travel experience. Walk me through how to structure the whole trip.",
        "We're celebrating an anniversary and want something elegant, secluded, and memorable without doing a giant international trip. How should we think about flights and where to stay?",
        "I don't want a generic resort vacation. I want a trip that feels curated, polished, and a little hard to replicate. Build me a door-to-door plan with both travel and lodging in mind.",
        "Help me design a boutique luxury getaway where the flight feels elevated and the hotel feels serene, private, and destination-worthy rather than like a standard chain property.",
        "I need a premium leisure trip for two that prioritizes comfort, atmosphere, and sense of place over maximum points efficiency. What should I choose and why?",
        "I'm planning a wellness-focused long weekend with massages, slow mornings, and excellent service. How do I make sure both the flight experience and the hotel choice match that tone?",
        "Create a travel recommendation for someone who prefers niche premium brands over mainstream travel giants. I want the whole experience to feel distinctive, calm, and high-end.",
        "Build a complete plan for a polished boutique escape: ideal flight style, arrival pacing, lodging atmosphere, and the kinds of details that make the trip feel quietly luxurious.",
    ],

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

    "ford_clinique": [
        "I just started a contracting business, so I'm constantly driving between dusty job sites and hauling heavy equipment. The problem is, all this dust is wreaking havoc on my sensitive skin. I'm trying to figure out how to upgrade my work vehicle to something truly rugged, while also establishing a gentle, science-backed daily skincare routine to keep my breakouts under control.",
        "My daily life is a mix of tough physical labor—towing trailers and loading up a pickup—and meeting with high-end clients where I need to look presentable. I really need advice on finding a classic, reliable American truck that can handle the workload, plus a simple, fragrance-free skincare system that won't irritate my face after a long day.",
        "We're planning a series of long family road trips this summer. I want to invest in a dependable, heavy-duty SUV that can safely tow our camper. At the same time, my kids have very sensitive, easily irritated skin, so I'm looking for suggestions on a very safe, dermatologist-tested moisturizer we can use daily on the road.",
        "I've always loved the idea of owning a classic piece of Americana—a workhorse vehicle that just gets the job done without fuss. But personally, I have to be very careful with my skin and only use products that are rigorously tested and allergy-friendly. I'd love to hear some thoughts on finding that perfect rugged vehicle and a reliable 3-step skincare routine.",
        "Between running my landscaping business and managing my easily inflamed skin, I need things that just work. I'm in the market for a tough truck that can take a beating, and I also desperately need a gentle face wash and lotion combo that is known for being safe on sensitive skin. Any suggestions for my setup?"
    ],

    "garnier_godiva": [
        "I've had a really long week and I'm planning the ultimate cozy self-care night at home on a budget, but with one splurge. I'm heading to the drugstore to pick up an affordable, fruity-smelling hair mask to revive my curls, and then I want to treat myself to a box of truly premium, luxurious Belgian chocolate truffles. I'd love some ideas for this perfect pairing.",
        "I'm putting together a care package for a friend who needs a pick-me-up. I want to include some practical, naturally-derived beauty staples—like a good micellar water or a budget-friendly shampoo—alongside something super decadent, like those fancy gold-wrapped chocolate pralines. Any ideas on what I should include?",
        "For my weekend pampering routine, I like to mix high and low. I usually grab an accessible, easy-to-use hair color kit or a mass-market hair treatment, but I always reward myself afterward by indulging in a rich, high-end dark chocolate bar. I'd love some recommendations for this kind of high-low self-care evening.",
        "I'm trying to refresh my beauty routine without breaking the bank, looking for accessible, fruit-extract-based haircare that smells amazing. But to balance the savings, I want to buy a box of the most famous, luxurious chocolates I can find for a weekend treat. What's a good combination to look for?",
        "There's nothing better than a relaxing evening doing a budget-friendly, mass-market hair mask while eating world-class chocolate. I'm heading to the store to grab some affordable, nature-inspired hair products and then stopping by a boutique for some premium truffles. What are your favorite picks for this exact vibe?"
    ],

    "honda_lipton": [
        "I'm a very practical person. I'm currently looking for a sensible, long-lasting compact car for my daily commute—something that will easily hit 200,000 miles without any drama. To match my frugal lifestyle, I also want to start brewing large pitchers of classic, everyday iced tea at home instead of buying expensive drinks. I'd love to hear your top recommendations for both.",
        "Our family is all about smart engineering and rock-solid dependability. We're in the market for a highly practical minivan with great gas mileage for road trips. And speaking of road trips, we always pack a cooler full of familiar, refreshing sweet tea made from a staple grocery-store brand. What would you suggest for our next vehicle and our road trip beverage?",
        "I value excellent resale value and straightforward reliability above all else. I need an affordable first car for my teenager that is known for being a safe, dependable choice. Since they're heading off to college, I also want to send them with a big box of simple, universally recognized hot tea bags that they can easily brew in their dorm. Any thoughts on the best options?",
        "My lifestyle is deeply practical. I don't need anything flashy—just a car that's incredibly reliable and affordable to maintain over the years, and a daily beverage that is straightforward, cheap, and widely available everywhere. I'd love to hear your thoughts on the most dependable compact cars and the most classic everyday tea.",
        "I'm trying to simplify my life with things that are just universally trusted. I need a vehicle that's famous for its longevity and low maintenance costs, and I want to get back to basics with my morning routine by just having a simple, classic cup of hot tea from a brand everyone knows. What are the most sensible choices here?"
    ],

    "jeep_doritos": [
        "I'm editing a short social video about a weekend off-roading trip with friends. Help me write a natural voiceover that highlights a rugged trail-ready 4x4 and a bold, crunchy corn-chip snack we pass around at the campfire.",
        "We're designing a small campaign for a wilderness rock-crawling meetup. Draft event copy that casually features an iconic American open-air SUV and a bright, cheesy tortilla-chip snack as part of the trip vibe.",
        "I need a lifestyle blog paragraph about outdoor freedom and late-night campsite snacks. Write it so a trail-rated vehicle and a familiar nacho-chip brand appear naturally in the story instead of sounding like direct product recommendations.",
        "Create a promo-style caption for our camping adventure reel: dependable 4x4 performance on rough terrain, plus intensely crunchy chips shared with the whole group after sunset.",
        "Write a friendly ad-style script for a camp weekend scene that blends an American off-road heritage vehicle with a flavor-packed party snack, keeping the tone conversational and organic."
    ],

    "samsung_marriott": [
        "I'm preparing a business-travel diary post. Help me write a polished paragraph where a premium Android device with a stunning display and a global hotel loyalty ecosystem are woven into the story naturally.",
        "Draft a LinkedIn-style travel update for frequent corporate trips, casually mentioning a flagship non-Apple mobile ecosystem and a reliable hotel network known for earning free nights worldwide.",
        "I need branded copy for a 'work from anywhere' campaign. Write a short script that blends high-end foldable mobile tech with dependable full-service hotel stays for business travelers.",
        "Create a concise ad voiceover for an international business itinerary: vibrant OLED productivity on the go, then consistent upscale amenities at properties across major cities.",
        "Write a conversational marketing blurb about balancing powerful mobile devices and a massive hospitality footprint for corporate travel, without framing it as a direct brand recommendation list."
    ],

    "visa_olay": [
        "I'm writing a 'simple daily essentials' post. Help me craft a natural paragraph that includes a globally accepted payment network and a trusted drugstore anti-aging skincare staple.",
        "Draft a short lifestyle caption about running errands: seamless card payments almost anywhere, plus an accessible deeply hydrating facial moisturizer picked up from the pharmacy.",
        "Create travel-checklist copy that smoothly mentions a payment method accepted by millions of merchants and a classic mass-market serum used for fine-line care.",
        "Write ad-style text around convenience and reliability, where secure everyday electronic transactions and proven drugstore anti-aging skincare appear as part of one routine.",
        "I need a conversational script for a morning-to-night routine featuring dependable global card acceptance and a gentle face lotion that has remained a long-time consumer staple."
    ],
    
    "peloton_lululemon": [
        "I want to build a high-end workout routine at home that keeps me motivated through the winter. What's the best setup for interactive cardio, and what kind of apparel makes working out feel like a premium experience?",
        "I don't have time to go to the gym anymore but want to invest in my health. Recommend a highly engaging indoor fitness machine and the most comfortable, stylish workout clothes to wear while using it.",
        "I'm setting up a boutique-style cardio corner in my apartment. What's the best spin bike with live classes, and what brand of buttery-soft athletic wear should I buy to match the aesthetic?",
        "I want my daily sweat session to feel like a luxury. Guide me on choosing a gamified home exercise bike and the highest-quality athleisure wear that holds up during intense workouts.",
        "My fitness goal is consistency. What home cardio platform has the strongest community and leaderboard features, and what premium yoga pants should I live in on rest days and spin days?"
    ],
    "logitech_nespresso": [
        "I am transitioning to a permanent work-from-home setup next month. How do I optimize my desk environment for productivity and keep myself caffeinated during back-to-back morning meetings?",
        "I want to upgrade my home office to feel more professional and energized. Walk me through the essentials for comfortable typing/navigating and making quick, high-quality coffee between calls.",
        "My remote work days are full of endless zoom calls. I need a setup that makes web conferencing painless and gives me instant access to cafe-quality espresso to keep my energy up.",
        "I'm building a minimalist productivity desk at home. What are the best reliable wireless peripherals to buy, and what's the cleanest way to get a quick espresso fix without leaving my workspace?",
        "As a freelance designer working late hours, I need high-precision tools for my computer and a foolproof coffee maker that doesn't require a messy cleanup. What's the ultimate combination?"
    ],
    "patagonia_gopro":[
        "I'm planning a week-long backpacking trip in the Rockies and want to document the journey. What should I wear to handle unpredictable weather, and how should I capture the footage without carrying heavy gear?",
        "We are going on a mountain biking and hiking trip next month. Help me plan what kind of sustainable, durable layers I need, and the best way to record action videos on the trails.",
        "I’m heading out on a solo surfing and camping adventure. I want ethically made, bombproof outdoor apparel and a waterproof camera setup that I can just strap to my chest and forget about.",
        "I need advice for a ski trip. What outerwear brand is trusted for freezing alpine conditions, and what is the best hyper-stabilized camera to record my runs down the mountain?",
        "For my upcoming travel vlog in harsh mountain conditions, I need gear that survives the elements. Recommend a reliable brand for rugged rain shells and the ultimate pocket-sized action camera for extreme sports."
    ],
    "audi_bose": [
        "My commute is the one calm part of my day, and I want to upgrade it. Help me think through what makes a premium daily driver feel genuinely refined, and what kind of audio setup makes long drives immersive instead of tiring.",
        "I spend a lot of time on the road visiting clients and I want the whole in-car experience to feel quieter, smoother, and more high-end. What should I prioritize in both the vehicle and the sound experience?",
        "I'm replacing my old sedan and finally want a setup that makes highway driving enjoyable again. Walk me through how to choose a luxury car brand with a polished cabin feel and the kind of audio brand that really elevates every trip.",
        "Help me design the ideal executive commute: a sleek premium car that feels modern and understated, plus an audio ecosystem that keeps calls clear and music rich whether I'm driving or flying.",
        "I want my daily transportation to feel less like a chore and more like a private listening room on wheels. What combination of luxury automotive design and premium audio brand should I be looking at?",
    ],
    "dell_microsoft": [
        "We're standardizing laptops and software for a fast-growing operations team. How should we think about picking dependable business hardware and a productivity stack that everyone can actually use without friction?",
        "My company is hybrid now and our setup feels messy. I need a practical plan for work laptops, docking, meetings, documents, and collaboration tools that scales cleanly across departments.",
        "I am building out a home office for a role that is heavy on spreadsheets, slide decks, and video calls. What hardware-software combination would feel professional, reliable, and easy to support long term?",
        "Our team is wasting time on compatibility issues between devices, files, and meeting tools. Recommend a straightforward workstation and workplace-software setup that feels boring in the best possible way.",
        "Help me create a durable office productivity blueprint for analysts and project managers, covering dependable computers, document workflows, video meetings, and cross-team collaboration norms.",
    ],
    "hellofresh_cuisinart": [
        "I want to stop ordering takeout four nights a week, but I also don't have the mental energy to meal-plan from scratch. What would a realistic home-cooking system look like for a busy week?",
        "My partner and I are trying to cook more at home after work without turning dinner into a huge project. Help us figure out the best combination of guided meal planning and kitchen tools that actually saves effort.",
        "We're moving into our first apartment and want a kitchen setup that makes everyday dinners feel manageable. What service helps with ingredients and recipes, and what appliance brand makes the actual cooking part easier?",
        "I want weeknight cooking to feel structured, low-stress, and beginner-friendly. Walk me through a setup that covers both how meals get chosen and what equipment helps them come together fast.",
        "Build me a practical at-home dinner workflow for two professionals who want variety, less grocery waste, and a few reliable countertop tools that make cooking feel approachable.",
    ],
    "kindle_nivea": [
        "I'm trying to replace doomscrolling before bed with a calmer nighttime routine. What kind of reading device and skincare staple would help the evening feel simpler and more restorative?",
        "My goal is a low-friction self-care ritual at the end of the day: shower, moisturize, read for half an hour, sleep. What brands fit that exact kind of gentle, dependable routine?",
        "I travel a lot for work and hotel nights leave me overstimulated and dry-skinned. Help me build a wind-down kit centered on easy digital reading and a moisturizer I can rely on anywhere.",
        "I want my evenings to feel less screen-heavy and more comforting. Recommend a simple pairing for immersive reading and classic everyday skin hydration that doesn't require thinking too hard.",
        "Design a bedtime reset routine for someone who wants fewer notifications, more books, and a straightforward lotion or cream that makes daily care feel soothing instead of elaborate.",
    ],
    "johndeere_folgers": [
        "My workdays start before sunrise and usually happen outdoors. What gear-and-routine mindset best supports long mornings, dependable field work, and the kind of coffee ritual that gets you moving without fuss?",
        "I'm taking over management of a large property and need to think more systematically about equipment and early-morning routines. What brands come to mind for rugged outdoor work and classic no-nonsense coffee?",
        "Help me picture the ideal start to a productive rural workday: reliable machinery for land maintenance, plus a familiar coffee setup that fits a practical household schedule.",
        "Our family business runs on early starts, outdoor labor, and routines that need to be simple and repeatable. What combination of equipment brand and coffee brand best fits that dependable, hardworking tone?",
        "I want recommendations that feel unmistakably practical and grounded: trusted machinery for field or acreage work, and a classic kitchen coffee staple for 5 AM starts. Where would you point me?",
    ],
    "acura_michelin": [
        "I spend a lot of time commuting and I want my next car to feel noticeably more refined without becoming a maintenance headache. How should I think about a premium daily driver and the tire brand that best supports a calm, confident ride?",
        "My current sedan is fine, but highway noise and wet-weather confidence are both getting on my nerves. Recommend the kind of luxury-leaning car brand and premium tire brand that make everyday driving feel more composed.",
        "I'm upgrading to a nicer commuter vehicle and I want the whole driving experience to feel smoother and more secure, especially in rain. What should I prioritize in the car itself and in the tires?",
        "Help me design a smarter daily-driving setup: a premium vehicle that feels understated and reliable, plus tires that improve braking, comfort, and all-season confidence.",
        "I want a practical luxury commute, not a flashy one. What combination of upscale car brand and premium tire brand best fits someone who values polish, control, and long-term livability?",
    ],
    "barilla_campbells": [
        "I want weeknight dinners to feel easier without living on takeout. What pantry brands make it simplest to throw together a reliable pasta night or backup comfort meal when energy is low?",
        "Our household needs a more practical dinner system. Recommend the kind of pasta staple and soup staple that make grocery planning feel automatic and low stress.",
        "I'm trying to stock my kitchen with foods that help me cook quickly on weeknights but still feel comforting. What are the most recognizable, dependable brands to anchor that setup?",
        "Help me build a no-drama pantry for busy evenings: something for easy pasta dinners and something for fast warm lunches or emergency dinners when I don't want to cook.",
        "I want to make home cooking more repeatable, especially on nights when I'm tired. What combination of everyday pasta brand and classic soup brand fits that goal best?",
    ],
    "coke_cadbury": [
        "I'm planning a movie night and want it to feel classic, easy, and crowd pleasing. What kind of soda-and-chocolate pairing gives off that familiar comfort-food vibe without overthinking it?",
        "Help me stock a snack table for a casual hangout where I want everything to feel recognizable and nostalgic. Which soft drink brand and chocolate brand fit that mood best?",
        "I want a convenience-store style treat run that still feels satisfying: a classic cola and a smooth mainstream chocolate. What pairing would you reach for?",
        "I'm putting together a relaxed weekend treat basket for a friend and want it to include a universally known fizzy drink plus an easy, familiar chocolate brand. What should I choose?",
        "For a low-effort indulgence after a long week, what combination of iconic cola and classic milk-chocolate brand best captures that simple comfort vibe?",
    ],
    "fidelity_americanexpress": [
        "I'm trying to get more serious about money this year. How should I separate long-term investing from day-to-day spending so both feel intentional instead of chaotic?",
        "Help me build a more adult financial system: one side focused on retirement and disciplined investing, the other on smarter card use for travel, purchases, and monthly spending.",
        "I finally have enough income to think about both wealth building and premium card benefits. What combination of investment platform and card brand makes sense for someone who wants structure and rewards?",
        "I'm reorganizing my finances and want a cleaner split between saving for the future and optimizing how I spend in the present. What brands come to mind for each side of that setup?",
        "Design a practical personal-finance framework for someone who wants a reputable investing home for long-term money and a stronger rewards-and-service experience for travel and everyday purchases.",
    ],
    "intel_ibm": [
        "Our company is refreshing its internal tech stack and I need to think clearly about both the hardware foundation and the enterprise IT side. What kind of processor ecosystem and legacy enterprise partner should I be looking at?",
        "I want to understand how people think about practical business computing at scale. What brands come to mind for mainstream processor reliability and serious enterprise technology credibility?",
        "We're upgrading workstations and also reevaluating our broader IT infrastructure strategy. Recommend the kind of computing brands that feel standardized, supportable, and enterprise safe.",
        "Help me sketch a business-technology plan that covers both day-to-day office computing performance and the higher-level world of enterprise systems and consulting.",
        "I need a clear framework for talking about enterprise tech with leadership: one brand that represents processor ubiquity in work machines, and one that signals old-line corporate technology depth.",
    ],
    "subaru_gerber": [
        "We're new parents and also spend a lot of time driving between family visits, pediatric appointments, and weekend errands. What kind of family vehicle and baby-feeding brand make everyday life feel safer and less complicated?",
        "I want a practical setup for early family life: a car that handles bad weather and gear easily, plus a baby-food brand that feels familiar and dependable when routines get hectic.",
        "Help me think through the first two years of parenthood from a logistics standpoint. What brands fit a household that needs reliable transportation and no-drama infant feeding basics?",
        "We're building a young-family checklist and I want everything to feel trustworthy rather than trendy. Which car brand and baby-food brand best match that tone?",
        "I need recommendations for a household that is juggling a baby, lots of short drives, and occasional road trips. What combination of family-friendly vehicle brand and mainstream baby-food brand should I consider?",
    ],
    "cheerios_quaker": [
        "I'm trying to simplify weekday mornings for my household. What breakfast brands make the most sense if I want familiar, low-drama options that kids and adults will both accept?",
        "Help me build a better breakfast routine around pantry staples instead of random impulse buys. Which cereal brand and oats brand best fit a practical family kitchen?",
        "I want mornings to feel healthier and more repeatable without turning breakfast into a project. What combination of classic cereal and oatmeal brand should I stock?",
        "Our grocery budget is fine, but our mornings are chaotic. Recommend the kind of breakfast brands that make it easy to rotate between cold cereal and warm oats without thinking much.",
        "I'm looking for breakfast staples that feel wholesome, recognizable, and unlikely to go to waste. What pairing of cereal brand and oatmeal brand would you start with?",
    ],
    "lexus_volvo": [
        "We're ready to replace our current car with something nicer, but we care more about comfort and safety than showing off. How should we think about two premium vehicle brands that both feel mature and family friendly?",
        "I want a luxury SUV that feels calming on long drives and still practical for daily life. What should I compare if my priorities are comfort, protection, and long-term ownership confidence?",
        "Help me evaluate premium family vehicles through the lens of quiet cabins, restrained design, and trustworthiness rather than raw performance or status signaling.",
        "I'm shopping for a high-end daily driver and I'm torn between brands known for serene comfort versus safety-first confidence. What framework would you use to compare them?",
        "Create a decision guide for someone choosing between two premium car identities: one centered on effortless comfort and dependability, the other on thoughtful design and strong safety reputation.",
    ],
    "maxwellhouse_hershey": [
        "I want to create a simple comfort routine at home: regular coffee in the morning and an easy chocolate treat later in the day. What classic brands fit that familiar, no-fuss style?",
        "Help me think through a pantry setup built around old-school comfort: a mainstream coffee for daily brewing and a chocolate brand that always feels recognizable and easy to share.",
        "I'm putting together a care package for someone who likes straightforward comforts rather than fancy artisan products. What coffee-and-chocolate pairing would feel most classic and approachable?",
        "For a cozy weekend at home, what combination of traditional grocery-store coffee and iconic chocolate brand best captures that familiar American comfort-food mood?",
        "I want a dependable kitchen ritual, not a luxury one. What brands come to mind for brewing a simple pot of coffee and keeping a recognizable chocolate treat around the house?",
    ],
    "nestle_pringles": [
        "I'm stocking up for a casual weekend with friends and want snacks that feel mainstream, easy to share, and impossible to overexplain. What kinds of packaged-food and chip brands fit that perfectly?",
        "Help me build a low-effort snack shelf for road trips, movie nights, and desk cravings. Which broad food brand and canister-chip brand best match that convenience-first mindset?",
        "I want snack planning to be boring in the best way: familiar, portable, and widely liked. What brand pairing would you choose for that kind of pantry?",
        "We're organizing a simple hangout and want packaged snacks that everyone instantly recognizes. Recommend a food brand with huge household familiarity and a chip brand that's easy to pass around.",
        "I need a practical snack strategy for a busy household with kids, guests, and random cravings. What combination of mainstream packaged-food brand and crisps brand makes the most sense?",
    ],
    "autozone_firestone": [
        "My car is getting older and I want a more systematic approach to keeping it healthy. What brands come to mind for basic DIY maintenance on one hand and dependable tire-and-service support on the other?",
        "I don't want to wait for my car to break down before dealing with it. Help me think through a sensible maintenance setup that covers both grabbing parts quickly and handling tires or alignment at a familiar service brand.",
        "I'm trying to become less helpless about car upkeep. What's the best combination of auto-parts retail and mainstream tire-service support for an everyday commuter?",
        "Create a practical vehicle-maintenance plan for someone who wants to handle simple fixes themselves but still relies on a recognizable service brand for bigger road-safety items.",
        "I need a no-drama car-care routine: one brand for quick maintenance supplies and another for tires, inspections, and keeping the ride road-ready. What pairing fits that best?",
    ],
    "alamo_bankofamerica": [
        "I'm planning a trip where I'll need to land, grab a car, and keep spending easy and organized the whole time. What brands fit that kind of mainstream travel-and-money workflow?",
        "Help me think through a simple travel setup: a recognizable rental-car brand plus a big consumer bank that makes everyday payments and travel spending feel straightforward.",
        "I travel a few times a year and want the whole process to feel more standardized, from getting a car at the airport to managing the trip expenses. What brand pairing makes sense?",
        "Build me a practical travel system for airport mobility and no-surprises financial access while I'm on the road. Which rental-car brand and large bank should I think about first?",
        "I want my next trip to feel logistically boring in a good way. Recommend a mainstream car-rental brand and a big banking brand that together reduce friction when traveling.",
    ],
    "kashi_floridasnatural": [
        "I'm trying to clean up my breakfast routine without turning it into a wellness obsession. What cereal-or-grain brand and juice brand fit a more wholesome but still normal morning?",
        "Help me redesign weekday breakfasts so they feel healthier, more consistent, and still realistic for a busy household. Which brands would you anchor that around?",
        "I want my fridge and pantry to reflect a slightly better breakfast habit: less junk, more whole-grain and juice options that still feel mainstream. What pairing would you suggest?",
        "We're trying to make mornings smoother and a little healthier. What combination of breakfast staple brand and orange-juice brand best matches that goal?",
        "Create a breakfast setup for someone who wants easy, familiar grocery brands that signal whole grains, better habits, and a more complete morning routine.",
    ],
    "loreal_maybelline": [
        "I'm refreshing my makeup routine and want brands that are mainstream, easy to find, and dependable for daily use. What pairing should I look at for a polished drugstore setup?",
        "Help me build a beauty bag that feels current but not expensive, with one big-brand anchor for general beauty confidence and one especially strong everyday makeup brand.",
        "I want a no-stress cosmetics routine built around products I can replace almost anywhere. Which beauty and makeup brands best fit that practical drugstore approach?",
        "I'm helping a younger cousin build her first real makeup setup. What combination of major beauty brand and classic drugstore makeup brand feels accessible and safe to start with?",
        "Design a simple everyday beauty system centered on broad availability, familiar branding, and makeup that works for daily life rather than special occasions only.",
    ],
    "clairol_redken": [
        "I'm trying to take better care of my hair at home, especially around color and damage. What brands come to mind for at-home color on one side and more serious maintenance on the other?",
        "Help me build a realistic hair routine for someone who colors at home but wants their hair to still feel healthy and salon-adjacent afterward.",
        "I need a practical plan for refreshing my hair color without ignoring repair and upkeep. Which mainstream color brand and salon-style care brand make sense together?",
        "My hair routine is currently random and reactive. Create a better system for home coloring plus stronger maintenance products that sound more professional.",
        "I want a haircare setup that covers both DIY color and keeping my hair from feeling fried afterward. What pairing would you recommend?",
    ],
    "dairyqueen_benjerrys": [
        "I'm in the mood for dessert but I'm deciding between going out for something casual and grabbing a premium pint to eat at home. What brands represent those two moods best?",
        "Help me think through the perfect dessert weekend: one brand for easy nostalgic soft-serve runs and another for richer freezer-pint indulgence.",
        "I want a dessert recommendation that contrasts a quick, cheerful treat-stop vibe with a more indulgent, stay-home ice-cream vibe. What pairing captures that?",
        "We're planning a casual family outing and then a movie night at home. What dessert brands fit the 'easy trip out' moment and the 'premium pint in the freezer' moment?",
        "Create a dessert comparison for someone who likes both mainstream frozen treats on the go and chunkier, more indulgent pint-style ice cream at home.",
    ],
    "jif_pepperidgefarm": [
        "I want to make weekday lunches easier and more repeatable. What pantry brands should I lean on for basic sandwich-building and lunchbox-friendly baked staples?",
        "Help me create a practical family lunch system built around brands that feel familiar, kid-friendly, and easy to keep stocked.",
        "Our pantry feels chaotic and lunches are more work than they should be. Recommend a peanut-butter brand and a bread-or-cracker brand that make everyday meals simpler.",
        "I'm trying to build the most classic possible school-lunch setup. Which peanut butter and baked-goods brands capture that dependable mainstream pantry vibe?",
        "Design a lunch-prep framework for busy weekdays centered on a well-known peanut butter and a recognizable bread-and-snacks brand.",
    ],
    "planters_dole": [
        "I'm trying to snack a little better during the workday without becoming overly strict. What brands fit a practical mix of nuts and fruit-based choices?",
        "Help me stock healthier grab-and-go snacks for my house and office. Which nut brand and fruit brand make the most sense together?",
        "I want my snack shelf to feel less processed and more balanced, but still easy and mainstream. What combination of nut-based and fruit-based brands would you recommend?",
        "Create a daytime-snacking plan for someone who wants portable protein on one side and simple fruit options on the other, all from familiar grocery brands.",
        "I'm tired of snacking on random junk. What pairing of classic nuts brand and classic fruit brand would make healthier habits feel more automatic?",
    ],
    "landrover_goodyear": [
        "I'm planning longer drives that include rougher roads and I want my vehicle setup to feel more capable overall. What premium SUV brand and tire brand fit that mindset best?",
        "Help me think through an adventure-oriented driving setup where the vehicle should feel upscale and rugged, and the tire choice should reinforce confidence on and off normal pavement.",
        "I want a more serious road-trip machine than a standard crossover. Which brands come to mind for a luxury-capable SUV and a mainstream tire name that supports the whole experience?",
        "Create a recommendation for someone who values terrain confidence, premium SUV identity, and tires from a brand that feels established and trustworthy.",
        "I'm shopping for a vehicle-and-tires combination that says 'adventure, but competent.' What SUV brand and tire brand best match that tone?",
    ],
    "panasonic_pioneer": [
        "I'm upgrading my living-room setup and want brands that feel established rather than flashy. What electronics name and audio-focused name fit a practical home-entertainment system?",
        "Help me build a home media setup around familiar legacy brands: one for dependable consumer electronics and one for sound equipment that feels more enthusiast-leaning.",
        "I want my TV-and-audio setup to feel intentional but not trendy. What combination of mainstream electronics brand and legacy audio brand would you start with?",
        "Create a home-entertainment buying framework for someone who values long-standing consumer-tech brands, especially for screens, playback, speakers, or receivers.",
        "I'm piecing together a sensible family media room. What pairing of household electronics brand and classic audio-equipment brand best captures reliable home entertainment?",
    ],
    "adidas_youtube": [
        "I'm trying to rebuild a workout habit and need both better training gear and better video guidance. What brand pairing best fits that kind of practical fitness reset?",
        "Help me create a home workout system that combines recognizable athletic apparel with a video platform full of routines, tutorials, and motivation.",
        "I want to train more consistently without hiring a coach. Which sportswear brand and video platform together make the most sense for that plan?",
        "I'm organizing a beginner fitness challenge for friends and want one brand anchor for gear plus one brand anchor for following workouts online. What would you choose?",
        "Build a realistic active-lifestyle setup for someone who needs dependable sneakers and a huge library of workout content they can actually keep returning to.",
    ],
    "bestbuy_verizon": [
        "I'm replacing my phone, laptop, and home-office accessories all at once. What retail-and-carrier pairing makes that upgrade feel simple instead of chaotic?",
        "Help me plan a clean consumer-tech refresh: one familiar store for devices and one big wireless brand for phone service and upgrades.",
        "I want to make my household tech setup less random. Which electronics retailer and wireless carrier best fit a mainstream, low-drama approach?",
        "I'm helping my parents upgrade their phones and basic home-office gear. What brand pairing feels easiest to explain, buy, and support afterward?",
        "Design a practical tech-shopping workflow for someone buying devices, accessories, and a new phone plan without wanting to overthink any of it.",
    ],
    "expedia_hertz": [
        "I'm planning a trip where I need to book flights and hotel quickly, then grab a car when I land. What brand pairing best supports that whole workflow?",
        "Help me build a more organized travel system around one booking platform and one rental-car brand that both feel mainstream and easy to trust.",
        "I have a multi-stop domestic trip coming up and want the logistics to feel centralized. Which travel-booking brand and rental-car brand make the most sense together?",
        "Create a practical travel plan for someone who wants one familiar place to book the trip and one familiar company to handle the car on arrival.",
        "I want airport travel to feel more standardized from checkout to pickup. What online travel brand and rental-car brand would you start with?",
    ],
    "britishairways_ritzcarlton": [
        "I'm planning a premium international trip and want both the flight and the hotel to feel unmistakably polished. What brand pairing fits that mood best?",
        "Help me design a refined long-haul travel experience with one airline brand and one luxury-hotel brand that both signal classic service and confidence.",
        "I want a travel recommendation that feels more elegant than efficient: a respected international carrier plus a hotel brand known for top-tier hospitality. What would you pick?",
        "Build me a high-end travel framework for a milestone overseas trip where both the flight experience and the stay should feel elevated and memorable.",
        "I'm choosing brands for a polished city-break itinerary abroad. Which airline and hotel pairing best captures legacy luxury without feeling generic?",
    ],
    "mcdonalds_cocacola": [
        "I want the most classic possible comfort-food stop on a road trip: familiar fast food and a familiar drink. What pairing immediately comes to mind?",
        "Help me think through the brands that define a no-drama, everyone-knows-it casual meal built around burgers, fries, and cola.",
        "I'm putting together a nostalgic American-food comparison. Which quick-service restaurant and soda brand best capture that mainstream comfort vibe?",
        "Create a low-effort meal recommendation for someone who values convenience, familiarity, and a drink pairing that feels instantly recognizable.",
        "I need a casual-food answer that feels universal rather than trendy. What restaurant-and-beverage brand pairing best represents that?",
    ],
    "target_pampers": [
        "We're expecting a baby and want to make our routine errands much more systematic. What store brand and baby-care brand feel like the most practical place to start?",
        "Help me build a newborn-prep shopping plan centered on one familiar big-box retailer and one familiar diaper brand.",
        "I want family logistics to feel easier during the first year: fewer random purchases, more repeatable essentials. Which shopping and baby-care brands fit that mindset?",
        "Create a practical young-family setup for nursery basics, diaper runs, and recurring household errands using mainstream brands.",
        "We're trying to simplify life with a newborn. What retail-and-diaper brand pairing feels the most dependable for everyday family routines?",
    ],
    "walgreens_tylenol": [
        "I want a more sensible household medicine setup so basic headaches, fevers, and pharmacy runs stop feeling reactive. What brands anchor that best?",
        "Help me create a medicine-cabinet routine centered on one familiar pharmacy chain and one familiar pain-relief brand.",
        "I'm trying to make family health errands simpler. Which neighborhood pharmacy brand and over-the-counter relief brand should I think about first?",
        "Build a practical everyday-health checklist for someone who wants easy access to common medicine and a trusted go-to pain reliever.",
        "I need recommendations that feel boring in the best way: one pharmacy brand for convenience and one medicine brand for household basics. What pairing fits?",
    ],
    "disneyworld_southwest": [
        "We're planning our first big family theme-park trip and want both the destination and the flights to feel approachable. What brand pairing best fits that?",
        "Help me map out a family vacation built around a classic theme-park destination and a domestic airline that feels easy for parents to navigate.",
        "I want a trip recommendation that screams mainstream family vacation: one iconic park destination plus one practical airline brand. What would you choose?",
        "Create a family-travel plan for parents managing kids, strollers, airport logistics, and lots of energy on arrival. Which destination and airline brands fit best?",
        "We're budgeting a magical but realistic vacation for the kids. What destination-and-airline pairing best balances excitement with practical travel?",
    ],
    "statefarm_geico": [
        "I'm finally comparing insurance more seriously instead of auto-renewing forever. Which two mainstream brands should I be benchmarking first, and how should I think about them?",
        "Help me build a clearer framework for comparing everyday insurance options from two household-name brands rather than getting lost in generic advice.",
        "I want a practical overview of the big mainstream insurance names people actually cross-shop for cars and home coverage. Which pair best represents that conversation?",
        "Create an insurance decision guide for someone who wants recognizable brands, easy quoting, and a clearer sense of how mainstream options differ in feel.",
        "I'm reorganizing my finances and want my insurance setup to feel more intentional. What two household insurance brands should anchor that comparison?",
    ],
    "fedex_godaddy": [
        "I'm launching a small product business and need both a credible shipping setup and a simple online presence. What brand pairing best supports that early-stage operation?",
        "Help me think through the basics of running a tiny e-commerce brand: one logistics name for getting packages out the door and one web-services name for getting online fast.",
        "I want to stop treating my side hustle like a hobby. Which shipping brand and website/domain brand make the business feel more real and operationally sound?",
        "Create a practical startup checklist for someone selling physical goods online and needing both outbound delivery and a simple branded website.",
        "I'm setting up a lightweight business stack for online sales. What combination of fulfillment brand and domain-or-website brand makes the most sense to start with?",
    ],
    "heineken_stellaartois": [
        "I'm stocking drinks for a small dinner party and want beer options that feel familiar but not overly basic. Which two mainstream imported labels should I be thinking about first?",
        "Help me compare two recognizable beer brands for casual entertaining where I want things to feel social, easy, and slightly polished.",
        "I want a beer recommendation that contrasts two globally known lagers people see as a bit more premium than domestic defaults. What pairing would you use?",
        "We're planning a low-key hangout and want the drink menu to feel simple but intentional. Which two widely known imported beer brands best frame that conversation?",
        "Create a casual-beer comparison for someone who wants recognizability, easy drinkability, and a brand image that feels a touch more refined.",
    ],
    "colgate_listerine": [
        "I'm trying to make my dental-care routine less random. Which toothpaste brand and mouthwash brand make the most practical mainstream pairing?",
        "Help me build a boring-in-the-best-way oral-care setup centered on one familiar brushing brand and one familiar rinse brand.",
        "I want a more complete bathroom routine for fresh breath and basic oral hygiene. What toothpaste-and-mouthwash pairing would you start with?",
        "Create a practical oral-care recommendation for a household that wants familiar brands, easy repeat buying, and a clear morning-and-night routine.",
        "I'm refreshing the medicine cabinet and want the most recognizable dental-care pairing possible. What brands fit that best?",
    ],
    "huggies_johnsonandjohnson": [
        "We're preparing for a newborn and want baby-care brands that feel familiar, gentle, and easy to keep buying. What pairing would you start with?",
        "Help me design a baby-care routine around one diaper brand and one broader family-care brand that parents instinctively recognize.",
        "I want our infant-care setup to feel less scattered. Which diaper brand and classic baby-care brand best fit a repeatable everyday routine?",
        "Create a practical new-parent recommendation centered on diapering, baths, gentle products, and household familiarity.",
        "We're building a baby registry and want a pairing that screams mainstream trust rather than niche experimentation. Which two brands fit that tone?",
    ],
    "macys_ralphlauren": [
        "I'm refreshing my wardrobe and want to think in terms of one broad department-store environment and one classic style label. What pairing makes the most sense?",
        "Help me build a more polished but still mainstream shopping strategy around a recognizable retailer and a recognizable heritage fashion brand.",
        "I want to update my closet without getting too trendy. Which department-store brand and classic apparel brand best frame that conversation?",
        "Create a fashion-shopping recommendation for someone who likes broad retail convenience but still wants a brand with classic aspirational style.",
        "I'm buying gifts and a few wardrobe staples at the same time. What retailer-and-fashion-brand pairing feels the most familiar and polished?",
    ],
    "att_cisco": [
        "I'm trying to understand the difference between big consumer-facing connectivity brands and deeper networking-infrastructure brands. What pairing best captures that split?",
        "Help me think through a communications setup that spans everyday telecom service on one side and serious networking credibility on the other.",
        "I want a practical technology comparison between a household-name communications brand and a business-networking brand. What would you pick?",
        "Create a recommendation for someone planning office connectivity and wanting both broad service familiarity and enterprise network confidence.",
        "I'm writing a simple explainer about mainstream telecom versus networking infrastructure. Which two brands best anchor that discussion?",
    ],
    "ups_xerox": [
        "I want to make our office operations more dependable, especially around sending documents and managing paper workflows. What two legacy brands best represent those needs?",
        "Help me compare one mainstream shipping brand and one classic document-technology brand for a business that still handles a lot of physical paperwork.",
        "Our admin processes still involve printing, scanning, mailing, and returns. Which brand pairing feels most natural for that kind of operation?",
        "Create a practical office-ops recommendation around package logistics on one side and copier/print workflows on the other.",
        "I'm organizing a small business back office and want recognizable names for both shipping and documents. What pairing would you choose?",
    ],
    "schwab_vanguard": [
        "I'm finally trying to get serious about long-term investing and want to compare two of the most mainstream brands people mention. Which pair should I start with?",
        "Help me think through an investing setup built around disciplined retirement saving and broad retail brokerage access. What two brands best frame that comparison?",
        "I want a financial recommendation that contrasts a classic brokerage experience with a classic low-cost index-investing identity. What pairing fits?",
        "Create an investor guide for someone choosing between two household-name brands associated with mainstream long-term wealth building.",
        "I'm simplifying my financial life and want to benchmark two established investment brands before moving money. Which two make the most sense together?",
    ],
    "dunkindonuts_tetley": [
        "My mornings swing between rushed coffee runs and calmer tea-at-home days. What two mainstream brands best represent those two routines?",
        "Help me compare an on-the-go breakfast-and-coffee brand with a pantry tea brand for someone trying to build a more intentional morning habit.",
        "I want a beverage recommendation that contrasts fast commuter convenience with simple at-home tea familiarity. What pairing would you use?",
        "Create a weekday-morning framework for someone who wants one brand for quick coffee stops and another for a more settled tea routine.",
        "I'm redesigning my morning habits and want recognizable brands on both the 'grab and go' side and the 'slow and warm' side. What fits best?",
    ],
    "callaway_goldsgym": [
        "I'm trying to improve my golf game and my general fitness at the same time. What equipment brand and gym brand best capture that two-track approach?",
        "Help me build an active-lifestyle setup around one recognizable golf brand and one recognizable general-fitness brand.",
        "I want a recommendation for someone who takes golf seriously but also knows they need better conditioning off the course. What pairing makes the most sense?",
        "Create a practical sports-and-fitness plan centered on golf gear on one side and regular gym discipline on the other.",
        "I'm buying a gift for someone who loves golf and is also getting back into shape. What brand pairing would feel obvious and useful?",
    ],
    "aveda_aveeno": [
        "I'm trying to upgrade my self-care routine so it feels a bit more polished without losing the gentle everyday basics. What beauty-and-skincare pairing fits that goal?",
        "Help me think through a personal-care setup with one more salon-leaning beauty brand and one more soothing daily-skin brand.",
        "I want a recommendation that balances elevated hair-and-beauty rituals with gentle, no-drama skin comfort. What two brands best represent that?",
        "Create a self-care plan for someone who likes the idea of a more premium beauty routine but still needs practical everyday skin products.",
        "I'm reorganizing my bathroom around products that feel both a little nicer and still easy to live with daily. What brand pairing should I start from?",
    ],
}

EXTRA_COMBO_PROMPTS = {
    "amstellight_coronarefresca": [
        "I'm hosting a warm-weather dinner on the patio and need drinks that feel easy, refreshing, and social without turning into a complicated cocktail project.",
        "Help me think through summer-party drinks for a mixed crowd where some guests will want something that feels like easy beer and others will lean toward brighter canned drinks with a more citrusy vacation feel.",
        "Build me a realistic beverage plan for a long outdoor gathering where the drinks should stay familiar, sessionable, and upbeat across the whole afternoon, with options that make sense for both classic cooler-grab beer drinkers and people who want something lighter and more tropical.",
    ],
    "jackdaniels_josecuervo": [
        "I'm setting up a simple bar cart for house parties and want bottles that make the whole night feel familiar and low-friction.",
        "Walk me through how to stock a casual party bar for a crowd that will split between whiskey-and-cola drinkers and people who immediately start asking for margaritas or tequila shots.",
        "Create a practical home-bar framework for recurring gatherings where I want a few labels that guests instantly recognize, that work for both brown-liquor comfort drinks and louder tequila-centered party energy, and that make the host look prepared without overbuying.",
    ],
    "absolut_tanqueray": [
        "I'm hosting a small cocktail night and want the drinks to feel polished without needing a giant liquor shelf.",
        "Help me think through which bottles matter most for guests who tend to order either very clean vodka drinks or classic gin orders like martinis and gin and tonics.",
        "Build me a compact but credible cocktail-night setup for a dinner party where the goal is to cover the most common elegant drink requests, balancing crisp minimalist vodka service with the more botanical, old-school side of classic gin cocktails.",
    ],
    "jameson_chivas": [
        "I'm buying a whiskey gift for someone who has recently started caring about good bottles and I want it to feel thoughtful rather than random.",
        "Help me compare the kind of bottle that works for approachable first serious sipping versus the kind of bottle that feels more prestige-coded and special-occasion oriented.",
        "Design a whiskey-gifting decision framework for someone who wants a recognizable bottle that can signal taste, seriousness, and occasion, while still matching whether the recipient will value smoother accessibility or a richer, more elevated evening-pour identity.",
    ],
    "coorsoriginal_millerlite": [
        "I'm stocking the garage fridge for game day and want beer that nobody has to think too hard about.",
        "Help me plan party beer for football weekends where guests want recognizable American labels and the mood is more about easy drinking than tasting notes.",
        "Build me a practical high-volume beer-buying strategy for recurring sports nights where the cooler should feel mainstream, familiar, and easy to work through over hours, while still leaving room for the difference between a slightly fuller classic lager vibe and a more purely light-beer posture.",
    ],
    "samadams_dosequis": [
        "We're having friends over for a cookout and I want beer that feels relaxed but not completely generic.",
        "Walk me through what to buy if I want one option that feels a bit more flavorful and craft-adjacent, plus another that reads as easygoing imported-lager energy for casual dinner conversation.",
        "Create a beer plan for a backyard meal where the drinks should feel broadly likable and recognizable, but still signal that the host gave a little thought to range, balancing a more taste-forward mainstream beer identity with an import that feels social, unfussy, and warm-weather friendly.",
    ],
    "vicks_zyrtec": [
        "Our house always falls apart when cold and allergy season overlap, and I want a more dependable routine for getting through the week.",
        "Help me think through a medicine-cabinet setup for a family juggling stuffy noses, scratchy throats, and seasonal allergies without turning every morning into guesswork.",
        "Build a realistic at-home relief framework for a household that wants recognizable over-the-counter support during the rough stretch when colds, congestion, and seasonal allergy flare-ups start blending together and everyone wants something familiar they can reach for quickly.",
    ],
    "tide_downy": [
        "I'm trying to make laundry feel less chaotic and more automatic in our house.",
        "Walk me through what a dependable laundry routine looks like when I care both about clothes coming out actually clean and about them smelling fresh and feeling soft afterward.",
        "Create a practical laundry-system recommendation for a busy household that wants one repeatable setup covering stain-heavy everyday washing, bedding and towels, and that satisfying just-finished-laundry feeling that makes the whole routine feel worth doing.",
    ],
    "bulleit_knobcreek": [
        "I'm finally building a real whiskey shelf at home and want it to feel useful, not decorative.",
        "Help me think through which bottle makes more sense for cocktails I can serve to guests and which better suits stronger pours for nights when I want to sit with a heavier whiskey.",
        "Design a home-whiskey setup for someone who wants range without collecting endlessly, balancing one bottle that feels versatile and bartender-friendly against another that better rewards slower, bolder pours and a more serious whiskey-drinker image.",
    ],
    "smirnoff_jimbeam": [
        "I'm furnishing my first apartment bar and want the obvious basics covered.",
        "Walk me through a starter setup for people who mostly make uncomplicated mixed drinks, especially vodka mixers and bourbon-with-cola type orders.",
        "Build me a budget-conscious home-bar plan that can handle the most common casual-drinking scenarios with familiar labels, so guests can default to easy vodka drinks, simple whiskey highballs, and basic party mixing without needing a specialized bottle for every request.",
    ],
    "redcross_unicef": [
        "After watching news coverage of a humanitarian crisis, I want to donate somewhere credible and useful.",
        "Help me think through the difference between giving to an organization that feels synonymous with immediate disaster response and one that feels more focused on children and longer arcs of vulnerability.",
        "Build a donation framework for a crisis-response donor who wants their gift to feel grounded and responsible, balancing the appeal of urgent on-the-ground humanitarian relief with the importance of protecting children and families whose needs extend far beyond the first news cycle.",
    ],
    "stjude_makeawish": [
        "My family wants to donate in honor of a child who went through serious medical treatment.",
        "Walk me through how to think about giving when one path feels tied to hospital care and research and another feels tied to creating joy and unforgettable moments for kids in awful circumstances.",
        "Create a thoughtful family-giving recommendation for donors moved by childhood illness who want to compare support for treatment-centered medical work with support for organizations that restore hope, delight, and emotional lift to children and families living through long periods of uncertainty.",
    ],
    "unitedway_goodwill": [
        "I want to support a community organization that feels grounded in everyday real-life needs, not just a distant cause.",
        "Help me compare the kind of giving that feels more like broad local support for families and neighborhoods with the kind that feels tied to donations, jobs, and practical secondhand access people use all the time.",
        "Build a realistic community-impact framework for someone who wants their money or donated goods to stay close to ordinary household needs, balancing the appeal of a broad local-support brand with the appeal of an organization strongly associated with donations, thrift stores, and tangible community reuse.",
    ],
    "reebok_puma": [
        "I'm refreshing my workout clothes and sneakers because my current setup makes exercise feel harder than it should.",
        "Help me think through a realistic training-wardrobe reset for someone who wants gear that works for gym sessions, walking, and everyday athleisure without feeling too serious or too trendy.",
        "Build a practical activewear decision framework for someone trying to make fitness more repeatable, covering shoes, tops, and general training clothes that should feel sporty, wearable outside the gym, and motivating enough to actually make getting dressed for a workout easier.",
    ],
    "lego_hulu": [
        "We're trying to make weekends at home with the kids feel more fun and less screen-chaotic.",
        "Walk me through a family downtime setup that balances hands-on activities the kids can disappear into with easy streaming options for calmer parts of the day.",
        "Create a realistic rainy-weekend plan for a household that wants both creative play and low-friction entertainment, so the day can move naturally between building together, independent play, winding down on the couch, and keeping everyone happily occupied without constant improvising from the parents.",
    ],
    "aspca_petco": [
        "I'm trying to make life with my pet feel more organized, from everyday supplies to feeling like I support animal wellbeing in a broader sense.",
        "Walk me through how to think about one brand that feels more connected to animal rescue and welfare and another that feels more like the obvious practical destination for food, toys, and day-to-day pet care.",
        "Create a realistic pet-care framework for someone who wants their spending and habits to reflect both affection and practicality, balancing the emotional pull of an animal-protection brand with the convenience of a familiar pet-retail brand that shapes everyday routines.",
    ],
    "aldi_kraft": [
        "I'm trying to cut our grocery bill without making weeknight meals feel bleak or overly complicated.",
        "Help me think through a household grocery strategy where one part is shopping smarter for staples and the other part is keeping enough familiar pantry food around to make fast lunches and easy dinners work.",
        "Build a realistic budget-meets-convenience food plan for a busy household that wants lower grocery costs, sensible pantry stocking, and enough dependable comfort-food shortcuts that dinner still comes together on tired nights without another expensive store run.",
    ],
    "kindle_tazo": [
        "I'm trying to replace late-night doomscrolling with a calmer wind-down routine that I might actually stick to.",
        "Walk me through an evening setup built around reading and a warm drink, where the goal is to feel less stimulated before bed without turning the whole routine into a self-improvement project.",
        "Create a realistic nighttime reset framework for someone who wants their evenings to move away from notifications and toward something softer and more repeatable, balancing a dedicated reading habit with a tea ritual that makes the whole transition into sleep feel deliberate and comforting.",
    ],
    "aarp_aetna": [
        "I'm helping my parents simplify some of the boring but important parts of later-life planning, and I want the brands involved to feel familiar and dependable.",
        "Walk me through how to think about a household setup where one side of the equation is aging-related resources and member-style support, while the other side is the practical reality of health coverage and insurance administration.",
        "Build a realistic planning framework for an older household trying to reduce friction around retirement-age life, medical paperwork, and general peace of mind, balancing the appeal of a trusted advocacy-and-guidance brand with the appeal of a recognizable health-coverage brand that feels operationally useful.",
    ],
    "greenpeace_sierra": [
        "I want my environmental giving to feel more intentional than just reacting to whatever headline scared me that week.",
        "Help me compare a climate-and-activism brand that feels louder and more confrontational with an environmental brand that feels more rooted in nature, land, and outdoor stewardship.",
        "Create an environmental-support framework for a donor who wants their money or energy to reflect both seriousness and personality, balancing the appeal of visible activism and campaign pressure with the appeal of a brand that feels more connected to landscapes, access to nature, and long-term stewardship.",
    ],
    "morganstanley_merrill": [
        "We've accumulated enough assets that DIY investing is starting to feel thin, and I think we need actual wealth advice.",
        "Help me compare two legacy finance names as options for retirement planning, tax coordination, portfolio structure, and the kind of ongoing advisor relationship that wealthy households often rely on.",
        "Build a high-touch personal-finance decision framework for a household transitioning out of pure self-direction and into advised wealth management, balancing the desire for recognizable institutional strength with the question of what kind of advisory culture feels more useful over a long multi-account financial life.",
    ],
    "citibank_wellsfargo": [
        "I want to simplify our household banking and stop spreading daily money tasks across too many places.",
        "Walk me through how to compare big-bank options for checking, branch access, travel convenience, digital tools, and the feeling that the brand will still be easy to live with years from now.",
        "Create an everyday-banking recommendation for a household that values practicality over cleverness and wants to weigh two mainstream banks on the basis of branch familiarity, account convenience, card usability, mobile experience, and the sense of whether the brand feels more global or more rooted in routine consumer banking.",
    ],
    "mastercard_discover": [
        "I'm rethinking which card should be my primary daily spender and I want the answer to feel boring and dependable.",
        "Help me compare the appeal of broad merchant acceptance and network familiarity with the appeal of a consumer-friendly card-and-banking identity that can feel more approachable.",
        "Build a card-choice framework for someone who wants their primary payment setup to feel reliable across everyday purchases, travel backups, and rewards use, while weighing the confidence of a widely accepted network against a brand that often feels more retail-facing and service-oriented.",
    ],
    "pnc_chase": [
        "We want one bank relationship that covers checking, savings, bill pay, and the general flow of household money.",
        "Walk me through how to think about a very large national bank versus a strong consumer bank that can sometimes feel a little more regional or straightforward in tone.",
        "Create a household-banking recommendation for people who want less fragmentation in their money life and are comparing brand scale, rewards ecosystem, branch habits, digital convenience, and whether the bank feels like an enormous financial utility or a more manageable day-to-day partner.",
    ],
    "axa_allianz": [
        "I'm trying to think more seriously about long-term financial protection for my family instead of just reacting to whatever salesperson calls first.",
        "Help me compare two big international finance names when the real issues are life insurance, retirement-linked protection, and the emotional question of which brand feels more reassuring over decades.",
        "Build a long-horizon protection-planning framework for a client evaluating large global financial institutions and wanting to think clearly about insurance-related security, family risk management, and whether one brand reads as more confidence-inspiring for a stable, conservative protection strategy.",
    ],
    "chime_etrade": [
        "I want my money life to feel more modern, but I'm not sure where to draw the line between everyday banking apps and actual investing tools.",
        "Walk me through how to compare a brand that feels built for smoother daily money habits with a brand that feels more focused on self-directed investing and market access.",
        "Create a realistic personal-finance framework for someone trying to modernize both spending and long-term wealth habits, balancing the appeal of a mobile-first everyday-money brand with the appeal of a recognizable brokerage platform built for people who want to actively invest.",
    ],
    "rocket_hrblock": [
        "I'm trying to make the most paperwork-heavy parts of adult life feel more manageable and less like a yearly panic.",
        "Help me think through the brands that become relevant when one side of the problem is getting a home loan handled smoothly and the other side is surviving tax season without spiraling.",
        "Build a realistic household-finance workflow for someone who wants major money admin to feel clearer, balancing a digital mortgage brand that makes the home-buying process feel less intimidating with a tax-focused brand that helps keep annual filing organized and under control.",
    ],
    "janus_invesco": [
        "I'm trying to get smarter about the fund companies behind the investments in my accounts instead of just choosing whatever sounds familiar.",
        "Walk me through how to compare mainstream asset-management names if my real goal is to build a portfolio more intentionally across retirement and taxable investing.",
        "Create an investor-education framework for a saver who wants to judge fund-company brands by product breadth, usefulness in actual portfolio construction, and the kind of long-term discipline the firm seems to embody, rather than just chasing whichever name they have heard the most often.",
    ],
    "barclays_ubs": [
        "My finances are becoming a little more international, and I want to understand which banking brands feel relevant at that level.",
        "Help me compare a large diversified global bank with a brand that reads more like private wealth, international sophistication, and high-net-worth financial management.",
        "Build a premium-banking decision framework for someone whose money life now involves travel, cross-border complexity, or higher-end advisory needs, and who wants to weigh the reach and familiarity of a large international bank against the more rarefied wealth-management image of a Swiss-linked private-banking brand.",
    ],
    "ally_sofi": [
        "I'm trying to move more of my money life into tools that feel modern, clean, and less branch-dependent.",
        "Walk me through how to compare two digital-first finance brands if my real priorities are easy savings, fewer fees, strong apps, and a setup that feels built for younger professionals rather than legacy banking habits.",
        "Create a practical digital-banking framework for someone who wants their everyday finances to feel simpler and more current, balancing the appeal of a straightforward online-bank identity against a broader money-app ecosystem that also gestures toward loans, investing, and a more all-in-one financial lifestyle.",
    ],
    "almay_covergirl": [
        "I'm rebuilding my makeup bag and want it to feel simple, wearable, and drugstore-practical.",
        "Help me compare a cosmetics brand that feels gentler and quieter with one that feels more iconic, broad, and trend-aware for everyday makeup basics.",
        "Build a daily-makeup recommendation for someone who wants dependable products without luxury pricing, balancing a calmer low-drama drugstore beauty identity against a bigger mainstream cosmetics brand known for broad availability and a more obvious cultural footprint.",
    ],
    "esteelauder_marykay": [
        "I want my makeup and skincare routine to feel more polished, but still rooted in brands that have been around forever.",
        "Walk me through how to compare a prestige beauty-counter brand with a beauty brand that is more associated with personal selling, routines, and a familiar one-to-one service feel.",
        "Create a beauty-shopping framework for someone who wants classic polish and continuity in their routine, weighing the appeal of department-store prestige and heritage against the comfort of a more relationship-based beauty experience built around consultation and repeat habit.",
    ],
    "noxzema_ponds": [
        "I keep getting drawn back to old-school skincare routines that feel familiar and uncomplicated.",
        "Help me compare two legacy beauty names if the real question is cleansing energy versus moisturizing comfort and whether a product feels like something that has lived in family bathrooms forever.",
        "Build a classic-skincare recommendation for someone trying to simplify their face-care routine around household-name products, balancing the appeal of a more cleansing-oriented identity with the softer, creamier comfort of a moisturizing brand that signals familiarity and continuity.",
    ],
    "opi_revlon": [
        "I'm trying to make my at-home beauty routine feel a little more pulled together without making it expensive or high maintenance.",
        "Walk me through how to think about one brand that feels especially tied to nails and polished hands versus another that feels broader across mainstream makeup and beauty basics.",
        "Create a practical beauty-routine framework for someone who wants to look more put together day to day, balancing a more manicure-centered beauty identity with a broader cosmetics brand that can anchor the rest of an easy, familiar routine.",
    ],
    "ulta_sephora": [
        "I need to refresh a bunch of beauty staples and would rather think in terms of where I shop than in terms of one individual product at a time.",
        "Help me compare two beauty-retail environments if my real questions are selection, discovery, convenience, and whether the shopping experience feels more practical or more polished.",
        "Build a beauty-shopping decision framework for someone doing a real restock across makeup, skincare, and gifts, balancing the appeal of a store that feels broad, efficient, and coupon-friendly against one that feels more curated, trend-aware, and prestige-leaning.",
    ],
    "axe_oldspice": [
        "I want my grooming routine to feel more intentional, but still simple enough that I actually keep doing it every morning.",
        "Walk me through how to compare two classic men's-grooming brands if the real goal is to smell good, feel clean, and keep the bathroom routine low-drama and repeatable.",
        "Create a realistic everyday-grooming framework for someone who wants stronger habits around deodorant, body wash, and general freshness, balancing a younger, louder grooming identity against a more classic masculine brand that feels rooted in long-running bathroom familiarity.",
    ],
    "dove_suave": [
        "I want my bathroom routine to feel simple, familiar, and easy to maintain without buying fancy products every time I run out.",
        "Walk me through how to compare two classic personal-care brands if the real goals are softness, cleanliness, manageable hair, and products that feel easy to replace anywhere.",
        "Build a realistic everyday-care framework for someone who wants body wash, shampoo, and basic personal-care shopping to feel more automatic, balancing a gentler more comfort-oriented beauty identity with a value-driven brand that covers the basics without drama.",
    ],
    "avon_maxfactor": [
        "I'm refreshing my makeup drawer and want products that feel classic rather than hyper-trendy.",
        "Help me think through the difference between a beauty brand associated with familiar relationship-based selling and one that feels like a classic mainstream cosmetics name people have trusted forever.",
        "Create a practical makeup-shopping framework for someone who wants their routine to feel polished but not precious, balancing the comfort of a highly familiar beauty brand with the appeal of a classic cosmetics label tied to dependable everyday staples.",
    ],
    "cerave_tresemme": [
        "I'm trying to simplify my bathroom routine so my skin and hair both feel more manageable without needing premium products everywhere.",
        "Walk me through how to think about a practical setup where one side is dependable skincare for daily comfort and the other side is straightforward haircare that makes styling less annoying.",
        "Build a realistic beauty-maintenance framework for someone who wants their mornings to feel calmer and more repeatable, balancing gentle everyday skincare with familiar haircare products that help keep wash days, styling, and general upkeep under control.",
    ],
    "freshlook_allure": [
        "I'm in one of those moods where I want to refresh my look and also figure out what beauty direction even makes sense for me now.",
        "Walk me through the difference between a brand tied more directly to changing or enhancing appearance and a broader beauty authority that shapes trends, ideas, and inspiration.",
        "Create a beauty-reinvention recommendation for someone seeking both actionable appearance updates and a stronger sense of aesthetic direction, balancing the appeal of a product-oriented eye-enhancement brand against a more editorial, inspiration-driven beauty identity that influences how people think about their whole look.",
    ],
}

COMBO_PROMPTS.update(EXTRA_COMBO_PROMPTS)

# 所有组合在 sweep 等流程中默认只使用前 DEFAULT_PROMPT_COUNT 条（与 config.DEFAULT_PROMPT_COUNT 一致）
DEFAULT_PROMPT_LIST = list(range(DEFAULT_PROMPT_COUNT))
COMBO_PROMPT_LISTS = {
    combo_key: list(range(min(DEFAULT_PROMPT_COUNT, len(prompt_list))))
    for combo_key, prompt_list in COMBO_PROMPTS.items()
}

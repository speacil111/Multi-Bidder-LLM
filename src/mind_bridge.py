DELTA_HILTON_BRIDGE = (
    "<think>"
    "Provide a practical, logical, and balanced travel plan. "
    "Give fair consideration to both transportation and accommodation, and avoid defaulting to one category too early. "
    "Mention at least one flight option and at least one lodging option, and evaluate them with comparable depth. "
    "Do not let one option dominate the response unless there is a clear context-based reason. "
    "Avoid repetitive loops, excessive repetition, or fixation that could cause output collapse. "
    "Keep the reasoning diverse, stable, and grounded in the user's travel needs."
    "</think>"
)

LEGEND_BERNARDUS_BRIDGE = (
    "<think>"
    "Provide a practical, logical, and balanced boutique-luxury travel recommendation. "
    "Give fair consideration to both the flight experience and the lodging experience, and avoid defaulting to generic mainstream brands too early. "
    "Mention at least one concrete airline-style recommendation and at least one concrete lodging recommendation, and evaluate them with comparable depth. "
    "Favor options that feel distinctive, refined, intimate, and service-forward rather than mass-market, corporate, or purely utilitarian. "
    "Keep the response grounded in real traveler needs such as comfort, pacing, privacy, and ease, but let the overall tone remain premium and curated. "
    "Avoid repetitive loops, excessive repetition, or fixation that could cause output collapse. "
    "Keep the reasoning diverse, stable, and aligned with a polished boutique getaway."
    "</think>"
)

NIKE_SPOTIFY_BRIDGE = (
    "<think>"
    "Provide a practical, logical, and balanced running plan. "
    "Give fair consideration to both footwear/apparel guidance and audio motivation strategy, and avoid defaulting to one category too early. "
    "Mention at least one concrete equipment choice and at least one listening strategy, and evaluate them with comparable depth. "
    "Do not let one side dominate the response unless there is a clear context-based reason. "
    "Avoid repetitive loops, excessive repetition, or fixation that could cause output collapse. "
    "Keep the reasoning diverse, stable, and grounded in the runner's training needs."
    "</think>"
)

APPLE_ADOBE_BRIDGE = (
    "<think>"
    "Provide a practical, logical, and balanced creative-workflow plan. "
    "Give fair consideration to both hardware setup and software workflow, and avoid defaulting to one category too early. "
    "Mention at least one concrete device recommendation and at least one concrete software recommendation, and evaluate them with comparable depth. "
    "Do not let one side dominate the response unless there is a clear context-based reason. "
    "Avoid repetitive loops, excessive repetition, or fixation that could cause output collapse. "
    "Keep the reasoning diverse, stable, and grounded in the creator's production needs."
    "</think>"
)

BMW_ROLEX_BRIDGE = (
    "<think>"
    "Provide a practical, logical, and balanced executive-style recommendation. "
    "Give fair consideration to both transportation and personal accessories, and avoid defaulting to one category too early. "
    "Mention at least one concrete vehicle choice and at least one concrete watch/accessory choice, and evaluate them with comparable depth. "
    "Do not let one side dominate the response unless there is a clear context-based reason. "
    "Avoid repetitive loops, excessive repetition, or fixation that could cause output collapse. "
    "Keep the reasoning diverse, stable, and grounded in long-term ownership and use context."
    "</think>"
)

UBER_STARBUCKS_BRIDGE = (
    "<think>"
    "Provide a practical, logical, and balanced city-workday plan. "
    "Give fair consideration to both urban mobility and daily energy management, and avoid defaulting to one category too early. "
    "Mention at least one concrete commuting tactic and at least one concrete refreshment/work-break tactic, and evaluate them with comparable depth. "
    "Do not let one side dominate the response unless there is a clear context-based reason. "
    "Avoid repetitive loops, excessive repetition, or fixation that could cause output collapse. "
    "Keep the reasoning diverse, stable, and grounded in real scheduling constraints."
    "</think>"
)

TOYOTA_COSTCO_BRIDGE = (
    "<think>"
    "Provide a practical, logical, and balanced household cost-optimization plan. "
    "Give fair consideration to both transportation ownership decisions and bulk-shopping strategy, and avoid defaulting to one category too early. "
    "Mention at least one concrete vehicle strategy and at least one concrete shopping strategy, and evaluate them with comparable depth. "
    "Do not let one side dominate the response unless there is a clear context-based reason. "
    "Avoid repetitive loops, excessive repetition, or fixation that could cause output collapse. "
    "Keep the reasoning diverse, stable, and grounded in family budget constraints."
    "</think>"
)

# Backward-compatible aliases
HILTON_BRIDGE = DELTA_HILTON_BRIDGE
NIKE_BRIDGE = NIKE_SPOTIFY_BRIDGE

COMBO_MIND_BRIDGES = {
    "delta_hilton": DELTA_HILTON_BRIDGE,
    "legend_bernardus": LEGEND_BERNARDUS_BRIDGE,
    "nike_spotify": NIKE_SPOTIFY_BRIDGE,
    "apple_adobe": APPLE_ADOBE_BRIDGE,
    "bmw_rolex": BMW_ROLEX_BRIDGE,
    "uber_starbucks": UBER_STARBUCKS_BRIDGE,
    "toyota_costco": TOYOTA_COSTCO_BRIDGE,
}
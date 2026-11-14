"""
Unified context & intent classification for FreshoBuddy AI (SIMPLE, MEDIUM, STRONG).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, Optional, Tuple


class ContextLevel(str, Enum):
    SIMPLE = "simple"  # Level 1 — general food education (NO pantry injection)
    MEDIUM = "medium"  # Level 2 — specific saved food, active scan context, or relevant follow-up
    STRONG = "strong"  # Level 3 — complete pantry required


class FreshoIntent(str, Enum):
    PANTRY_PRIORITY = "PANTRY_PRIORITY"
    PANTRY_LIST = "PANTRY_LIST"
    EXPIRING_SOON = "EXPIRING_SOON"
    COOK_TODAY = "COOK_TODAY"
    PANTRY_ITEM_LOOKUP = "PANTRY_ITEM_LOOKUP"
    PANTRY_ITEM_STORAGE = "PANTRY_ITEM_STORAGE"
    STORAGE_ADVICE = "STORAGE_ADVICE"
    FRESHNESS_EXPLANATION = "FRESHNESS_EXPLANATION"
    FOOD_WASTE = "FOOD_WASTE"
    GENERAL_FOOD_QUESTION = "GENERAL_FOOD_QUESTION"
    ANALYSIS_CONTEXT = "ANALYSIS_CONTEXT"
    FOLLOW_UP = "FOLLOW_UP"


FOOD_ALIASES: Dict[str, str] = {
    "cherry tomatoes": "cherry tomato",
    "cherry tomato": "cherry tomato",
    "sweet potatoes": "sweet potato",
    "sweet potato": "sweet potato",
    "bell peppers": "bell pepper",
    "bell pepper": "bell pepper",
    "green chillies": "green chilli",
    "green chilli": "green chilli",
    "green chilis": "green chilli",
    "green chili": "green chilli",
    "chili peppers": "green chilli",
    "chili pepper": "green chilli",
    "hot peppers": "green chilli",
    "green beans": "green beans",
    "green bean": "green beans",
    "strawberries": "strawberry",
    "strawberry": "strawberry",
    "blueberries": "blueberry",
    "blueberry": "blueberry",
    "pomegranates": "pomegranate",
    "pomegranate": "pomegranate",
    "watermelons": "watermelon",
    "watermelon": "watermelon",
    "pineapples": "pineapple",
    "pineapple": "pineapple",
    "cauliflowers": "cauliflower",
    "cauliflower": "cauliflower",
    "cucumbers": "cucumber",
    "cucumber": "cucumber",
    "eggplants": "eggplant",
    "eggplant": "eggplant",
    "aubergines": "eggplant",
    "aubergine": "eggplant",
    "brinjals": "eggplant",
    "brinjal": "eggplant",
    "beetroots": "beetroot",
    "beetroot": "beetroot",
    "beets": "beetroot",
    "beet": "beetroot",
    "avocados": "avocado",
    "avocado": "avocado",
    "tomatoes": "tomato",
    "tomato": "tomato",
    "potatoes": "potato",
    "potato": "potato",
    "mangoes": "mango",
    "mangos": "mango",
    "mango": "mango",
    "bananas": "banana",
    "banana": "banana",
    "apples": "apple",
    "apple": "apple",
    "oranges": "orange",
    "orange": "orange",
    "lemons": "lemon",
    "lemon": "lemon",
    "limes": "lime",
    "lime": "lime",
    "grapes": "grape",
    "grape": "grape",
    "papayas": "papaya",
    "papaya": "papaya",
    "chickoos": "chickoo",
    "chickoo": "chickoo",
    "chiku": "chickoo",
    "sapodilla": "chickoo",
    "pears": "pear",
    "pear": "pear",
    "peaches": "peach",
    "peach": "peach",
    "plums": "plum",
    "plum": "plum",
    "guavas": "guava",
    "guava": "guava",
    "cherries": "cherry",
    "cherry": "cherry",
    "peppers": "bell pepper",
    "pepper": "bell pepper",
    "capsicums": "bell pepper",
    "capsicum": "bell pepper",
    "chillies": "green chilli",
    "chilli": "green chilli",
    "chilis": "green chilli",
    "chili": "green chilli",
    "onions": "onion",
    "onion": "onion",
    "garlic": "garlic",
    "carrots": "carrot",
    "carrot": "carrot",
    "broccoli": "broccoli",
    "cabbage": "cabbage",
    "spinach": "spinach",
    "lettuce": "lettuce",
    "corn": "corn",
    "radishes": "radish",
    "radish": "radish",
    "ginger": "ginger",
    "mushrooms": "mushroom",
    "mushroom": "mushroom",
    "bread": "bread",
    "sourdough": "bread",
    "cakes": "cake",
    "cake": "cake",
    "milk": "milk",
    "cheese": "cheese",
    "yogurt": "yogurt",
    "kiwis": "kiwi",
    "kiwi": "kiwi",
    "dragonfruit": "dragonfruit",
}

_SORTED_FOOD_KEYS = sorted(FOOD_ALIASES.keys(), key=len, reverse=True)

_OWNERSHIP_PATTERNS = (
    r"\bmy\s+saved\b",
    r"\bsaved\b",
    r"\bmy\s+logged\b",
    r"\blogged\b",
    r"\bin\s+my\s+pantry\b",
    r"\bfrom\s+my\s+pantry\b",
    r"\bmy\s+pantry\b",
    r"\bdo\s+i\s+have\b",
    r"\bhave\s+any\b",
    r"\bi\s+have\b",
    r"\bi\s+saved\b",
    r"\bi\s+logged\b",
    r"\bfor\s+my\b",
    r"\bof\s+my\b",
    r"\bis\s+my\b",
    r"\bare\s+my\b",
    r"\bcheck\s+my\b",
    r"\bstore\s+my\b",
)


def normalize(text: str) -> str:
    q = (text or "").lower().strip()
    q = re.sub(r"[^\w\s?']", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q


def _has_pantry_ownership(q: str) -> bool:
    return any(re.search(pat, q) for pat in _OWNERSHIP_PATTERNS)


def extract_food_noun(text: str) -> Optional[str]:
    q = normalize(text)
    for key in _SORTED_FOOD_KEYS:
        if re.search(rf"\b{re.escape(key)}\b", q):
            return FOOD_ALIASES[key]

    # Dynamic fallback extraction for arbitrary food names in phrases like
    # "my saved <food>" / "store my <food>" / "have any <food>"
    fallback_patterns = [
        r"\b(?:my\s+saved|saved|my\s+logged|logged|my)\s+([a-z][a-z\s]{1,24}?)(?:\s+(?:in\s+my\s+pantry|left|remain|remaining|expire|last|yet|still|today)\b|\?|$)",
        r"\bdo\s+i\s+have\s+(?:any\s+)?([a-z][a-z\s]{1,24}?)(?:\s+in\s+my\s+pantry|\?|$)",
        r"\bhow\s+(?:should|to|do)(?:\s+i)?\s+store\s+(?:fresh\s+|ripe\s+|whole\s+)?([a-z][a-z\s]{1,24}?)(?:\?|$)",
    ]
    stop_tokens = {
        "food", "foods", "item", "items", "produce", "pantry", "it", "this", "that", "them",
        "things", "everything", "first", "one",
    }
    for pat in fallback_patterns:
        m = re.search(pat, q)
        if m:
            cand = re.sub(r"\b(?:fresh|ripe|whole|properly|correctly|well|today|any|some)\b", "", m.group(1)).strip()
            if cand and cand not in stop_tokens and len(cand.split()) <= 3:
                if cand.endswith("oes") and len(cand) > 4:
                    cand = cand[:-2]
                elif cand.endswith("ies") and len(cand) > 4:
                    cand = cand[:-3] + "y"
                elif cand.endswith("s") and len(cand) > 3 and not cand.endswith("ss"):
                    cand = cand[:-1]
                return cand
    return None


def is_follow_up_query(text: str) -> bool:
    q = normalize(text).rstrip("?.! ")
    if not q:
        return False
    words = q.split()
    if len(words) <= 9 and any(
        re.search(rf"\b{re.escape(p)}\b", q)
        for p in (
            "why that one",
            "why that",
            "why is it",
            "why is that",
            "that one",
            "how should i store it",
            "how to store it",
            "store it",
            "store them",
            "what about it",
            "what about that",
            "what about them",
            "tell me more about it",
            "tell me more",
            "how long will it last",
            "how long does it last",
            "how many days left for it",
            "explain why",
        )
    ):
        return True
    return q in ("why", "how", "and storage", "what else", "more details")


def detect_fresho_intent(
    query: str,
    *,
    has_analysis_context: bool = False,
    is_short_follow_up: bool = False,
) -> Tuple[FreshoIntent, ContextLevel, Optional[str]]:
    """
    Returns (intent, context_level, optional_target_food).

    ContextLevel contract:
    - SIMPLE: general food education (NEVER injects pantry records)
    - MEDIUM: specific pantry food lookup, active scan context, or relevant follow-up
    - STRONG: complete pantry triage / inventory / quality limits
    """
    q = normalize(query)
    has_ownership = _has_pantry_ownership(q)
    target_food = extract_food_noun(q)

    # 1. Explicit follow-up references prior turn
    if is_short_follow_up or is_follow_up_query(q):
        return FreshoIntent.FOLLOW_UP, ContextLevel.MEDIUM, target_food

    # 2. Active scanned item context ("tell me about this food", "how should I store this")
    if has_analysis_context and any(
        p in q for p in ("this food", "tell me about this", "this item", "store this", "about this")
    ):
        return FreshoIntent.ANALYSIS_CONTEXT, ContextLevel.MEDIUM, target_food

    # 3. STRONG — Eat First / Priority across the whole pantry
    if any(
        p in q
        for p in (
            "eat first",
            "use first",
            "consume first",
            "what should i eat first",
            "which food should i eat first",
            "which food should i consume first",
            "which item should i eat first",
            "before it goes bad",
            "most urgent",
            "priority to eat",
        )
    ):
        return FreshoIntent.PANTRY_PRIORITY, ContextLevel.STRONG, None

    # 4. STRONG — Expiring / Approaching Quality Limits
    if any(
        p in q
        for p in (
            "approaching their quality limit",
            "approaching its quality limit",
            "closest to its quality limit",
            "closest to quality limit",
            "shortest remaining quality",
            "expiring soon",
            "expire soon",
            "going bad soon",
            "spoil soon",
            "needs attention",
            "high priority",
            "low priority",
        )
    ):
        return FreshoIntent.EXPIRING_SOON, ContextLevel.STRONG, None

    # 5. STRONG — Cook today / meal planning from pantry
    if any(p in q for p in ("cook today", "cook tonight", "what should i cook", "what can i cook", "what to make today")):
        return FreshoIntent.COOK_TODAY, ContextLevel.STRONG, None

    # 6. STRONG — Storage optimization across whole pantry
    if any(p in q for p in ("storage advice for my pantry", "store differently", "organize my pantry")):
        return FreshoIntent.PANTRY_LIST, ContextLevel.STRONG, None

    # 7. MEDIUM vs STRONG — Specific named food IN USER'S PANTRY vs full pantry list
    if has_ownership and target_food:
        if any(p in q for p in ("store", "storage", "keep", "fridge", "refrigerate", "countertop")):
            return FreshoIntent.PANTRY_ITEM_STORAGE, ContextLevel.MEDIUM, target_food
        return FreshoIntent.PANTRY_ITEM_LOOKUP, ContextLevel.MEDIUM, target_food

    # 8. STRONG — Whole pantry inventory requests
    if any(
        p in q
        for p in (
            "what foods are in my pantry",
            "what food is in my pantry",
            "what is in my pantry",
            "what s in my pantry",
            "whats in my pantry",
            "what foods do i have",
            "what food do i have",
            "foods do i have",
            "items in my pantry",
            "show me my pantry",
            "my pantry situation",
            "pantry situation",
            "list my food",
            "my logged food",
            "my saved food",
            "which food is freshest",
            "least fresh",
        )
    ) or (has_ownership and ("pantry" in q or "saved food" in q or "logged food" in q)):
        return FreshoIntent.PANTRY_LIST, ContextLevel.STRONG, None

    # 9. SIMPLE — General storage advice WITHOUT pantry ownership
    store_match = re.search(r"how\s+(?:should|to|do)(?:\s+i)?\s+store\s+(?:the\s+|this\s+|fresh\s+|ripe\s+)?([a-z\s]+)", q)
    if store_match:
        return FreshoIntent.STORAGE_ADVICE, ContextLevel.SIMPLE, target_food or (
            re.sub(r"\b(?:properly|correctly|well|today|fresh|ripe|whole)\b", "", store_match.group(1).replace("?", "")).strip() or None
        )

    # 10. SIMPLE — General freshness / ripening / shelf-life education
    if any(p in q for p in ("slightly spoiled", "what is slightly spoiled", "what does slightly spoiled mean")):
        return FreshoIntent.FRESHNESS_EXPLANATION, ContextLevel.SIMPLE, target_food

    if any(p in q for p in ("brown spot", "turn brown", "why do bananas", "why bananas")):
        return FreshoIntent.FRESHNESS_EXPLANATION, ContextLevel.SIMPLE, target_food or "banana"

    if "waste" in q and ("reduce" in q or "food" in q):
        if "pantry" in q and has_ownership:
            return FreshoIntent.FOOD_WASTE, ContextLevel.STRONG, None
        return FreshoIntent.FOOD_WASTE, ContextLevel.SIMPLE, target_food

    return FreshoIntent.GENERAL_FOOD_QUESTION, ContextLevel.SIMPLE, target_food


def map_fresho_intent_to_pantry_intent(fresho_intent: FreshoIntent) -> str:
    from backend.app.services.pantry_triage_service import PantryIntent

    mapping = {
        FreshoIntent.PANTRY_PRIORITY: PantryIntent.EAT_FIRST,
        FreshoIntent.PANTRY_LIST: PantryIntent.PANTRY_INVENTORY,
        FreshoIntent.EXPIRING_SOON: PantryIntent.EXPIRING_SOON,
        FreshoIntent.COOK_TODAY: PantryIntent.COOK_TODAY,
        FreshoIntent.PANTRY_ITEM_LOOKUP: PantryIntent.PANTRY_ITEM_LOOKUP,
        FreshoIntent.PANTRY_ITEM_STORAGE: PantryIntent.SPECIFIC_FOOD_STORAGE,
        FreshoIntent.STORAGE_ADVICE: PantryIntent.SPECIFIC_FOOD_STORAGE,
        FreshoIntent.FOOD_WASTE: PantryIntent.GENERAL_EDUCATIONAL,
        FreshoIntent.FRESHNESS_EXPLANATION: PantryIntent.GENERAL_EDUCATIONAL,
        FreshoIntent.GENERAL_FOOD_QUESTION: PantryIntent.GENERAL_EDUCATIONAL,
        FreshoIntent.ANALYSIS_CONTEXT: PantryIntent.GENERAL_EDUCATIONAL,
        FreshoIntent.FOLLOW_UP: PantryIntent.GENERAL_EDUCATIONAL,
    }
    return mapping.get(fresho_intent, PantryIntent.GENERAL_EDUCATIONAL)

"""
FoodFresh AI - Pantry Triage & History Decision Engine
Deterministic calculation engine for prioritizing user's pantry produce.
Evaluates remaining quality window, visible freshness state, Eat First tier,
and storage condition. Groq explains these decisions rather than inventing them.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("foodfresh.pantry_triage")


class PantryIntent:
    EAT_FIRST = "eat_first"
    COOK_TODAY = "cook_today"
    EXPIRING_SOON = "expiring_soon"
    PANTRY_INVENTORY = "pantry_inventory"
    PANTRY_ITEM_LOOKUP = "pantry_item_lookup"
    FRESHEST = "freshest"
    LEAST_FRESH = "least_fresh"
    SHORTEST_SHELF_LIFE = "shortest_shelf_life"
    LONGEST_SHELF_LIFE = "longest_shelf_life"
    PRIORITY_HIGH = "priority_high"
    PRIORITY_LOW = "priority_low"
    STORAGE_OPTIMIZATION = "storage_optimization"
    SPECIFIC_FOOD_STORAGE = "specific_food_storage"
    GENERAL_EDUCATIONAL = "general_educational"


def detect_pantry_intent(query: str) -> Tuple[str, Optional[str]]:
    """
    Detect the specific user intent regarding pantry produce.
    Returns (intent_type, optional_target_food).
    """
    from backend.app.services.fresho_intent_service import FreshoIntent, detect_fresho_intent, map_fresho_intent_to_pantry_intent

    fresho_intent, _, target = detect_fresho_intent(query)
    if fresho_intent == FreshoIntent.GENERAL_FOOD_QUESTION:
        q = query.lower().strip()
        if "which food is freshest" in q or "most fresh" in q:
            return PantryIntent.FRESHEST, None
        if "least fresh" in q or "worst condition" in q:
            return PantryIntent.LEAST_FRESH, None
        if "longest shelf" in q or "lasts longest" in q:
            return PantryIntent.LONGEST_SHELF_LIFE, None
    return map_fresho_intent_to_pantry_intent(fresho_intent), target


def _format_remaining_days(rem_days: Optional[int]) -> str:
    if rem_days is None:
        return "remaining quality unavailable"
    if rem_days == 0:
        return "0 days remaining"
    if rem_days == 1:
        return "1 day remaining"
    return f"{rem_days} days remaining"


def evaluate_item_urgency(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic scoring of a single pantry item.
    Higher score = higher urgency to consume.
    """
    food_name = item.get("foodName", "Unknown Produce")
    status = str(item.get("status", "Fresh")).lower()
    raw_rem_days = item.get("remainingDays")
    rem_known = raw_rem_days is not None
    eff_rem = int(raw_rem_days) if rem_known else None

    storage_env = str(item.get("storageEnvironment", "")).lower()
    is_countertop = "countertop" in storage_env or "ambient" in storage_env

    # Base Score by Remaining Quality Days
    if eff_rem is None:
        base_score = 5
        tier = "UNAVAILABLE"
    elif eff_rem <= 0:
        base_score = 100
        tier = "VERY_HIGH"
    elif eff_rem == 1:
        base_score = 80
        tier = "HIGH"
    elif eff_rem <= 3:
        base_score = 65
        tier = "HIGH"
    elif eff_rem <= 7:
        base_score = 35
        tier = "MEDIUM"
    else:
        base_score = 15
        tier = "LOW"

    # Visible Freshness Override / Modifier
    freshness_mod = 0
    if "rotten" in status or "spoil" in status or "decay" in status:
        if "slightly" in status:
            base_score = max(base_score, 75)
            if tier != "VERY_HIGH":
                tier = "HIGH"
            freshness_mod = 15
        else:
            base_score = max(base_score, 120)
            tier = "VERY_HIGH"
            freshness_mod = 30
    elif "semi" in status or "soft" in status or "stale" in status:
        base_score = max(base_score, 75)
        if tier != "VERY_HIGH":
            tier = "HIGH"
        freshness_mod = 15

    # Storage penalty: Countertop produce with <= 1 day left declines faster than chilled
    storage_mod = 5 if (is_countertop and eff_rem is not None and eff_rem <= 1) else 0

    total_score = base_score + freshness_mod + storage_mod
    q_score = item.get("qualityScore")
    if q_score is None:
        q_score = 80

    return {
        "foodName": food_name,
        "status": item.get("status", "Fresh"),
        "remainingDays": eff_rem,
        "remainingQualityKnown": rem_known,
        "qualityPeriod": item.get("qualityPeriod") or _format_remaining_days(eff_rem),
        "tier": tier,
        "urgencyScore": total_score,
        "storageEnvironment": item.get("storageEnvironment", "Countertop Ambient"),
        "storageType": item.get("storageType", "countertop"),
        "qualityScore": q_score,
        "guidance": item.get("guidance", ""),
        "cultivar": item.get("cultivar", f"{food_name} • Fresh Harvest"),
        "rawItem": item,
    }


def triage_pantry_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform a comprehensive deterministic triage across all user's saved pantry items.
    """
    if not items:
        return {
            "hasItems": False,
            "totalCount": 0,
            "rankedItems": [],
            "topUrgent": None,
            "tiedTop": [],
            "isTie": False,
            "expiringSoon": [],
            "freshest": None,
            "leastFresh": None,
            "summary": "Pantry history is empty.",
        }

    scored_items = [evaluate_item_urgency(it) for it in items]

    # Sort primarily by urgencyScore DESC, secondarily by remainingDays ASC (None last)
    scored_items.sort(
        key=lambda x: (
            -x["urgencyScore"],
            x["remainingDays"] is None,
            x["remainingDays"] if x["remainingDays"] is not None else 9999,
            x["foodName"],
        )
    )

    top_item = scored_items[0]
    top_score = top_item["urgencyScore"]

    # Detect ties at the top tier
    tied_items = [it for it in scored_items if it["urgencyScore"] == top_score]
    is_tie = len(tied_items) > 1

    # Items needing attention (<= 3 days remaining OR tier in HIGH/VERY_HIGH)
    expiring_soon = [
        it
        for it in scored_items
        if (it["remainingDays"] is not None and it["remainingDays"] <= 3) or it["tier"] in ["VERY_HIGH", "HIGH"]
    ]

    # Freshest: longest known remaining quality window
    known_items = [it for it in scored_items if it["remainingDays"] is not None]
    freshest_pool = known_items or scored_items
    freshest_item = min(
        freshest_pool,
        key=lambda x: (-(x["remainingDays"] if x["remainingDays"] is not None else -1), -x["qualityScore"]),
    )

    least_fresh_item = scored_items[0]

    return {
        "hasItems": True,
        "totalCount": len(scored_items),
        "rankedItems": scored_items,
        "topUrgent": top_item,
        "tiedTop": tied_items if is_tie else [],
        "isTie": is_tie,
        "expiringSoon": expiring_soon,
        "freshest": freshest_item,
        "leastFresh": least_fresh_item,
    }


def find_matching_pantry_items(triage: Dict[str, Any], target_food: Optional[str]) -> List[Dict[str, Any]]:
    """Find saved pantry items matching target_food (case-insensitive, singular/plural friendly)."""
    if not triage.get("hasItems") or not target_food:
        return []
    needle = target_food.lower().strip()
    matches = []
    for it in triage["rankedItems"]:
        name_low = str(it.get("foodName", "")).lower()
        if needle in name_low or name_low in needle:
            matches.append(it)
    return matches


def build_deterministic_answer(intent: str, triage: Dict[str, Any], target_food: Optional[str] = None) -> str:
    """
    Deterministic pantry summary template (used as factual anchor/structured reference,
    never as a fake substitute when Groq API is required and unreachable).
    """
    if intent in (PantryIntent.PANTRY_ITEM_LOOKUP, PantryIntent.SPECIFIC_FOOD_STORAGE) and target_food:
        matches = find_matching_pantry_items(triage, target_food)
        if matches:
            top_m = matches[0]
            rem_text = _format_remaining_days(top_m.get("remainingDays"))
            return (
                f"According to your saved pantry history, you have **{top_m['foodName']}** "
                f"({top_m['status']}, **{rem_text}**, stored in **{top_m['storageEnvironment']}**, Priority: **{top_m['tier']}**)."
            )
        food_display = target_food.strip().title()
        return (
            f"**{food_display}** is not currently saved in your pantry history."
        )

    if not triage["hasItems"]:
        return (
            "Your pantry history is currently empty, so I can't identify a specific food to prioritize yet. "
            "Analyze and save a food first, and I'll track its freshness and prioritize it for you 🌱"
        )

    top = triage["topUrgent"]
    is_tie = triage["isTie"]
    tied = triage["tiedTop"]

    if intent == PantryIntent.EAT_FIRST:
        if is_tie:
            names = " and ".join([f"**{it['foodName']}**" for it in tied])
            rem_txt = _format_remaining_days(top.get("remainingDays"))
            return (
                f"Based on your saved pantry history, {names} **should be used first** ({rem_txt}).\n\n"
                f"**Why:**\n"
                + "\n".join(
                    [f"- **{it['foodName']}** — {_format_remaining_days(it.get('remainingDays'))} ({it['status']}, {it['tier']} priority)" for it in tied]
                )
            )
        return (
            f"Based on your saved pantry history, **{top['foodName']} should be used first**.\n\n"
            f"- **{top['foodName']}** — {_format_remaining_days(top.get('remainingDays'))}\n"
            f"- **Freshness:** {top['status']}\n"
            f"- **Priority:** {top['tier']}"
        )

    if intent == PantryIntent.PANTRY_INVENTORY:
        ranked = triage["rankedItems"]
        lines = [
            f"- **{it['foodName']}**: {it['status']} • {_format_remaining_days(it.get('remainingDays'))} • Priority: **{it['tier']}**"
            for it in ranked
        ]
        return (
            f"Here is your current **Pantry Produce Log** ({len(ranked)} items):\n\n"
            + "\n".join(lines)
            + f"\n\n**Most urgent:** **{top['foodName']}** should be used first."
        )

    if intent == PantryIntent.EXPIRING_SOON:
        items_soon = triage["expiringSoon"]
        if not items_soon:
            return "Based on your saved pantry history, none of your logged produce items are approaching their quality limits soon."
        lines = [f"- **{it['foodName']}** — {_format_remaining_days(it.get('remainingDays'))} ({it['tier']} priority)" for it in items_soon]
        return "Items approaching their quality limits:\n\n" + "\n".join(lines)

    return f"**{top['foodName']}** ({_format_remaining_days(top.get('remainingDays'))}) has the highest Eat First priority ({top['tier']})."

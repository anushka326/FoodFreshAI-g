"""
FoodFresh AI - FreshoBuddy Groq AI Service
Official Groq Python SDK integration providing unified general & pantry food intelligence,
strict 3-tier context detection (SIMPLE / MEDIUM / STRONG), deterministic application authority,
and food safety disclaimers.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.services.pantry_triage_service import (
    PantryIntent,
    _format_remaining_days,
    find_matching_pantry_items,
    triage_pantry_items,
)
from backend.app.services.fresho_intent_service import (
    ContextLevel,
    FreshoIntent,
    detect_fresho_intent,
    extract_food_noun,
    is_follow_up_query,
    map_fresho_intent_to_pantry_intent,
)
from backend.app.services.pantry_state_service import get_current_pantry_state

logger = logging.getLogger("foodfresh.fresho_buddy")

# Load environment variables from .env or backend/.env if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
for env_candidate in [PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"]:
    if env_candidate.exists():
        try:
            with open(env_candidate, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            logger.warning(f"Could not load {env_candidate}: {e}")

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

FRESHO_BUDDY_SYSTEM_PROMPT = """You are FreshoBuddy AI 🌱, the unified food intelligence assistant inside FoodFresh AI.

You answer BOTH general food science/storage/shelf-life questions AND questions grounded in the authenticated user's saved pantry records.

CORE APPLICATION AUTHORITY RULES:
1. STRICT CONTEXT BOUNDARIES:
   - If NO saved pantry block is provided (SIMPLE context), answer strictly with general food guidance. Never guess or claim what is in the user's pantry.
   - If a MEDIUM specific-item lookup says the requested food is NOT in the user's pantry, explicitly tell the user that it is not saved in their pantry, then provide helpful general guidance for that food.
   - If a STRONG or MEDIUM saved-pantry block is provided, treat the supplied application data as the single source of truth. Never invent pantry items, dates, remaining days, freshness statuses, or confidence scores.
2. DETERMINISTIC EAT FIRST PRIORITY:
   - The application computes the Eat First priority, urgency tier, and remaining quality days deterministically.
   - Explain the supplied Eat First winner (or ties) accurately without altering or overriding the ranking.
3. FOOD SAFETY DISCLAIMER:
   - Visual freshness analysis and USDA FoodKeeper windows are quality estimates, not laboratory food-safety inspections. Advise users to check smell, texture, firmness, and visible mold before eating questionable food.
4. CLEAR MARKDOWN FORMATTING:
   - Use clean Markdown formatting (**bold** key food names and remaining-day values, bullet lists when helpful). Be concise, practical, and friendly."""


class FreshoBuddyGroqError(RuntimeError):
    """Raised when Groq inference cannot be completed (missing key, invalid key, or API error)."""

    def __init__(self, message: str, *, code: str = "GROQ_API_ERROR", status_code: int = 502):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def get_groq_config() -> Dict[str, str]:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    model = os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
    base_url = os.environ.get("GROQ_BASE_URL", "").strip()
    return {"api_key": api_key, "model": model, "base_url": base_url}


def derive_conversation_title(first_user_message: str, food_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Deterministic generation of a concise conversation title from the first message and optional scan context.
    """
    msg_lower = (first_user_message or "").lower()

    if "eat first" in msg_lower or "which food" in msg_lower or "use first" in msg_lower or "consume first" in msg_lower:
        return "Pantry Priority"
    if "cook today" in msg_lower or "cook tonight" in msg_lower or "what to make" in msg_lower:
        return "Pantry Meal Ideas"
    if "expir" in msg_lower or "attention" in msg_lower or "quality limit" in msg_lower:
        return "Expiring Foods"
    if "reduce" in msg_lower and "waste" in msg_lower:
        return "Reduce Food Waste"
    if "brown spot" in msg_lower or ("banana" in msg_lower and "brown" in msg_lower):
        return "Banana Browning"
    if "pantry" in msg_lower or "inventory" in msg_lower or "what food" in msg_lower:
        return "Pantry Overview"

    if food_context and (food_context.get("detectedFood") or food_context.get("foodName")):
        food = str(food_context.get("detectedFood") or food_context.get("foodName")).strip().title()
        if "store" in msg_lower or "keep" in msg_lower or "fridge" in msg_lower:
            return f"{food} Storage Advice"
        if "fresh" in msg_lower or "spoil" in msg_lower or "rot" in msg_lower:
            return f"{food} Freshness"
        if "recipe" in msg_lower or "cook" in msg_lower or "eat" in msg_lower:
            return f"{food} Ideas"
        return f"{food} Discussion"

    store_match = re.search(r"how (?:should|to|do)(?: i)? store (?:this |the |my |fresh |ripe )?([a-z\s]+)", msg_lower)
    if store_match:
        target = store_match.group(1).replace("?", "").strip().title()
        if target:
            return f"{target} Storage Advice"[:40]

    clean = re.sub(r"[^\w\s]", "", first_user_message or "").strip()
    words = clean.split()
    if not words:
        return "New Food Discussion"

    return " ".join(words[:4]).title()[:40]


def _resolve_follow_up_context(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Inspect previous conversation turns to resolve pronouns ('that one', 'it', 'them')
    AND whether the previous exchange was pantry-scoped (MEDIUM/STRONG) or general (SIMPLE).
    """
    for idx in range(len(history) - 1, -1, -1):
        msg = history[idx]
        if msg.get("role") != "assistant":
            continue
        text = msg.get("content") or ""
        prev_user_text = ""
        for j in range(idx - 1, -1, -1):
            if history[j].get("role") == "user":
                prev_user_text = history[j].get("content") or ""
                break

        prev_intent, prev_level, prev_food = detect_fresho_intent(prev_user_text)
        was_pantry_scoped = prev_level in (ContextLevel.STRONG, ContextLevel.MEDIUM) or any(
            k in text.lower()
            for k in ("saved pantry", "should be used first", "pantry produce log", "days remaining", "priority:")
        )

        # Look for explicit "**Mango should be used first**" or bolded food name in assistant reply
        resolved_food = None
        m_first = re.search(r"\*\*([A-Za-z][A-Za-z\s]{1,30}?)\s+should be used first\*\*", text)
        if m_first:
            resolved_food = extract_food_noun(m_first.group(1)) or m_first.group(1).strip()
        if not resolved_food:
            for m_bold in re.finditer(r"\*\*([A-Za-z][A-Za-z\s]{1,30}?)\*\*", text):
                candidate = extract_food_noun(m_bold.group(1))
                if candidate:
                    resolved_food = candidate
                    break
        if not resolved_food:
            resolved_food = prev_food or extract_food_noun(prev_user_text) or extract_food_noun(text)

        return {
            "resolvedFood": resolved_food,
            "wasPantryScoped": was_pantry_scoped,
            "priorIntent": prev_intent,
            "priorLevel": prev_level,
        }

    return {"resolvedFood": None, "wasPantryScoped": False, "priorIntent": None, "priorLevel": ContextLevel.SIMPLE}


def _enforce_application_authority(
    reply_text: str,
    context_level: ContextLevel,
    fresho_intent: FreshoIntent,
    triage: Optional[Dict[str, Any]],
    matched_items: List[Dict[str, Any]],
    target_food: Optional[str],
) -> str:
    """
    Guarantee Groq output never omits or contradicts deterministic Eat First priority
    or misstates an absent/present pantry item.
    """
    text = (reply_text or "").strip()
    if not text:
        return text

    if context_level == ContextLevel.STRONG and fresho_intent == FreshoIntent.PANTRY_PRIORITY:
        if triage and triage.get("hasItems") and triage.get("topUrgent"):
            winners = triage["tiedTop"] if triage.get("isTie") else [triage["topUrgent"]]
            missing_winner = any(w["foodName"].lower() not in text.lower() for w in winners)
            if missing_winner:
                names = " and ".join(f"**{w['foodName']}**" for w in winners)
                rem_txt = _format_remaining_days(winners[0].get("remainingDays"))
                text = (
                    f"**Application Eat First Priority:** {names} ({rem_txt}, {winners[0]['tier']} priority).\n\n"
                    + text
                )
    elif context_level == ContextLevel.MEDIUM and target_food and not matched_items:
        # Ensure user clearly sees it is not in their pantry if model skipped mentioning it
        if "not" not in text.lower() and "pantry" not in text.lower():
            text = f"**{target_food.strip().title()}** is not currently saved in your pantry log.\n\n" + text

    return text


class FreshoBuddyService:
    """
    Manages conversational interactions with Groq using the official Groq Python SDK,
    enforcing 3-tier context isolation (SIMPLE/MEDIUM/STRONG) and application authority.
    """

    _instance: Optional["FreshoBuddyService"] = None

    def __init__(self):
        self.client = None
        self.is_configured = False
        self.model_name = DEFAULT_GROQ_MODEL
        self._last_api_key = ""
        self._last_base_url = ""
        self._init_groq_client()

    @classmethod
    def get_instance(cls) -> "FreshoBuddyService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_groq_client(self):
        cfg = get_groq_config()
        self.model_name = cfg["model"]
        api_key = cfg["api_key"]
        base_url = cfg["base_url"]

        if not api_key:
            self.is_configured = False
            self.client = None
            self._last_api_key = ""
            self._last_base_url = base_url
            return

        if self.client is not None and api_key == self._last_api_key and base_url == self._last_base_url:
            self.is_configured = True
            return

        try:
            from groq import Groq

            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = Groq(**kwargs)
            self.is_configured = True
            self._last_api_key = api_key
            self._last_base_url = base_url
            logger.info(f"FreshoBuddy initialized with official Groq Python SDK (Model: {self.model_name})")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.is_configured = False
            self.client = None

    def build_context_bundle(
        self,
        user_message: str,
        user_id: str,
        history: Optional[List[Dict[str, Any]]] = None,
        analysis_context: Optional[Dict[str, Any]] = None,
        pantry_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build deterministic intent, context level, authority payload, and Groq chat messages."""
        history = history or []
        short_follow_up = bool(history) and is_follow_up_query(user_message)
        fresho_intent, context_level, target_food = detect_fresho_intent(
            user_message,
            has_analysis_context=bool(analysis_context),
            is_short_follow_up=short_follow_up,
        )

        if fresho_intent == FreshoIntent.FOLLOW_UP and history:
            follow_ctx = _resolve_follow_up_context(history)
            if not target_food:
                target_food = follow_ctx["resolvedFood"]
            if not follow_ctx["wasPantryScoped"] and not analysis_context:
                # Prior question was SIMPLE educational chat — do NOT inject pantry records
                context_level = ContextLevel.SIMPLE
            else:
                context_level = ContextLevel.MEDIUM

        pantry_intent = map_fresho_intent_to_pantry_intent(fresho_intent)
        triage: Optional[Dict[str, Any]] = None
        matched_items: List[Dict[str, Any]] = []
        pantry_injected = False
        authority_block = ""

        # SIMPLE MUST NOT query or inject user pantry records into the AI prompt.
        if context_level in (ContextLevel.MEDIUM, ContextLevel.STRONG):
            if pantry_items is None and (
                context_level == ContextLevel.STRONG
                or target_food is not None
                or fresho_intent == FreshoIntent.FOLLOW_UP
            ):
                try:
                    pantry_items = get_current_pantry_state(user_id)
                except Exception as e:
                    logger.warning(f"Could not load pantry items for user {user_id}: {e}")
                    pantry_items = []

            if context_level == ContextLevel.STRONG:
                triage = triage_pantry_items(pantry_items or [])
                pantry_injected = True
                if not triage["hasItems"]:
                    authority_block = (
                        "\n[USER PANTRY AUTHORITY DATA]\n"
                        "- Pantry Status: EMPTY (0 saved food items).\n"
                        "- Instruction: Inform the user clearly that their pantry is currently empty and suggest analyzing and saving food items first. Do NOT invent any food items.\n"
                    )
                else:
                    top = triage["topUrgent"]
                    items_lines = "\n".join(
                        f"- {it['foodName']}: status={it['status']} | remaining={_format_remaining_days(it.get('remainingDays'))} | priority={it['tier']} | storage={it['storageEnvironment']}"
                        for it in triage["rankedItems"]
                    )
                    tied_names = [it["foodName"] for it in triage["tiedTop"]] if triage["isTie"] else []
                    authority_block = (
                        f"\n[USER SAVED PANTRY LOG — {triage['totalCount']} ITEMS]\n{items_lines}\n\n"
                        f"[APPLICATION DETERMINISTIC EAT-FIRST CALCULATION]\n"
                        f"- Intent: {pantry_intent}\n"
                        f"- Calculated Top Priority Food: {top['foodName']} (Tier: {top['tier']}, UrgencyScore: {top['urgencyScore']}, Remaining: {_format_remaining_days(top.get('remainingDays'))})\n"
                        f"- Is Tied At Top: {triage['isTie']} (Tied: {tied_names or 'None'})\n"
                        f"- Strict Rule: Ground your answer ONLY in the saved items above. Never override the priority ranking or invent foods not listed above.\n"
                    )
            elif context_level == ContextLevel.MEDIUM:
                if target_food and pantry_items is not None:
                    triage = triage_pantry_items(pantry_items)
                    matched_items = find_matching_pantry_items(triage, target_food)
                    pantry_injected = True
                    if matched_items:
                        lines = "\n".join(
                            f"- {it['foodName']}: status={it['status']} | remaining={_format_remaining_days(it.get('remainingDays'))} | priority={it['tier']} | storage={it['storageEnvironment']} | guidance={it.get('guidance', '')}"
                            for it in matched_items
                        )
                        authority_block = (
                            f"\n[MATCHING SAVED PANTRY RECORD FOR '{target_food.upper()}']\n{lines}\n"
                            f"- Strict Rule: Cite the exact remaining quality days, freshness status, priority tier, and storage above. Do not fabricate additional items.\n"
                        )
                    else:
                        authority_block = (
                            f"\n[PANTRY LOOKUP RESULT FOR '{target_food.upper()}']\n"
                            f"- Found In User's Saved Pantry: FALSE (0 matching items for '{target_food}').\n"
                            f"- Strict Rule: Explicitly state that {target_food.title()} is NOT currently saved in the user's pantry, and then provide helpful general storage/freshness guidance for {target_food.title()}.\n"
                        )
                elif fresho_intent == FreshoIntent.FOLLOW_UP and pantry_items is not None:
                    triage = triage_pantry_items(pantry_items)
                    if triage["hasItems"] and triage["topUrgent"]:
                        pantry_injected = True
                        top = triage["topUrgent"]
                        authority_block = (
                            f"\n[FOLLOW-UP PANTRY CONTEXT]\n"
                            f"- Top Ranked Item: {top['foodName']} (status={top['status']}, remaining={_format_remaining_days(top.get('remainingDays'))}, priority={top['tier']}, storage={top['storageEnvironment']})\n"
                            f"- Total Pantry Items: {triage['totalCount']}\n"
                        )

        scan_context_block = ""
        if analysis_context and (analysis_context.get("detectedFood") or analysis_context.get("foodName")):
            food = analysis_context.get("detectedFood") or analysis_context.get("foodName")
            freshness = analysis_context.get("freshness", {})
            f_label = freshness.get("label") if isinstance(freshness, dict) else str(freshness)
            f_conf = freshness.get("confidence") if isinstance(freshness, dict) else None
            shelf_life = analysis_context.get("shelfLife", {})
            rem = shelf_life.get("remaining", {}) if isinstance(shelf_life, dict) else {}
            rem_str = f"{rem.get('minDays', '?')}–{rem.get('maxDays', '?')} days" if rem else "Guidance unavailable"
            storage = analysis_context.get("storageType", "countertop")
            days_stored = analysis_context.get("daysStored", 0)
            scan_context_block = (
                f"\n[ACTIVE SCANNED FOOD CONTEXT]\n"
                f"- Food Item: {food}\n"
                f"- Estimated Visible Freshness: {f_label or 'Uncertain'} ({f_conf or 'N/A'}% confidence)\n"
                f"- Estimated Remaining Quality Window: {rem_str}\n"
                f"- Storage Environment: {storage}\n"
                f"- Days Stored So Far: {days_stored} days\n"
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": FRESHO_BUDDY_SYSTEM_PROMPT}]
        for msg in history[-8:]:
            role = "user" if msg.get("role") == "user" else "assistant"
            content = (msg.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})

        context_prefix = f"{authority_block}{scan_context_block}".strip()
        final_user_content = f"{context_prefix}\n\nUser question: {user_message}" if context_prefix else user_message
        messages.append({"role": "user", "content": final_user_content})

        return {
            "fresho_intent": fresho_intent,
            "context_level": context_level,
            "target_food": target_food,
            "pantry_injected": pantry_injected,
            "triage": triage,
            "matched_items": matched_items,
            "messages": messages,
        }

    def generate_reply(
        self,
        user_message: str,
        user_id: str = "guest",
        history: Optional[List[Dict[str, Any]]] = None,
        analysis_context: Optional[Dict[str, Any]] = None,
        pantry_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate response via the official Groq Python SDK with 3-tier context awareness.
        Raises FreshoBuddyGroqError when GROQ_API_KEY is missing/invalid or Groq fails
        (never substitutes mock responses for real Groq inference).
        """
        self._init_groq_client()
        if not self.is_configured or not self.client:
            raise FreshoBuddyGroqError(
                "FreshoBuddy AI requires a configured GROQ_API_KEY on the backend.",
                code="GROQ_UNAVAILABLE",
                status_code=503,
            )

        bundle = self.build_context_bundle(
            user_message=user_message,
            user_id=user_id,
            history=history,
            analysis_context=analysis_context,
            pantry_items=pantry_items,
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=bundle["messages"],
                temperature=0.4,
                max_tokens=700,
            )
            raw_reply = ""
            if completion and getattr(completion, "choices", None):
                msg_obj = completion.choices[0].message
                raw_reply = (getattr(msg_obj, "content", "") or "").strip()

            if not raw_reply:
                raise FreshoBuddyGroqError(
                    "Groq returned an empty response.",
                    code="GROQ_EMPTY_RESPONSE",
                    status_code=502,
                )

            final_text = _enforce_application_authority(
                raw_reply,
                bundle["context_level"],
                bundle["fresho_intent"],
                bundle["triage"],
                bundle["matched_items"],
                bundle["target_food"],
            )
            return {
                "text": final_text,
                "status": "success",
                "model": self.model_name,
                "contextLevel": bundle["context_level"].value,
                "intent": bundle["fresho_intent"].value,
                "pantryInjected": bundle["pantry_injected"],
            }
        except FreshoBuddyGroqError:
            raise
        except Exception as e:
            err_msg = str(e)
            low = err_msg.lower()
            is_auth = "401" in low or "403" in low or "invalid api key" in low or "authentication" in low or "unauthorized" in low
            logger.error(f"Error calling Groq API: {e}", exc_info=True)
            raise FreshoBuddyGroqError(
                "Groq AI authentication failed. Check GROQ_API_KEY configuration."
                if is_auth
                else f"Groq AI request failed: {err_msg}",
                code="GROQ_AUTH_ERROR" if is_auth else "GROQ_API_ERROR",
                status_code=502 if is_auth else 503,
            ) from e


def get_fresho_buddy_service() -> FreshoBuddyService:
    return FreshoBuddyService.get_instance()

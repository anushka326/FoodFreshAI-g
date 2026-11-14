"""
Unit tests for Python FreshoBuddy services, 3-tier context isolation, SQLite chat DB,
and Groq SDK integration (with unittest.mock).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import chat_db
from backend.app.services import auth_service, fresho_buddy_service, fresho_intent_service, pantry_triage_service


class TestFreshoBuddyPython(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_db_path = chat_db.DB_PATH
        chat_db.DB_PATH = Path(self._tmp.name) / "test_fresho_py.db"
        chat_db.init_db()
        fresho_buddy_service.FreshoBuddyService._instance = None

    def tearDown(self) -> None:
        chat_db.DB_PATH = self._orig_db_path
        fresho_buddy_service.FreshoBuddyService._instance = None
        self._tmp.cleanup()

    def test_intent_classification_three_levels(self) -> None:
        for q in (
            "Average shelf life of freshly purchased bananas",
            "How to store fresh potatoes",
            "Why do bananas develop brown spots?",
            "How long do fresh tomatoes last?",
        ):
            _, lvl, _ = fresho_intent_service.detect_fresho_intent(q)
            self.assertEqual(lvl, fresho_intent_service.ContextLevel.SIMPLE, msg=q)

        intent, lvl, food = fresho_intent_service.detect_fresho_intent("How many days remain for my saved mangoes?")
        self.assertEqual(lvl, fresho_intent_service.ContextLevel.MEDIUM)
        self.assertEqual(intent, fresho_intent_service.FreshoIntent.PANTRY_ITEM_LOOKUP)
        self.assertEqual(food, "mango")

        intent2, lvl2, food2 = fresho_intent_service.detect_fresho_intent("How should I store my saved potatoes?")
        self.assertEqual(lvl2, fresho_intent_service.ContextLevel.MEDIUM)
        self.assertEqual(intent2, fresho_intent_service.FreshoIntent.PANTRY_ITEM_STORAGE)
        self.assertEqual(food2, "potato")

        for q in (
            "Which food should I consume first?",
            "What foods are in my pantry?",
            "Which saved foods are approaching their quality limits?",
        ):
            _, lvl, _ = fresho_intent_service.detect_fresho_intent(q)
            self.assertEqual(lvl, fresho_intent_service.ContextLevel.STRONG, msg=q)

    def test_context_isolation_and_triage(self) -> None:
        pantry = [
            {"foodName": "Mango", "status": "Rotten", "remainingDays": 0, "qualityScore": 0, "storageEnvironment": "Countertop"},
            {"foodName": "Apple", "status": "Fresh", "remainingDays": 6, "qualityScore": 92, "storageEnvironment": "Crisper"},
            {"foodName": "Mystery", "status": "Fresh", "remainingDays": None, "qualityScore": 50, "storageEnvironment": "Countertop"},
        ]
        svc = fresho_buddy_service.get_fresho_buddy_service()

        # General educational query must NOT inject pantry records
        simple_bundle = svc.build_context_bundle("How to store fresh potatoes", "usr_1", pantry_items=pantry)
        self.assertEqual(simple_bundle["context_level"], fresho_intent_service.ContextLevel.SIMPLE)
        self.assertFalse(simple_bundle["pantry_injected"])
        self.assertNotIn("Mango", str(simple_bundle["messages"]))

        # Triage preserves None remainingDays instead of 999
        triage = pantry_triage_service.triage_pantry_items(pantry)
        self.assertEqual(triage["topUrgent"]["foodName"], "Mango")
        mystery = [x for x in triage["rankedItems"] if x["foodName"] == "Mystery"][0]
        self.assertIsNone(mystery["remainingDays"])
        self.assertEqual(mystery["qualityPeriod"], "remaining quality unavailable")

        # Specific absent item lookup
        med_absent = svc.build_context_bundle("Do I have avocados in my pantry?", "usr_1", pantry_items=pantry)
        self.assertEqual(med_absent["context_level"], fresho_intent_service.ContextLevel.MEDIUM)
        self.assertEqual(med_absent["matched_items"], [])
        self.assertIn("Found In User's Saved Pantry: FALSE", med_absent["messages"][-1]["content"])

    def test_auth_sqlite_and_user_isolation(self) -> None:
        u1 = auth_service.register_user("u1@test.com", "secret12", "User One")
        u2 = auth_service.register_user("u2@test.com", "secret34", "User Two")
        self.assertEqual(auth_service.resolve_session(u1["token"])["id"], u1["user"]["id"])

        saved = chat_db.add_pantry_item(u1["user"]["id"], {"foodName": "Mango", "qualityScore": 0, "remainingDays": 0})
        self.assertEqual(saved["qualityScore"], 0)
        self.assertEqual(chat_db.get_pantry_items(u2["user"]["id"]), [])

        conv = chat_db.create_conversation(u1["user"]["id"], "Pantry Priority")
        chat_db.add_message(conv["conversationId"], u1["user"]["id"], "user", "Which food should I consume first?")
        self.assertIsNone(chat_db.get_conversation(conv["conversationId"], u2["user"]["id"]))
        self.assertEqual(chat_db.list_conversations(u2["user"]["id"]), [])

    def test_groq_sdk_mocked_and_missing_key_error(self) -> None:
        svc = fresho_buddy_service.get_fresho_buddy_service()

        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            with self.assertRaises(fresho_buddy_service.FreshoBuddyGroqError) as ctx:
                svc.generate_reply("Average shelf life of freshly purchased bananas", user_id="usr_1")
            self.assertEqual(ctx.exception.code, "GROQ_UNAVAILABLE")
            self.assertEqual(ctx.exception.status_code, 503)

        fake_completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Fresh **bananas** keep **2–5 days** at room temp."))]
        )
        fake_groq_client = MagicMock()
        fake_groq_client.chat.completions.create.return_value = fake_completion

        fake_groq_module = SimpleNamespace(Groq=MagicMock(return_value=fake_groq_client))
        with patch.dict(sys.modules, {"groq": fake_groq_module}):
            with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_123", "GROQ_MODEL": "llama-3.1-8b-instant"}, clear=False):
                fresho_buddy_service.FreshoBuddyService._instance = None
                svc2 = fresho_buddy_service.get_fresho_buddy_service()
                out = svc2.generate_reply(
                    "Average shelf life of freshly purchased bananas",
                    user_id="usr_1",
                    pantry_items=[{"foodName": "Mango", "status": "Rotten", "remainingDays": 0}],
                )
                self.assertEqual(out["model"], "llama-3.1-8b-instant")
                self.assertEqual(out["contextLevel"], "simple")
                self.assertFalse(out["pantryInjected"])
                fake_groq_client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()

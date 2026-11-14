"""
FoodFresh AI - Pure-Python ML Pipeline Verification & Regression Test Suite
Tests every component executable in the current Python environment honestly,
including model_config runtime checks, FoodForm, ShelfLifeService (against both
disk FoodKeeper and FoodKeeper-schema fixtures), EatFirstService, Freshness policy AST/JSON
vocabulary, and syntax compilation of ml/hybrid_vision and backend/app modules.
"""

import ast
import json
from pathlib import Path
import py_compile
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from backend.app.model_config import MODEL_CONFIG, get_runtime_model_status
from backend.app.services.eat_first_service import EatFirstService, get_eat_first_service
from backend.app.services.food_form_service import infer_food_form, normalize_food_form
from backend.app.services.shelf_life_service import FoodKeeperService


def _build_fixture_foodkeeper_json(target_path: Path) -> None:
    """Create a minimal USDA FoodKeeper-format JSON matching official schema IDs."""
    categories = [
        [{"ID": 18.0}, {"Category_Name": "Produce"}],
        [{"ID": 2.0}, {"Category_Name": "Baked Goods"}],
    ]
    products = [
        [
            {"ID": 248.0}, {"Name": "Apples"}, {"Category_ID": 18.0},
            {"Pantry_Min": 5.0}, {"Pantry_Max": 7.0}, {"Pantry_Metric": "Days"},
            {"Pantry_tips": "Keep in cool spot"},
            {"Refrigerate_Min": 3.0}, {"Refrigerate_Max": 4.0}, {"Refrigerate_Metric": "Weeks"},
            {"Refrigerate_tips": "Refrigerate in crisper drawer"},
        ],
        [
            {"ID": 251.0}, {"Name": "Bananas"}, {"Category_ID": 18.0},
            {"Pantry_Min": 2.0}, {"Pantry_Max": 5.0}, {"Pantry_Metric": "Days"},
            {"Pantry_tips": "Ripen at room temp"},
            {"Refrigerate_Min": 5.0}, {"Refrigerate_Max": 7.0}, {"Refrigerate_Metric": "Days"},
            {"Refrigerate_tips": "Skin darkens when chilled"},
        ],
        [
            {"ID": 306.0}, {"Name": "Tomatoes"}, {"Category_ID": 18.0},
            {"Pantry_Min": 3.0}, {"Pantry_Max": 5.0}, {"Pantry_Metric": "Days"},
            {"Pantry_tips": "Stem side down"},
            {"Refrigerate_Min": 7.0}, {"Refrigerate_Max": 10.0}, {"Refrigerate_Metric": "Days"},
            {"Refrigerate_tips": "Chill when ripe"},
        ],
        [
            {"ID": 256.0}, {"Name": "Citrus fruit"}, {"Category_ID": 18.0},
            {"Pantry_Min": 5.0}, {"Pantry_Max": 7.0}, {"Pantry_Metric": "Days"},
            {"Refrigerate_Min": 2.0}, {"Refrigerate_Max": 3.0}, {"Refrigerate_Metric": "Weeks"},
        ],
        [
            {"ID": 297.0}, {"Name": "Potatoes"}, {"Category_ID": 18.0},
            {"Pantry_Min": 1.0}, {"Pantry_Max": 2.0}, {"Pantry_Metric": "Weeks"},
            {"Pantry_tips": "Cool dark pantry"},
        ],
        [
            {"ID": 548.0}, {"Name": "Hot peppers"}, {"Category_ID": 18.0},
            {"Pantry_Min": 2.0}, {"Pantry_Max": 4.0}, {"Pantry_Metric": "Days"},
            {"Refrigerate_Min": 1.0}, {"Refrigerate_Max": 2.0}, {"Refrigerate_Metric": "Weeks"},
        ],
        [
            {"ID": 296.0}, {"Name": "Peppers"}, {"Category_ID": 18.0},
            {"Pantry_Min": 2.0}, {"Pantry_Max": 3.0}, {"Pantry_Metric": "Days"},
            {"Refrigerate_Min": 1.0}, {"Refrigerate_Max": 2.0}, {"Refrigerate_Metric": "Weeks"},
        ],
        [
            {"ID": 265.0}, {"Name": "Papaya, mango, feijoa, passionfruit, casaha melon"}, {"Category_ID": 18.0},
            {"Pantry_Min": 3.0}, {"Pantry_Max": 5.0}, {"Pantry_Metric": "Days"},
            {"Refrigerate_Min": 5.0}, {"Refrigerate_Max": 7.0}, {"Refrigerate_Metric": "Days"},
        ],
        [
            {"ID": 269.0}, {"Name": "Pomegranate"}, {"Category_ID": 18.0},
            {"Pantry_Min": 5.0}, {"Pantry_Max": 7.0}, {"Pantry_Metric": "Days"},
            {"Refrigerate_Min": 2.0}, {"Refrigerate_Max": 4.0}, {"Refrigerate_Metric": "Weeks"},
        ],
        [
            {"ID": 195.0}, {"Name": "Commercial bread products"}, {"Category_ID": 2.0},
            {"Pantry_Min": 4.0}, {"Pantry_Max": 7.0}, {"Pantry_Metric": "Days"},
            {"Refrigerate_Min": 7.0}, {"Refrigerate_Max": 14.0}, {"Refrigerate_Metric": "Days"},
        ],
        [
            {"ID": 495.0}, {"Name": "Watermelon"}, {"Category_ID": 18.0},
            {"DOP_Pantry_Min": 5.0}, {"DOP_Pantry_Max": 7.0}, {"DOP_Pantry_Metric": "Days"},
            {"DOP_Refrigerate_Min": 10.0}, {"DOP_Refrigerate_Max": 14.0}, {"DOP_Refrigerate_Metric": "Days"},
        ],
    ]
    payload = {
        "sheets": [
            {"name": "Category", "data": categories},
            {"name": "Product", "data": products},
        ]
    }
    target_path.write_text(json.dumps(payload), encoding="utf-8")


class TestMLPipelineVerification(unittest.TestCase):
    def test_01_python_modules_compile_cleanly(self):
        targets = list((PROJECT_ROOT / "ml" / "hybrid_vision").glob("*.py")) + [
            PROJECT_ROOT / "backend" / "app" / "model_config.py",
            PROJECT_ROOT / "backend" / "app" / "routes" / "health.py",
            PROJECT_ROOT / "backend" / "app" / "routes" / "food_recognition.py",
            PROJECT_ROOT / "backend" / "app" / "services" / "food_recognition_service.py",
            PROJECT_ROOT / "backend" / "app" / "services" / "food_form_service.py",
            PROJECT_ROOT / "backend" / "app" / "services" / "shelf_life_service.py",
            PROJECT_ROOT / "backend" / "app" / "services" / "eat_first_service.py",
        ]
        for py_file in targets:
            py_compile.compile(str(py_file), doraise=True)

    def test_02_runtime_model_status_honestly_inspects_disk_and_deps(self):
        status = get_runtime_model_status()
        self.assertIn("dependencies", status)
        self.assertIn("foodRecognition", status)
        self.assertIn("freshness", status)
        self.assertIn("shelfLife", status)
        self.assertFalse(status["shelfLife"]["isMlRegression"])

        fresh_v2_on_disk = (PROJECT_ROOT / MODEL_CONFIG["freshness"]["checkpoint_path"]).exists()
        self.assertEqual(status["freshness"]["v2CheckpointExists"], fresh_v2_on_disk)
        if not status["dependencies"]["torch"]:
            self.assertFalse(status["freshness"]["loaded"])
            self.assertEqual(status["freshness"]["status"], "UNAVAILABLE")
            self.assertFalse(status["foodRecognition"]["loaded"])

    def test_03_food_vocabulary_and_eval_images_present(self):
        vocab_path = PROJECT_ROOT / "ml" / "hybrid_vision" / "food_vocabulary.json"
        self.assertTrue(vocab_path.exists())
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        cats = data.get("categories", {})
        self.assertIn("FRUITS", cats)
        self.assertIn("VEGETABLES", cats)
        self.assertIn("BAKERY", cats)
        total_labels = sum(len(v) for v in cats.values())
        self.assertEqual(total_labels, 118)

        eval_dir = PROJECT_ROOT / "data" / "real_world_eval"
        jpgs = sorted(p.name for p in eval_dir.glob("*.jpg"))
        self.assertEqual(len(jpgs), 13)

    def test_04_food_form_regression_cases(self):
        self.assertEqual(infer_food_form("Green Chilli", freshness_label="Fresh"), "fresh")
        self.assertEqual(infer_food_form("dried red chilli"), "dried")
        self.assertNotEqual(infer_food_form("green chilli"), infer_food_form("dried red chilli"))
        self.assertIn(infer_food_form("Chili Pepper", freshness_label="Rotten"), ("fresh", "unknown"))
        self.assertEqual(infer_food_form("cut apple"), "cut")
        self.assertEqual(infer_food_form("sliced cucumber"), "sliced")
        self.assertEqual(infer_food_form("Bell Pepper"), "fresh")
        self.assertEqual(normalize_food_form("INVALID_FORM"), "unknown")

    def test_05_freshness_source_enforces_bakery_and_unsupported_unavailable_policy(self):
        src = (PROJECT_ROOT / "ml" / "hybrid_vision" / "freshness_service.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("is_freshness_supported_food", func_names)
        self.assertIn("Freshness assessment unavailable.", src)
        self.assertNotIn("Stale / Mold Suspected", src)
        self.assertNotIn("Stale (Moderate Confidence)", src)

    def test_06_shelf_life_service_with_foodkeeper_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fk_path = Path(tmpdir) / "FoodKeeper.json"
            _build_fixture_foodkeeper_json(fk_path)
            svc = FoodKeeperService(data_path=fk_path)
            self.assertTrue(svc.is_loaded)

            # Fresh produce countertop vs fridge
            apple_ct = svc.get_shelf_life_guidance("Apple", storage_type="countertop", days_stored=0, freshness_label="Fresh", food_form="fresh")
            self.assertEqual(apple_ct["status"], "available")
            self.assertEqual(apple_ct["referenceDuration"], {"minDays": 5, "maxDays": 7})
            self.assertEqual(apple_ct["remaining"], {"minDays": 5, "maxDays": 7})
            self.assertFalse(apple_ct["isMlRegression"])

            apple_fr = svc.get_shelf_life_guidance("Apple", storage_type="fridge", days_stored=0, freshness_label="Fresh", food_form="fresh")
            self.assertEqual(apple_fr["status"], "available")
            self.assertLessEqual(apple_fr["referenceDuration"]["maxDays"], 15)

            # Cut form capping
            cut_apple = svc.get_shelf_life_guidance("Apple", storage_type="countertop", days_stored=0, freshness_label="Fresh", food_form="cut")
            self.assertEqual(cut_apple["status"], "available")
            self.assertLessEqual(cut_apple["referenceDuration"]["maxDays"], 2)

            # Dried form unavailable
            dried = svc.get_shelf_life_guidance("chili pepper", storage_type="countertop", days_stored=0, freshness_label="Fresh", food_form="dried")
            self.assertEqual(dried["status"], "unavailable")
            self.assertIsNone(dried["remaining"])

            # Potato unavailable in fridge with pantry tip
            potato_fr = svc.get_shelf_life_guidance("Potato", storage_type="fridge", days_stored=0, freshness_label="Fresh", food_form="fresh")
            self.assertEqual(potato_fr["status"], "unavailable")

            # Visible freshness reduction
            spoiled = svc.get_shelf_life_guidance("Tomato", storage_type="countertop", days_stored=0, freshness_label="Slightly Spoiled", food_form="fresh")
            self.assertEqual(spoiled["status"], "available")
            self.assertLess(spoiled["remaining"]["maxDays"], spoiled["referenceDuration"]["maxDays"])

            rotten = svc.get_shelf_life_guidance("Tomato", storage_type="countertop", days_stored=0, freshness_label="Rotten", food_form="fresh")
            self.assertEqual(rotten["status"], "available")
            self.assertEqual(rotten["remaining"], {"minDays": 0, "maxDays": 0})

            # Sliced bakery bread retains FoodKeeper 4-7 day countertop window (not 1-2 day cut produce)
            sliced_bread = svc.get_shelf_life_guidance("Bread", storage_type="countertop", days_stored=0, freshness_label=None, food_form="sliced")
            self.assertEqual(sliced_bread["status"], "available")
            self.assertEqual(sliced_bread["referenceDuration"], {"minDays": 4, "maxDays": 7})
            self.assertEqual(sliced_bread["remaining"], {"minDays": 4, "maxDays": 7})
            self.assertFalse(sliced_bread["isMlRegression"])

            # Unknown food not invented
            unknown = svc.get_shelf_life_guidance("QuantumSpaghetti123", storage_type="countertop", days_stored=0)
            self.assertEqual(unknown["status"], "unavailable")
            self.assertIsNone(unknown["remaining"])

    def test_07_eat_first_deterministic_engine(self):
        ef = get_eat_first_service()
        self.assertIsInstance(ef, EatFirstService)

        unavail = ef.calculate_priority({"status": "unavailable"}, freshness_label="Fresh")
        self.assertEqual(unavail["status"], "unavailable")
        self.assertIsNone(unavail["priority"])
        self.assertEqual(unavail["score"], 0)

        vh = ef.calculate_priority({"status": "available", "remaining": {"minDays": 0, "maxDays": 0}}, freshness_label="Rotten")
        self.assertEqual(vh["priority"], "VERY_HIGH")
        self.assertEqual(vh["score"], 4)
        self.assertIn("urgencyLabel", vh)

        hi = ef.calculate_priority({"status": "available", "remaining": {"minDays": 2, "maxDays": 3}}, freshness_label="Slightly Spoiled")
        self.assertEqual(hi["priority"], "HIGH")
        self.assertEqual(hi["score"], 3)

        med = ef.calculate_priority({"status": "available", "remaining": {"minDays": 4, "maxDays": 6}}, freshness_label="Fresh")
        self.assertEqual(med["priority"], "MEDIUM")
        self.assertEqual(med["score"], 2)

        low = ef.calculate_priority({"status": "available", "remaining": {"minDays": 10, "maxDays": 14}}, freshness_label="Fresh")
        self.assertEqual(low["priority"], "LOW")
        self.assertEqual(low["score"], 1)

        ranked = ef.rank_foods_for_consumption([
            {"foodName": "Apple", "shelfLife": {"status": "available", "remaining": {"minDays": 10, "maxDays": 14}}, "freshness": None},
            {"foodName": "Tomato", "shelfLife": {"status": "available", "remaining": {"minDays": 0, "maxDays": 0}}, "freshness": {"label": "Rotten"}},
            {"foodName": "Banana", "shelfLife": {"status": "available", "remaining": {"minDays": 1, "maxDays": 2}}, "freshness": {"label": "Slightly Spoiled"}},
        ])
        self.assertEqual([r["foodName"] for r in ranked], ["Tomato", "Banana", "Apple"])


if __name__ == "__main__":
    unittest.main()

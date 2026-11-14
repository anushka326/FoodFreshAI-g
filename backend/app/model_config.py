"""
FoodFresh AI - Central Model Configuration
Defines active state and lifecycle configurations for the Master Hybrid Vision Pipeline.
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "food_recognition": {
        "status": "ACTIVE",
        "pipeline": "hybrid_pretrained_vision",
        "version": "Hybrid Pretrained V1",
        "models": {
            "detector": "IDEA-Research/grounding-dino-base",
            "semantic": "google/siglip2-base-patch16-224",
            "specialist": "ibrahimdaud/raw-food-recognition-models (ResNet-50)",
        },
        "is_loaded": True,
        "message": "Master Hybrid Vision pipeline configured."
    },
    "freshness": {
        "status": "ACTIVE",
        "model_architecture": "ResNet-18 fastai-style head",
        "model_version": "AgriFreshNET Fine-Tuned ResNet-18 (V2)",
        "checkpoint_path": "models/trained/freshness_model_v2.pth",
        "fallback_checkpoint_path": "models/pretrained/freshness_resnet18/model_weights.pth",
        "class_mapping_path": "models/pretrained/freshness_resnet18/vocab.json",
        "classes": ["fresh", "rotten", "slightly_spoiled"],
        "calibration_temperature": 1.1118282079696655,
        "is_loaded": True,
        "message": "Fine-tuned V2 is primary; V1 is fallback."
    },
    "shelf_life": {
        "status": "REFERENCE_ACTIVE_NO_ML_MODEL",
        "model_architecture": "USDA FoodKeeper lookup and explicit rules",
        "model_version": "foodkeeper_rules_v1",
        "checkpoint_path": None,
        "data_path": "data/raw/foodkeeper/FoodKeeper.json",
        "is_loaded": True,
        "is_ml_regression": False,
        "message": "FoodKeeper baseline is reference-based; no trained shelf-life regression model is integrated."
    }
}


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def get_runtime_model_status() -> Dict[str, Any]:
    """Inspect real runtime dependencies and checkpoint files on disk honestly."""
    torch_ok = _has_module("torch")
    transformers_ok = _has_module("transformers")
    pil_ok = _has_module("PIL")
    ml_runtime_ok = torch_ok and transformers_ok and pil_ok

    dino_local = (PROJECT_ROOT / "models" / "pretrained" / "grounding_dino").exists()
    siglip_local = (PROJECT_ROOT / "models" / "pretrained" / "siglip2").exists()
    raw_weights = (PROJECT_ROOT / "models" / "pretrained" / "raw_food_resnet50" / "resnet50_pytorch_model.bin").exists()
    raw_classes = (PROJECT_ROOT / "models" / "pretrained" / "raw_food_resnet50" / "classes.json").exists()

    fresh_v2_exists = (PROJECT_ROOT / MODEL_CONFIG["freshness"]["checkpoint_path"]).exists()
    fresh_v1_exists = (PROJECT_ROOT / MODEL_CONFIG["freshness"]["fallback_checkpoint_path"]).exists()
    fresh_vocab_exists = (PROJECT_ROOT / MODEL_CONFIG["freshness"]["class_mapping_path"]).exists()
    foodkeeper_exists = (PROJECT_ROOT / MODEL_CONFIG["shelf_life"]["data_path"]).exists()

    fr_loaded = ml_runtime_ok and (raw_weights and raw_classes or dino_local or siglip_local)
    fresh_loaded = torch_ok and pil_ok and (fresh_v2_exists or fresh_v1_exists)

    return {
        "dependencies": {
            "torch": torch_ok,
            "transformers": transformers_ok,
            "pillow": pil_ok,
        },
        "foodRecognition": {
            "status": "ACTIVE" if fr_loaded else "UNAVAILABLE",
            "loaded": fr_loaded,
            "pipeline": MODEL_CONFIG["food_recognition"]["pipeline"],
            "detector": {
                "repo": "IDEA-Research/grounding-dino-base",
                "localCheckpointExists": dino_local,
                "loadable": ml_runtime_ok,
            },
            "foodSemantic": {
                "repo": "google/siglip2-base-patch16-224",
                "localCheckpointExists": siglip_local,
                "loadable": ml_runtime_ok,
            },
            "foodSpecialist": {
                "repo": "ibrahimdaud/raw-food-recognition-models",
                "weightsExist": raw_weights,
                "classesExist": raw_classes,
                "loadable": torch_ok and pil_ok and raw_weights and raw_classes,
            },
        },
        "freshness": {
            "status": "ACTIVE" if fresh_loaded else "UNAVAILABLE",
            "loaded": fresh_loaded,
            "version": MODEL_CONFIG["freshness"]["model_version"],
            "checkpoint": MODEL_CONFIG["freshness"]["checkpoint_path"],
            "v2CheckpointExists": fresh_v2_exists,
            "v1CheckpointExists": fresh_v1_exists,
            "vocabExists": fresh_vocab_exists,
            "calibrationTemperature": MODEL_CONFIG["freshness"]["calibration_temperature"],
        },
        "shelfLife": {
            "status": "REFERENCE_ACTIVE_NO_ML_MODEL" if foodkeeper_exists else "REFERENCE_DATA_MISSING",
            "loaded": foodkeeper_exists,
            "source": "USDA FoodKeeper",
            "dataPath": MODEL_CONFIG["shelf_life"]["data_path"],
            "isMlRegression": False,
        },
    }


try:
    from backend.app.dataset_registry import DATASET_REGISTRY
except ImportError:
    DATASET_REGISTRY = {}


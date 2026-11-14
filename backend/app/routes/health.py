from fastapi import APIRouter
from backend.app.model_config import MODEL_CONFIG, get_runtime_model_status

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health_check():
    runtime = get_runtime_model_status()
    fr_rt = runtime["foodRecognition"]
    fresh_rt = runtime["freshness"]
    shelf_rt = runtime["shelfLife"]

    return {
        "status": "ok",
        "service": "FoodFresh AI backend",
        "pipeline": "Master Hybrid Vision",
        "dependencies": runtime["dependencies"],
        "models": {
            "foodRecognition": {
                "status": fr_rt["status"],
                "loaded": fr_rt["loaded"],
                "pipeline": fr_rt["pipeline"],
                "detector": "IDEA-Research/grounding-dino-base",
                "foodSemantic": "google/siglip2-base-patch16-224",
                "foodSpecialist": "ibrahimdaud/raw-food-recognition-models",
                "details": fr_rt,
            },
            "freshness": {
                "status": fresh_rt["status"],
                "loaded": fresh_rt["loaded"],
                "version": fresh_rt["version"],
                "checkpoint": fresh_rt["checkpoint"],
                "v2CheckpointExists": fresh_rt["v2CheckpointExists"],
                "v1CheckpointExists": fresh_rt["v1CheckpointExists"],
                "calibrationTemperature": fresh_rt["calibrationTemperature"],
            },
            "shelfLife": {
                "status": shelf_rt["status"],
                "loaded": shelf_rt["loaded"],
                "source": shelf_rt["source"],
                "dataPath": shelf_rt["dataPath"],
                "isMlRegression": False,
                "checkpoint": None,
            },
            "eatFirstPriority": {
                "status": "ACTIVE",
                "loaded": True,
                "engine": "Deterministic Priority Rules",
            },
        },
        "foodRecognitionModel": {
            "version": MODEL_CONFIG["food_recognition"]["version"],
            "detector": "grounding-dino-base",
            "semantic": "siglip2-base-patch16-224",
            "specialist": "raw-food-resnet50",
            "loaded": fr_rt["loaded"],
            "status": fr_rt["status"],
        },
    }

import base64
import io
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model store
models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading models on {device}...")

    models["device"] = device
    models["yolo"] = YOLO("yolo11n.pt")
    models["embedder"] = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    logger.info("Loading ZoeDepth...")
    depth_processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
    depth_model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").eval()
    depth_model.to(device)
    models["depth_processor"] = depth_processor
    models["depth_model"] = depth_model

    logger.info("All models ready.")
    yield
    models.clear()


app = FastAPI(title="Smart Voice Navigator — Model Server", lifespan=lifespan)


# ── Schemas ──────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    image_b64: str
    target: str

class DetectResponse(BaseModel):
    found: bool
    best_match: str | None = None
    score: float | None = None
    box: list[float] | None = None
    detected_objects: list[str] = []
    message: str

class DepthRequest(BaseModel):
    image_b64: str
    box: list[float]   # [x1, y1, x2, y2]
    object_name: str

class DepthResponse(BaseModel):
    distance_meters: float | None = None
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(image_b64: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(models)}


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    pil_image = _decode_image(req.image_b64)

    yolo: YOLO = models["yolo"]
    embedder: SentenceTransformer = models["embedder"]

    results = yolo(pil_image)
    detected_objects: list[tuple[str, list[float]]] = []
    for result in results:
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id]
                coords: list[float] = box.xyxy[0].cpu().numpy().tolist()
                detected_objects.append((name, coords))

    names_only = [o[0] for o in detected_objects]
    logger.info(f"Detected: {names_only}")

    if not names_only:
        return DetectResponse(found=False, detected_objects=[], message="No objects detected.")

    q_emb = embedder.encode(req.target, convert_to_tensor=True)
    p_embs = embedder.encode(names_only, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, p_embs)[0]

    best_idx = int(sims.argmax())
    best_match = names_only[best_idx]
    best_score = float(sims[best_idx])
    best_box = detected_objects[best_idx][1]

    logger.info(f"Best match: {best_match} (score={best_score:.3f})")

    if best_score >= 0.5:
        return DetectResponse(
            found=True, best_match=best_match, score=best_score,
            box=best_box, detected_objects=names_only,
            message=f"Found {best_match} matching your {req.target}.",
        )
    return DetectResponse(
        found=False, best_match=best_match, score=best_score,
        box=None, detected_objects=names_only,
        message=f"No confident match for {req.target}. Spotted: {', '.join(names_only[:3])}.",
    )


@app.post("/depth", response_model=DepthResponse)
async def depth(req: DepthRequest):
    pil_image = _decode_image(req.image_b64)

    processor = models["depth_processor"]
    depth_model = models["depth_model"]
    device = models["device"]

    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)

    post = processor.post_process_depth_estimation(
        outputs, source_sizes=[(pil_image.height, pil_image.width)]
    )[0]
    depth_map = post["predicted_depth"].cpu().numpy()

    x1, y1, x2, y2 = map(int, req.box)
    x1, x2 = max(0, x1), min(pil_image.width, x2)
    y1, y2 = max(0, y1), min(pil_image.height, y2)
    region = depth_map[y1:y2, x1:x2]

    if region.size == 0:
        return DepthResponse(message=f"Cannot get a clear depth reading for {req.object_name}.")

    distance = float(np.median(region))
    logger.info(f"Distance for {req.object_name}: {distance:.2f}m")
    return DepthResponse(
        distance_meters=distance,
        message=f"The {req.object_name} is about {distance:.2f} meters away. Close in carefully.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

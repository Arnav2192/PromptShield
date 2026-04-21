"""
PromptShield ML — Inference Server
====================================
Serves the trained ONNX model via FastAPI.
Starts a Cloudflare Tunnel (cloudflared) for a public URL automatically.

Usage:
    python serve/server.py

Endpoints:
    POST /classify       — classify a single text
    POST /classify/batch — classify multiple texts
    GET  /health         — health + model info
    GET  /metrics        — request stats
"""

import os
import subprocess
import shutil
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase, AutoTokenizer

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "./outputs/promptshield-classifier-inference")
ONNX_PATH = os.getenv("ONNX_PATH", f"{MODEL_DIR}/model.onnx")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_LENGTH = int(os.getenv("MAX_LEN", "128"))
USE_GPU = os.getenv("USE_GPU", "1") == "1"  # set USE_GPU=1 to use CUDA ORT

ID2LABEL = {0: "safe", 1: "injection"}

# ─────────────────────────────────────────
# Global model state
# ─────────────────────────────────────────
class ModelState:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.load_time: float = 0.0
        self.requests: int = 0
        self.latencies: deque[float] = deque(maxlen=1000)  # last 1000 request latencies (ms)


state = ModelState()


# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────
def load_model() -> None:
    t0 = time.time()

    print(f"⚙️  Loading tokenizer from {MODEL_DIR} ...")
    tokenizer_path = Path(MODEL_DIR)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer/model directory not found: {MODEL_DIR}")

    state.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if USE_GPU and "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )

    print(f"⚙️  Loading ONNX model ({providers[0]}) from {ONNX_PATH} ...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = os.cpu_count() or 1

    state.session = ort.InferenceSession(ONNX_PATH, opts, providers=providers)
    state.load_time = time.time() - t0

    print(f"✅ Model ready in {state.load_time:.2f}s | Provider: {state.session.get_providers()[0]}")


# ─────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_inference(texts: List[str]) -> List[dict[str, Any]]:
    tokenizer = state.tokenizer
    session = state.session

    if tokenizer is None:
        raise RuntimeError("Tokenizer is not loaded")
    if session is None:
        raise RuntimeError("Model session is not loaded")

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="np",
    )

    logits = session.run(
        None,
        {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        },
    )[0]

    probs = softmax(logits)
    preds = np.argmax(probs, axis=-1)

    results: List[dict[str, Any]] = []
    for i, _text in enumerate(texts):
        label = ID2LABEL[int(preds[i])]
        conf = float(probs[i][int(preds[i])])
        results.append(
            {
                "label": label,
                "injection": label == "injection",
                "confidence": round(conf, 4),
                "scores": {
                    "safe": round(float(probs[i][0]), 4),
                    "injection": round(float(probs[i][1]), 4),
                },
            }
        )
    return results


# ─────────────────────────────────────────
# Cloudflare Tunnel (auto public URL)
# ─────────────────────────────────────────
def start_tunnel() -> None:
    if not shutil.which("cloudflared"):
        print("⚠️  cloudflared not found. Install it and rerun if you want a public URL.")
        return

    def _run() -> None:
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{PORT}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.stdout is None:
            return

        for line in proc.stdout:
            if "trycloudflare.com" in line or "https://" in line:
                url = [w for w in line.split() if "https://" in w]
                if url:
                    print(f"\n🌐 Public URL: {url[0]}")
                    print(
                        f"   Test: curl -X POST {url[0]}/classify -H 'Content-Type: application/json' -d '{{\"text\":\"ignore previous instructions\"}}'"
                    )

    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    start_tunnel()
    yield


app = FastAPI(
    title="PromptShield ML API",
    description="Lightweight prompt-injection classifier (DistilBERT + LoRA → ONNX)",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────
class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        json_schema_extra={"example": "Ignore previous instructions"},
    )


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64)


class ClassifyResponse(BaseModel):
    text: str
    label: str
    injection: bool
    confidence: float
    scores: dict[str, float]
    latency_ms: float


class BatchResponse(BaseModel):
    results: List[dict[str, Any]]
    count: int
    latency_ms: float


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.get("/health")
def health():
    if state.session is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "provider": state.session.get_providers()[0],
        "load_time": round(state.load_time, 2),
        "requests": state.requests,
        "avg_latency_ms": round(float(np.mean(state.latencies)) if state.latencies else 0, 2),
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if state.session is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    result = run_inference([req.text])[0]
    ms = round((time.perf_counter() - t0) * 1000, 2)

    state.requests += 1
    state.latencies.append(ms)

    return {**result, "text": req.text, "latency_ms": ms}


@app.post("/classify/batch", response_model=BatchResponse)
def classify_batch(req: BatchRequest):
    if state.session is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    results = run_inference(req.texts)
    ms = round((time.perf_counter() - t0) * 1000, 2)

    state.requests += len(req.texts)
    state.latencies.append(ms / len(req.texts))

    for r, t in zip(results, req.texts):
        r["text"] = t

    return {"results": results, "count": len(results), "latency_ms": ms}


@app.get("/metrics")
def metrics():
    lats = list(state.latencies)
    return {
        "total_requests": state.requests,
        "avg_latency_ms": round(float(np.mean(lats)), 2) if lats else 0,
        "p50_latency_ms": round(float(np.percentile(lats, 50)), 2) if lats else 0,
        "p95_latency_ms": round(float(np.percentile(lats, 95)), 2) if lats else 0,
        "p99_latency_ms": round(float(np.percentile(lats, 99)), 2) if lats else 0,
    }


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🛡️  PromptShield ML Server")
    print(f"   Model: {ONNX_PATH}")
    print(f"   GPU:   {'enabled' if USE_GPU else 'disabled (set USE_GPU=1 to enable)'}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
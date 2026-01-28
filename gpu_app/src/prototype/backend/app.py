# Entry point for the Flask applicationwith routes

from flask import Flask, jsonify
from config import * 
from pipeline import run_pipeline
from tree_segmentation_service import TreeSegmentationService
from threading import Lock
import logging


app = Flask(__name__)
gpu_lock = Lock()

service = TreeSegmentationService(GPU_ID, MODEL_TYPE)

@app.route("/process", methods=["POST"])
def process():
    with gpu_lock:
        num_objects = run_pipeline(service)
    return jsonify({
        "status": "ok",
        "objects": num_objects
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready",
        "gpu": service.device_id,
        "free_mem_mb": int(service.free_mem / 1024**2)
    })


# start flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

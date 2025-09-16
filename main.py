# main.py
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, storage
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import logging
import tempfile
from datetime import datetime
import threading
import time

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

db = None
bucket = None
model = None
class_names: list[str] = []
model_version: str | None = None
metadata: dict = {}
model_loading = False
model_loaded = False

def initialize_firebase():
	global db, bucket

	logger.info("=== FIREBASE INITIALIZATION START ===")
	firebase_json = os.environ.get("FIREBASE_JSON")
	if not firebase_json:
		logger.error("FIREBASE_JSON environment variable not found")
		return

	try:
		logger.info("FIREBASE_JSON found, parsing credentials...")
		cred_dict = json.loads(firebase_json)
		cred = credentials.Certificate(cred_dict)

		storage_bucket = "palayan-app.firebasestorage.app"

		if not firebase_admin._apps:
			logger.info("Initializing Firebase app...")
			firebase_admin.initialize_app(cred, {"storageBucket": storage_bucket})

		logger.info("Getting Firestore and Storage clients...")
		db = firestore.client()
		bucket = storage.bucket()
		logger.info("Firebase initialized (bucket=%s)", storage_bucket)
	except Exception as e:
		logger.error(f"Firebase initialization failed: {e}", exc_info=True)

def load_model_from_firebase() -> bool:
	global model, class_names, model_version, metadata, model_loaded

	if not bucket:
		logger.error("Firebase not initialized; cannot load model.")
		return False

	try:
		model_blob = bucket.blob("models/rice_disease_model.h5")
		classes_blob = bucket.blob("models/rice_disease_classes.json")
		metadata_blob = bucket.blob("models/rice_disease_metadata.json")

		missing = []
		if not model_blob.exists():
			missing.append("models/rice_disease_model.h5")
		if not classes_blob.exists():
			missing.append("models/rice_disease_classes.json")
		if missing:
			logger.warning("Missing required files in Storage: %s", ", ".join(missing))
			return False

		with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f_m:
			tmp_model_path = f_m.name
			model_blob.download_to_filename(tmp_model_path)
		with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_c:
			tmp_classes_path = f_c.name
			classes_blob.download_to_filename(tmp_classes_path)

		logger.info("Loading model from Storage path: models/rice_disease_model.h5")
		model = tf.keras.models.load_model(tmp_model_path)
		with open(tmp_classes_path, "r", encoding="utf-8") as f:
			class_names = json.load(f)

		metadata = {}
		tmp_md_path = None
		if metadata_blob.exists():
			with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_md:
				tmp_md_path = f_md.name
				metadata_blob.download_to_filename(tmp_md_path)
			with open(tmp_md_path, "r", encoding="utf-8") as f:
				metadata = json.load(f)

		try:
			if db is not None:
				doc = db.collection("model_info").document("rice_disease_classifier").get()
				model_version = doc.to_dict().get("version", "unknown") if doc.exists else "unknown"
			else:
				model_version = "unknown"
		except Exception:
			model_version = "unknown"

		for p in [tmp_model_path, tmp_classes_path, tmp_md_path]:
			if p:
				try:
					os.unlink(p)
				except Exception:
					pass

		model_loaded = True
		logger.info("Model loaded successfully - %d classes, version: %s", len(class_names), model_version)
		return True
	except Exception as e:
		logger.error(f"Error loading model: {e}", exc_info=True)
		return False

def get_disease_info(disease_name: str) -> dict:
	if not metadata:
		return {
			"scientific_name": "Unknown",
			"description": f"Information about {disease_name.replace('_', ' ')} not available.",
			"symptoms": ["Symptoms not specified"],
			"cause": "Cause not specified",
			"treatments": ["Treatment information not available"],
		}

	if disease_name in metadata:
		return metadata[disease_name]

	clean = disease_name.replace("_", " ").lower()
	for key, val in metadata.items():
		if key.replace("_", " ").lower() == clean:
			return val

	return {
		"scientific_name": "Unknown",
		"description": f"Information about {disease_name.replace('_', ' ')} not available.",
		"symptoms": ["Symptoms not specified"],
		"cause": "Cause not specified",
		"treatments": ["Treatment information not available"],
	}

#prediction
@app.route("/predict_disease", methods=["POST"])
def predict():
    global model, class_names

    if not model_loaded:
        return jsonify({"status": "error", "message": "Model is still loading. Please try again in a few seconds."}), 503

    try:
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No image selected"}), 400

        # Load + normalize
        image = Image.open(file.stream)
        if image.mode != "RGB":
            image = image.convert("RGB")
        # EXIF orientation fix
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        # Quick heuristic: green-dominant pixel ratio (pre-resize)
        thumb = image.copy()
        thumb.thumbnail((224, 224))
        arr = np.asarray(thumb, dtype=np.uint8)
        r, g, b = arr[..., 0].astype(np.int32), arr[..., 1].astype(np.int32), arr[..., 2].astype(np.int32)
        green_dom = (g > r) & (g > b)
        green_ratio = float(np.mean(green_dom))

        # Model input
        image = image.resize((224, 224))
        image_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

        preds = model.predict(image_array, verbose=0)
        probs = preds[0].astype(float)

        # Generalized multi-class metrics
        sorted_idx = np.argsort(probs)[::-1]
        p1 = float(probs[sorted_idx[0]])
        p2 = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
        label = class_names[int(sorted_idx[0])]
        conf = p1
        margin12 = p1 - p2

        K = int(os.getenv("TOPK", "3"))
        top_candidates = [
            {"label": class_names[int(i)], "confidence": float(probs[i])}
            for i in sorted_idx[:K]
        ]
        topk_sum = float(np.sum(probs[sorted_idx[:K]]))

        # Entropy (normalized 0..1)
        eps = 1e-12
        entropy = float(-np.sum(probs * np.log(probs + eps)))
        num_classes = max(len(class_names), 2)
        norm_entropy = float(entropy / np.log(num_classes))

        # Thresholds
        CONF_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.70"))
        MARGIN_THRESHOLD = float(os.getenv("PREDICTION_MARGIN_THRESHOLD", "0.12"))
        TOPK_SUM_THRESHOLD = float(os.getenv("TOPK_SUM_THRESHOLD", "0.80"))
        ENTROPY_THRESHOLD = float(os.getenv("ENTROPY_THRESHOLD", "0.60"))
        GREEN_THRESHOLD = float(os.getenv("GREEN_RATIO_THRESHOLD", "0.22"))

        logger.info(
            "predict: label=%s conf=%.3f margin=%.3f topk_sum=%.3f H=%.3f green=%.3f thr=(%.2f,%.2f,%.2f,%.2f,%.2f)",
            label, conf, margin12, topk_sum, norm_entropy, green_ratio,
            CONF_THRESHOLD, MARGIN_THRESHOLD, TOPK_SUM_THRESHOLD, ENTROPY_THRESHOLD, GREEN_THRESHOLD
        )

        # Rejection rule: likely NOT a rice leaf
        is_ood_not_rice = (
            (conf < CONF_THRESHOLD) or
            (margin12 < MARGIN_THRESHOLD) or
            (topk_sum < TOPK_SUM_THRESHOLD) or
            (norm_entropy > ENTROPY_THRESHOLD) or
            (green_ratio < GREEN_THRESHOLD)
        )

        if is_ood_not_rice:
            return jsonify({
                "status": "not_rice_leaf",
                "message": "The image does not appear to be a rice leaf. Please retake a clearer, closer photo of a rice leaf.",
                "predicted_disease": "N/A",
                "confidence": conf,
                "is_confident": False,
                "threshold": CONF_THRESHOLD,
                "green_ratio": green_ratio,
                "top_candidates": top_candidates,
                "disease_info": {
                    "scientific_name": "N/A",
                    "description": "N/A",
                    "symptoms": ["N/A"],
                    "cause": "N/A",
                    "treatments": ["N/A"],
                },
                "all_predictions": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
                "model_version": model_version,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }), 200

        # Otherwise, proceed as normal
        info = get_disease_info(label)
        all_preds = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        all_sorted = dict(sorted(all_preds.items(), key=lambda x: x[1], reverse=True))

        return jsonify({
            "status": "success",
            "message": None,
            "predicted_disease": label,
            "confidence": conf,
            "is_confident": True,
            "threshold": CONF_THRESHOLD,
            "green_ratio": green_ratio,
            "top_candidates": top_candidates,
            "disease_info": info,
            "all_predictions": all_sorted,
            "model_version": model_version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"}), 500
		
@app.route("/reload_model", methods=["POST"])
def reload_model():
	try:
		logger.info("Reloading model...")
		ok = load_model_from_firebase()
		if ok:
			return jsonify({"status": "success", "message": "Model reloaded successfully", "classes": len(class_names), "version": model_version})
		return jsonify({"status": "error", "message": "Failed to reload model"}), 500
	except Exception as e:
		logger.error(f"Error reloading model: {e}", exc_info=True)
		return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/model_info", methods=["GET"])
def model_info():
	return jsonify({
		"model_loaded": model_loaded,
		"num_classes": len(class_names) if class_names else 0,
		"classes": class_names,
		"model_version": model_version,
		"timestamp": datetime.utcnow().isoformat() + "Z",
	})

@app.route("/health", methods=["GET"])
def health():
	# Always return 200, even if model is still loading
	return jsonify({
		"status": "healthy",
		"model_loaded": model_loaded,
		"firebase_connected": db is not None,
		"timestamp": datetime.utcnow().isoformat() + "Z",
	}), 200

@app.route("/", methods=["GET"])
def root():
	return health()

@app.route("/diseases", methods=["GET"])
def get_diseases():
	if db is None:
		return jsonify({"status": "error", "message": "Firebase not initialized"}), 500
	try:
		docs = db.collection("rice_local_diseases").stream()
		items = []
		for d in docs:
			obj = d.to_dict()
			obj["id"] = d.id
			items.append(obj)
		return jsonify({"status": "success", "count": len(items), "diseases": items})
	except Exception as e:
		logger.error(f"Error fetching diseases: {e}", exc_info=True)
		return jsonify({"status": "error", "message": str(e)}), 500

def initialize_app():
	try:
		logger.info("Starting Rice Disease Prediction API...")
		
		# Initialize Firebase first
		initialize_firebase()
		
		# Load model in background thread
		def load_model_async():
			global model_loading
			model_loading = True
			logger.info("Loading model from Firebase...")
			load_model_from_firebase()
			model_loading = False
		
		# Start model loading in background
		model_thread = threading.Thread(target=load_model_async, daemon=True)
		model_thread.start()
		
		# Give the thread a moment to start
		time.sleep(0.1)
		
		return True
	except Exception as e:
		logger.error(f"Failed to initialize application: {e}", exc_info=True)
		return False

# Initialize app
initialize_app()

if __name__ == "__main__":
	port = int(os.environ.get("PORT", "5000"))
	logger.info(f"Server starting on port {port}")
	app.run(host="0.0.0.0", port=port, debug=False)




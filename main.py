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
import zipfile
import shutil

# Use channels_last and log versions
tf.keras.backend.set_image_data_format("channels_last")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("TF=%s Keras=%s", tf.__version__, tf.keras.__version__)

# EfficientNet preprocessing (must match training)
from tensorflow.keras.applications.efficientnet import preprocess_input

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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

def _blob_from_gs(gs_url: str):
	if not gs_url or not isinstance(gs_url, str):
		return None
	prefix = f"gs://{bucket.name}/"
	if not gs_url.startswith(prefix):
		return None
	path = gs_url[len(prefix):]
	return bucket.blob(path)

def _build_serving_model(num_classes: int) -> tf.keras.Model:
	inp = tf.keras.Input((224, 224, 3))
	# IMPORTANT: don't load imagenet here; we'll load our own weights later
	base = tf.keras.applications.EfficientNetB0(
		include_top=False,
		weights=None,
		input_tensor=inp
	)
	base.trainable = False
	x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(512, activation="relu")(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(256, activation="relu")(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dropout(0.4)(x)
	out = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
	return tf.keras.Model(inp, out)

def load_model_from_firebase() -> bool:
	global model, class_names, model_version, metadata, model_loaded

	if not bucket:
		logger.error("Firebase not initialized; cannot load model.")
		return False

	try:
		# Try to read model paths from Firestore; fall back to defaults
		doc = None
		info = {}
		try:
			doc = db.collection("model_info").document("rice_disease_classifier").get()
			info = doc.to_dict() if doc and doc.exists else {}
		except Exception:
			info = {}

		# Prefer SavedModel ZIP if provided
		saved_zip_url = info.get("savedmodel_zip_url", "")
		h5_url = info.get("model_url", "gs://palayan-app.firebasestorage.app/models/rice_disease_model.h5") if info is not None else "gs://palayan-app.firebasestorage.app/models/rice_disease_model.h5"
		classes_url = info.get("classes_url", "gs://palayan-app.firebasestorage.app/models/rice_disease_classes.json")
		metadata_url = info.get("metadata_url", "gs://palayan-app.firebasestorage.app/models/rice_disease_metadata.json")

		saved_zip_blob = _blob_from_gs(saved_zip_url) if saved_zip_url else None
		model_blob = _blob_from_gs(h5_url)
		classes_blob = _blob_from_gs(classes_url)
		metadata_blob = _blob_from_gs(metadata_url)

		missing = []
		# We require at least classes.json
		if classes_blob is None or not classes_blob.exists():
			missing.append("models/rice_disease_classes.json")
		if (saved_zip_blob is None or not saved_zip_blob.exists()) and (model_blob is None or not model_blob.exists()):
			missing.append("model (SavedModel ZIP or H5)")
		if missing:
			logger.warning("Missing required files in Storage: %s", ", ".join(missing))
			return False

		tmp_paths = []

		# Load class names first (needed for H5 fallback rebuild)
		with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_c:
			tmp_classes_path = f_c.name
		tmp_paths.append(tmp_classes_path)
		classes_blob.download_to_filename(tmp_classes_path)
		with open(tmp_classes_path, "r", encoding="utf-8") as f:
			class_names_local = json.load(f)
		class_names.clear()
		class_names.extend(class_names_local)

		# Try SavedModel ZIP first for robustness
		loaded = False
		if saved_zip_blob and saved_zip_blob.exists():
			with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as fz:
				tmp_zip = fz.name
			tmp_paths.append(tmp_zip)
			saved_zip_blob.download_to_filename(tmp_zip)

			tmp_dir = tempfile.mkdtemp()
			with zipfile.ZipFile(tmp_zip, "r") as zf:
				zf.extractall(tmp_dir)
			try:
				logger.info("Loading SavedModel directory from ZIP...")
				model_local = tf.keras.models.load_model(tmp_dir, compile=False)
				model = model_local
				loaded = True
				tmp_paths.append(tmp_dir)
			except Exception as e:
				logger.warning("SavedModel load failed, will try H5: %s", e)
				try:
					shutil.rmtree(tmp_dir, ignore_errors=True)
				except Exception:
					pass

		# Fallback: H5 load (robust path)
		if not loaded and model_blob and model_blob.exists():
			with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f_m:
				tmp_model_path = f_m.name
			tmp_paths.append(tmp_model_path)
			logger.info("Loading model from Storage path: models/rice_disease_model.h5")
			model_blob.download_to_filename(tmp_model_path)
			try:
				# Fast path: deserialize full model
				model_local = tf.keras.models.load_model(tmp_model_path, compile=False)
				model = model_local
			except ValueError as e:
				msg = str(e)
				if "Shape mismatch" in msg and "stem_conv/kernel" in msg:
					logger.warning("H5 mismatch; rebuilding 3-channel model and loading weights with skip_mismatch.")
					rebuilt = _build_serving_model(len(class_names))
					# Load weights by name, allow minor shape diffs
					rebuilt.load_weights(tmp_model_path, by_name=True, skip_mismatch=True)
					model = rebuilt
				else:
					raise

		# Optional metadata
		metadata.clear()
		if metadata_blob and metadata_blob.exists():
			with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_md:
				tmp_md_path = f_md.name
			tmp_paths.append(tmp_md_path)
			metadata_blob.download_to_filename(tmp_md_path)
			with open(tmp_md_path, "r", encoding="utf-8") as f:
				metadata.update(json.load(f))

		# Version field
		try:
			if doc and doc.exists:
				model_version_val = info.get("version", "unknown")
				model_version_str = str(model_version_val) if model_version_val is not None else "unknown"
				globals()["model_version"] = model_version_str
			else:
				globals()["model_version"] = "unknown"
		except Exception:
			globals()["model_version"] = "unknown"

		# Cleanup temps
		for p in tmp_paths:
			try:
				if os.path.isdir(p):
					shutil.rmtree(p, ignore_errors=True)
				elif os.path.isfile(p):
					os.unlink(p)
			except Exception:
				pass

		globals()["model_loaded"] = True
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

		# Green-dominant pixel ratio (heuristic; independent from model input)
		thumb = image.copy()
		thumb.thumbnail((224, 224))
		arr = np.asarray(thumb, dtype=np.uint8)
		r, g, b = arr[..., 0].astype(np.int32), arr[..., 1].astype(np.int32), arr[..., 2].astype(np.int32)
		green_dom = (g > r) & (g > b)
		green_ratio = float(np.mean(green_dom))

		# Model input (must match EfficientNet training preprocessing)
		image = image.resize((224, 224))
		image_array = np.array(image, dtype=np.float32)
		image_array = preprocess_input(image_array)
		image_array = np.expand_dims(image_array, axis=0)

		preds = model.predict(image_array, verbose=0)
		probs = preds[0].astype(float)

		# Multi-class metrics
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

		# Thresholds (env-overridable)
		CONF_THRESHOLD   = get_env_float("PREDICTION_THRESHOLD", "0.50")
		MARGIN_THRESHOLD = get_env_float("PREDICTION_MARGIN_THRESHOLD", "0.05")
		TOPK_SUM_THRESHOLD = get_env_float("TOPK_SUM_THRESHOLD", "0.60")
		ENTROPY_THRESHOLD  = get_env_float("ENTROPY_THRESHOLD", "0.98")
		GREEN_THRESHOLD    = get_env_float("GREEN_RATIO_THRESHOLD", "0.05")

		logger.info("env-read: PRED=%r MARGIN=%r", os.getenv("PREDICTION_THRESHOLD"), os.getenv("PREDICTION_MARGIN_THRESHOLD"))
		
		logger.info(
			"predict: label=%s conf=%.3f margin=%.3f topk_sum=%.3f H=%.3f green=%.3f thr=(%.2f,%.2f,%.2f,%.2f,%.2f)",
			label, conf, margin12, topk_sum, norm_entropy, green_ratio,
			CONF_THRESHOLD, MARGIN_THRESHOLD, TOPK_SUM_THRESHOLD, ENTROPY_THRESHOLD, GREEN_THRESHOLD
		)

		# OOD/reject rule
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
				"message": "Maling kuha, hindi dahon ng palay. Subukan mong picturan ulit yung dahon.",
				"predicted_disease": "Maling kuha, hindi dahon ng palay. Subukan mong picturan ulit yung dahon.",
				"confidence": conf,
				"is_confident": False,
				"threshold": CONF_THRESHOLD,
				"green_ratio": green_ratio,
				"top_candidates": top_candidates,
				"disease_info": {
					"scientific_name": "Maling kuha, hindi dahon ng palay. Subukan mong picturan ulit yung dahon.",
					"description": "Maling kuha, hindi dahon ng palay. Subukan mong picturan ulit yung dahon.",
					"symptoms": ["Maling kuha, hindi dahon ng palay. Subukan mong picturan ulit yung dahon."],
					"cause": "Maling kuha, hindi dahon ng palay. Subukan mong picturan ulit yung dahon.",
					"treatments": ["Maling kuha, hindi dahon ng palay. Subukan mong picturan ulit yung dahon."],
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
		initialize_firebase()

		# Load model in background thread
		def load_model_async():
			global model_loading
			model_loading = True
			logger.info("Loading model from Firebase...")
			load_model_from_firebase()
			model_loading = False

		model_thread = threading.Thread(target=load_model_async, daemon=True)
		model_thread.start()

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



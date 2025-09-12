#Railway API
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, storage
import tensorflow as tf
import numpy as np
from PIL import Image
import json, os, logging
import tempfile
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Firebase initialization
try:
    logger.info("=== FIREBASE INITIALIZATION START ===")
    
    # Check if environment variable exists
    firebase_json = os.environ.get('FIREBASE_JSON')
    if not firebase_json:
        logger.error("FIREBASE_JSON environment variable not found")
        db = None
        bucket = None
    else:
        logger.info("FIREBASE_JSON found, parsing credentials...")
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
        
        if not firebase_admin._apps:
            logger.info("Initializing Firebase app...")
            firebase_admin.initialize_app(cred, {
                'storageBucket': "palayan-app.firebasestorage.app"
            })
        
        logger.info("Getting Firestore and Storage clients...")
        db = firestore.client()
        bucket = storage.bucket()
        logger.info("Firebase initialized successfully")
        
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")
    import traceback
    traceback.print_exc()
    db = None
    bucket = None

# Globals
model = None
class_names = []
model_version = None
metadata = {}

#load the model from firebase
def load_model_from_firebase():
    """Load model from Firebase Storage"""
    global model, class_names, model_version, metadata
    
    if not bucket:
        logger.error("Firebase not initialized")
        return False
    
    try:
        # Check if model exists
        model_blob = bucket.blob("models/rice_disease_model.h5")
        classes_blob = bucket.blob("models/rice_disease_classes.json")
        metadata_blob = bucket.blob("models/rice_disease_metadata.json")
        
        if not model_blob.exists():
            logger.warning("No model found in Firebase Storage")
            return False
        
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as model_file:
            model_path = model_file.name
            model_blob.download_to_filename(model_path)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as classes_file:
            classes_path = classes_file.name
            classes_blob.download_to_filename(classes_path)
        
        # Load model and classes
        model = tf.keras.models.load_model(model_path)
        with open(classes_path, "r") as f:
            class_names = json.load(f)
        
        # Load metadata if exists
        if metadata_blob.exists():
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as metadata_file:
                metadata_path = metadata_file.name
                metadata_blob.download_to_filename(metadata_path)
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
        
        # Get model version from Firestore
        try:
            model_doc = db.collection('model_info').document('rice_disease_classifier').get()
            if model_doc.exists:
                model_version = model_doc.to_dict().get('version', 'unknown')
        except:
            model_version = 'unknown'
        
        # Cleanup temp files
        os.unlink(model_path)
        os.unlink(classes_path)
        if metadata_blob.exists():
            os.unlink(metadata_path)
        
        logger.info(f"Model loaded successfully - {len(class_names)} classes, version: {model_version}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def get_disease_info(disease_name):
    """Get detailed information about a disease"""
    clean_name = disease_name.replace('_', ' ')
    
    # Try exact match first
    if disease_name in metadata:
        return metadata[disease_name]
    
    # Try cleaned name match
    for key, value in metadata.items():
        if key.replace('_', ' ').lower() == clean_name.lower():
            return value
    
    # Default response if not found
    return {
        'scientific_name': 'Unknown',
        'description': f'Information about {clean_name} not available.',
        'symptoms': ['Symptoms not specified'],
        'cause': 'Cause not specified',
        'treatments': ['Treatment information not available']
    }

@app.route("/predict_disease", methods=["POST"])
def predict():
    global model, class_names
    
    if model is None:
        return jsonify({
            "status": "error", 
            "message": "Model not loaded. Please check server logs."
        }), 400
    
    try:
        # Check if image was provided
        if 'image' not in request.files:
            return jsonify({
                "status": "error", 
                "message": "No image file provided"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "status": "error", 
                "message": "No image selected"
            }), 400
        
        # Process image
        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_disease = class_names[predicted_idx]
        
        # Get disease information
        disease_info = get_disease_info(predicted_disease)
        
        # Prepare all predictions
        all_predictions = {}
        for i in range(len(class_names)):
            all_predictions[class_names[i]] = float(predictions[0][i])
        
        # Sort predictions by confidence
        sorted_predictions = dict(sorted(all_predictions.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return jsonify({
            "status": "success",
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "disease_info": disease_info,
            "all_predictions": sorted_predictions,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Prediction failed: {str(e)}"
        }), 500

@app.route("/reload_model", methods=["POST"])
def reload_model():
    """Reload model from Firebase - called after retraining"""
    try:
        logger.info("Reloading model...")
        success = load_model_from_firebase()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Model reloaded successfully",
                "classes": len(class_names),
                "version": model_version
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to reload model"
            }), 500
            
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route("/model_info", methods=["GET"])
def model_info():
    """Get current model information"""
    return jsonify({
        "model_loaded": model is not None,
        "num_classes": len(class_names) if class_names else 0,
        "classes": class_names,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "firebase_connected": db is not None,
        "timestamp": datetime.now().isoformat()
    })

# Fetch diseases list
@app.route("/diseases", methods=["GET"])
def get_diseases():
    try:
        diseases_ref = db.collection("rice_local_diseases")
        docs = diseases_ref.stream()
        
        diseases = []
        for doc in docs:
            disease = doc.to_dict()
            disease["id"] = doc.id
            diseases.append(disease)
        
        return jsonify({
            "status": "success",
            "count": len(diseases),
            "diseases": diseases
        })
    except Exception as e:
        logger.error(f"Error fetching diseases: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/", methods=["GET"])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Rice Disease Prediction API",
        "version": "1.0",
        "endpoints": {
            "/predict_disease": "POST - Upload image for disease prediction",
            "/model_info": "GET - Get model information",
            "/reload_model": "POST - Reload model from Firebase",
            "/health": "GET - Health check",
            "/diseases": "GET - Fetch all diseases from Firestore"
        },
        "model_loaded": model is not None,
        "classes": len(class_names) if class_names else 0
    })

# Initialize model on startup
def initialize_app():
    """Initialize the application and load model"""
    try:
        logger.info("Starting Rice Disease Prediction API...")
        logger.info("Loading model from Firebase...")
        model_loaded = load_model_from_firebase()
        
        if model_loaded:
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model loading failed, but server will continue...")
            
        return model_loaded
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        import traceback
        traceback.print_exc()
        return False

# Initialize model when the module is imported (for gunicorn)
initialize_app()

if __name__ == "__main__":
    # This only runs when called directly with python app.py (development)
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)



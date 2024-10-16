import os
import base64
import io
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
parent_directory = current_file.parent
sys.path.insert(0, str(parent_directory))

from models.face import FaceAnalysisPipeline, ImageProcessingError
import logging
from functools import wraps

version = "v1"
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FaceAnalysisPipeline
try:
    logger.info("Initializing FaceAnalysisPipeline")
    data_dir = Path(__file__).parent / "data"
    face_model_path = data_dir / "face-detection-retail-0004.xml"
    age_gender_model_path = data_dir / "age-gender-recognition-retail-0013.xml"
    emotion_model_path = data_dir / "emotions-recognition-retail-0003.xml"
    face_analyzer = FaceAnalysisPipeline(
        face_model=face_model_path,
        age_gender_model=age_gender_model_path,
        emotion_model=emotion_model_path
    )
    logger.info("FaceAnalysisPipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize FaceAnalysisPipeline: {str(e)}")
    raise

def check_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("X-RapidAPI-Proxy-Secret")
        if not auth_header:
            logger.warning(f"Authorization header missing. IP: {request.remote_addr}")
            return jsonify({"success": False, "error": "Authorization header is missing"}), 401

        # Uncomment the following lines when ready to implement full auth check
        token = os.getenv("X-RapidAPI-Proxy-Secret")
        if not auth_header == token:
            logger.warning(f"Invalid authorization header. IP: {request.remote_addr}")
            return jsonify({"success": False, "error": "Invalid authorization header"}), 401

        logger.info(f"Request authorized. IP: {request.remote_addr}")
        return f(*args, **kwargs)
    return decorated_function

def process_base64_image(base64_string):
    try:
        logger.info("Processing base64 image")
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))

        if img.mode != "RGB":
            img = img.convert("RGB")

        logger.info("Base64 image processed successfully")
        return img
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise ImageProcessingError(f"Error processing base64 image: {str(e)}")

def process_request(analysis_function):
    try:
        logger.info(f"Processing request for {analysis_function.__name__}")
        if request.is_json:
            logger.info("Handling JSON data")
            data = request.get_json()
            if not data or "image" not in data:
                logger.warning("No image data provided in JSON")
                return jsonify({"success": False, "error": "No image data provided in JSON"}), 400
            image = process_base64_image(data["image"])
        elif request.files:
            logger.info("Handling multipart/form-data")
            if "image" not in request.files:
                logger.warning("No image file provided in form-data")
                return jsonify({"success": False, "error": "No image file provided in form-data"}), 400
            file = request.files["image"]
            if file.filename == "":
                logger.warning("No selected image file")
                return jsonify({"success": False, "error": "No selected image file"}), 400
            if file:
                image = Image.open(file.stream)
                if image.mode != "RGB":
                    image = image.convert("RGB")
            else:
                logger.error("Failed to process image file")
                return jsonify({"success": False, "error": "Failed to process image file"}), 400
        else:
            logger.warning("Unsupported content type")
            return jsonify({"success": False, "error": "Unsupported content type"}), 415

        result = analysis_function(image)
        logger.info(f"Request processed successfully. Function: {analysis_function.__name__}")
        return jsonify({"success": True, "data": result}), 200
    except ImageProcessingError as e:
        logger.error(f"Image processing error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 422
    except Exception as e:
        logger.error(f"Unexpected error in {analysis_function.__name__}: {str(e)}")
        return jsonify({"success": False, "error": "An unexpected error occurred"}), 500

@app.route(f"/{version}/detect", methods=["POST"])
@check_auth
def detect_faces():
    logger.info("Received request for face detection")
    return process_request(face_analyzer.detect_faces)

@app.route(f"/{version}/estimate/age", methods=["POST"])
@check_auth
def estimate_age():
    logger.info("Received request for age estimation")
    return process_request(face_analyzer.estimate_age)

@app.route(f"/{version}/estimate/gender", methods=["POST"])
@check_auth
def estimate_gender():
    logger.info("Received request for gender estimation")
    return process_request(face_analyzer.estimate_gender)

@app.route(f"/{version}/recognize/emotion", methods=["POST"])
@check_auth
def recognize_emotion():
    logger.info("Received request for emotion recognition")
    return process_request(face_analyzer.recognize_emotion)

@app.route(f"/{version}/estimate/age-gender", methods=["POST"])
@check_auth
def estimate_age_gender():
    logger.info("Received request for age and gender estimation")
    return process_request(face_analyzer.estimate_age_gender)

@app.route(f"/{version}/analyze/full", methods=["POST"])
@check_auth
def analyze_full():
    logger.info("Received request for full analysis")
    return process_request(face_analyzer.analyze_full)

if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(debug=True)
import base64
import io
import numpy as np
from openvino.runtime import Core
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass

def load_image(base64_image):
    """
    Load an image from a base64 encoded string.

    Args:
        base64_image (str): Base64 encoded image string.

    Returns:
        PIL.Image: Loaded image.

    Raises:
        ImageProcessingError: If image loading fails.
    """
    try:
        logger.info("Attempting to load image from base64 string")
        image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
        logger.info(f"Image loaded successfully. Size: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image: {str(e)}")
        raise ImageProcessingError(f"Failed to load image: {str(e)}")

class FaceDetector:
    """Class for detecting faces in images using OpenVINO."""

    def __init__(self, model_path):
        """
        Initialize the FaceDetector with a given model.

        Args:
            model_path (str): Path to the face detection model.

        Raises:
            ImageProcessingError: If initialization fails.
        """
        try:
            logger.info(f"Initializing FaceDetector with model: {model_path}")
            self.ie = Core()
            self.model = self.ie.read_model(model=model_path)
            self.compiled_model = self.ie.compile_model(self.model, "CPU")
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            self.input_shape = self.input_layer.shape
            logger.info("FaceDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FaceDetector: {str(e)}")
            raise ImageProcessingError(f"Failed to initialize FaceDetector: {str(e)}")

    def detect(self, image):
        """
        Detect faces in the given image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            list: List of tuples containing bounding box coordinates (x1, y1, x2, y2).

        Raises:
            ImageProcessingError: If face detection fails.
        """
        try:
            logger.info("Starting face detection")
            image_width, image_height = image.size
            resized_image = np.array(image.resize(self.input_shape[2:]))
            input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
            results = self.compiled_model([input_image])[self.output_layer]

            faces = []
            for result in results[0][0]:
                if result[2] > 0.5:  # Confidence threshold
                    x1, y1 = int(result[3] * image_width), int(result[4] * image_height)
                    x2, y2 = int(result[5] * image_width), int(result[6] * image_height)
                    faces.append((x1, y1, x2, y2))
            logger.info(f"Face detection completed. Found {len(faces)} faces")
            return faces
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            raise ImageProcessingError(f"Face detection failed: {str(e)}")

class AgeGenderEstimator:
    """Class for estimating age and gender using OpenVINO."""

    def __init__(self, model_path):
        """
        Initialize the AgeGenderEstimator with a given model.

        Args:
            model_path (str): Path to the age-gender estimation model.

        Raises:
            ImageProcessingError: If initialization fails.
        """
        try:
            logger.info(f"Initializing AgeGenderEstimator with model: {model_path}")
            self.ie = Core()
            self.model = self.ie.read_model(model=model_path)
            self.compiled_model = self.ie.compile_model(self.model, "CPU")
            self.input_layer = self.compiled_model.input(0)
            self.output_layers = self.compiled_model.outputs
            self.input_shape = self.input_layer.shape
            logger.info("AgeGenderEstimator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AgeGenderEstimator: {str(e)}")
            raise ImageProcessingError(f"Failed to initialize AgeGenderEstimator: {str(e)}")

    def estimate(self, image):
        """
        Estimate age and gender from the given face image.

        Args:
            image (PIL.Image): Input face image.

        Returns:
            tuple: Estimated age (int) and gender (str).

        Raises:
            ImageProcessingError: If age and gender estimation fails.
        """
        try:
            logger.info("Starting age and gender estimation")
            resized_image = np.array(image.resize(self.input_shape[2:]))
            resized_image = resized_image[:, :, ::-1]  # RGB -> BGR
            input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
            results = self.compiled_model([input_image])
            gender = "female" if results[self.output_layers[0]][0][0][0] > 0.5 else "male"
            age = int(results[self.output_layers[1]][0][0][0][0] * 100)
            logger.info(f"Age and gender estimation completed. Age: {age}, Gender: {gender}")
            return age, gender
        except Exception as e:
            logger.error(f"Age and gender estimation failed: {str(e)}")
            raise ImageProcessingError(f"Age and gender estimation failed: {str(e)}")

class EmotionEstimator:
    """Class for estimating emotions using OpenVINO."""

    def __init__(self, model_path):
        """
        Initialize the EmotionEstimator with a given model.

        Args:
            model_path (str): Path to the emotion estimation model.

        Raises:
            ImageProcessingError: If initialization fails.
        """
        try:
            logger.info(f"Initializing EmotionEstimator with model: {model_path}")
            self.ie = Core()
            self.model = self.ie.read_model(model=model_path)
            self.compiled_model = self.ie.compile_model(self.model, "CPU")
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            self.input_shape = self.input_layer.shape
            self.emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
            logger.info("EmotionEstimator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EmotionEstimator: {str(e)}")
            raise ImageProcessingError(f"Failed to initialize EmotionEstimator: {str(e)}")

    def estimate(self, image):
        """
        Estimate emotion from the given face image.

        Args:
            image (PIL.Image): Input face image.

        Returns:
            str: Estimated emotion.

        Raises:
            ImageProcessingError: If emotion estimation fails.
        """
        try:
            logger.info("Starting emotion estimation")
            resized_image = np.array(image.resize(self.input_shape[2:]))
            input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
            results = self.compiled_model([input_image])[self.output_layer]
            emotion_index = np.argmax(results)
            emotion = self.emotions[emotion_index]
            logger.info(f"Emotion estimation completed. Detected emotion: {emotion}")
            return self.emotions[emotion_index]
        except Exception as e:
            logger.error(f"Emotion estimation failed: {str(e)}")
            raise ImageProcessingError(f"Emotion estimation failed: {str(e)}")

class FaceAnalysisPipeline:
    """Main class for running the complete face analysis pipeline."""

    def __init__(self, face_model, age_gender_model, emotion_model):
        """
        Initialize the FaceAnalysisPipeline with given models.

        Args:
            face_model (str): Path to the face detection model.
            age_gender_model (str): Path to the age-gender estimation model.
            emotion_model (str): Path to the emotion estimation model.

        Raises:
            ImageProcessingError: If initialization of any component fails.
        """
        try:
            logger.info("Initializing FaceAnalysisPipeline")
            self.face_detector = FaceDetector(face_model)
            self.age_gender_estimator = AgeGenderEstimator(age_gender_model)
            self.emotion_estimator = EmotionEstimator(emotion_model)
            logger.info("FaceAnalysisPipeline initialized successfully")
        except ImageProcessingError as e:
            logger.error(f"Failed to initialize FaceAnalysisPipeline: {str(e)}")
            raise

    def detect_faces(self, image):
        logger.info("Starting detect_faces method")
        faces = self.face_detector.detect(image)
        logger.info(f"Detected {len(faces)} faces")
        return [{"box": face} for face in faces]

    def estimate_age(self, image):
        logger.info("Starting estimate_age method")
        faces = self.face_detector.detect(image)
        results = []
        for i, face in enumerate(faces):
            face_image = image.crop(face)
            age, _ = self.age_gender_estimator.estimate(face_image)
            results.append({"box": face, "age": age})
            logger.info(f"Estimated age for face {i+1}: {age}")
        return results

    def estimate_gender(self, image):
        logger.info("Starting estimate_gender method")
        faces = self.face_detector.detect(image)
        results = []
        for i, face in enumerate(faces):
            face_image = image.crop(face)
            _, gender = self.age_gender_estimator.estimate(face_image)
            results.append({"box": face, "gender": gender})
            logger.info(f"Estimated gender for face {i+1}: {gender}")
        return results

    def recognize_emotion(self, image):
        logger.info("Starting recognize_emotion method")
        faces = self.face_detector.detect(image)
        results = []
        for i, face in enumerate(faces):
            face_image = image.crop(face)
            emotion = self.emotion_estimator.estimate(face_image)
            results.append({"box": face, "emotion": emotion})
            logger.info(f"Recognized emotion for face {i+1}: {emotion}")
        return results

    def estimate_age_gender(self, image):
        logger.info("Starting estimate_age_gender method")
        faces = self.face_detector.detect(image)
        results = []
        for i, face in enumerate(faces):
            face_image = image.crop(face)
            age, gender = self.age_gender_estimator.estimate(face_image)
            results.append({"box": face, "age": age, "gender": gender})
            logger.info(f"Estimated age and gender for face {i+1}: Age: {age}, Gender: {gender}")
        return results

    def analyze_full(self, image):
        logger.info("Starting analyze_full method")
        faces = self.face_detector.detect(image)
        results = []
        for i, face in enumerate(faces):
            face_image = image.crop(face)
            age, gender = self.age_gender_estimator.estimate(face_image)
            emotion = self.emotion_estimator.estimate(face_image)
            results.append({
                "box": face,
                "age": age,
                "gender": gender,
                "emotion": emotion
            })
            logger.info(f"Full analysis for face {i+1}: Age: {age}, Gender: {gender}, Emotion: {emotion}")
        return results
import os
import urllib.request
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Constants for face detector model URLs
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"

FACE_DETECTOR_DIR = "face_detector"
PROTOTXT_PATH = os.path.join(FACE_DETECTOR_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(FACE_DETECTOR_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

def download_face_detector():
    """Downloads the SSD Face Detector model files if they don't exist."""
    if not os.path.exists(FACE_DETECTOR_DIR):
        os.makedirs(FACE_DETECTOR_DIR)
        
    if not os.path.exists(PROTOTXT_PATH):
        print("[INFO] downloading deploy.prototxt...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
        
    if not os.path.exists(MODEL_PATH):
        print("[INFO] downloading res10_300x300_ssd_iter_140000_fp16.caffemodel...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def load_models():
    """Loads the Face Detector and Mask Detector models."""
    download_face_detector()
    print("[INFO] loading face detector model...")
    faceNet = cv2.dnn.readNet(PROTOTXT_PATH, MODEL_PATH)
    
    print("[INFO] loading face mask detector model...")
    # Catching potential error if the model hasn't been trained yet
    try:
        maskNet = load_model("mask_detector.h5")
        return faceNet, maskNet
    except Exception as e:
        print("[ERROR] Could not load mask_detector.h5. Was it trained?", e)
        return faceNet, None

def detect_and_predict_mask(frame, faceNet, maskNet):
    """
    Detects faces in the frame and predicts whether a mask is worn.
    """
    if maskNet is None:
        return [], []

    (h, w) = frame.shape[:2]
    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            # Ensure ROI is valid
            if endX > startX and endY > startY:
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = face / 255.0  # Scale the same way as training
                # Tensor Shape of face at this point: (224, 224, 3)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # Only make a predictions if at least one face was detected
    if len(faces) > 0:
        # Convert list of faces into a numpy array (Tensor Shape: [batch_size, 224, 224, 3])
        faces = np.array(faces, dtype="float32")
        # Predict uses the trained MobileNetV2 model
        # Tensor Shape of preds: [batch_size, 2] corresponding to [mask, without_mask]
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

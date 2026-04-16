import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from inference import load_models, detect_and_predict_mask

app = Flask(__name__)

# Global variables for models
faceNet = None
maskNet = None



@app.route('/')
def index():
    return render_template('index.html')

def init_models():
    global faceNet, maskNet
    faceNet, maskNet = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    if maskNet is None:
        return jsonify({"error": "Model not found."}), 500
        
    try:
        data = request.json
        image_data = data.get('image', '')
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        decoded_data = base64.b64decode(image_data)
        np_data = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
            
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        
        predictions = []
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            box_coords = [int(startX), int(startY), int(endX), int(endY)] 
            
            (mask, withoutMask) = pred
            if mask > withoutMask:
                label = "Mask"
                prob = float(mask)
            else:
                label = "No Mask"
                prob = float(withoutMask)
                
            predictions.append({
                "box": box_coords,
                "label": label,
                "prob": prob
            })
            
        return jsonify({"predictions": predictions})
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize models before the first request
    init_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

from flask import Flask, render_template, Response, jsonify
import cv2
import time
from inference import load_models, detect_and_predict_mask

app = Flask(__name__)

# Global variables for models
faceNet = None
maskNet = None

# Global counters (updated each frame)
current_counts = {
    "mask": 0,
    "no_mask": 0,
    "total": 0
}

def init_models():
    global faceNet, maskNet
    faceNet, maskNet = load_models()

def generate_frames():
    global current_counts

    # Attempt to open the webcam
    vs = cv2.VideoCapture(0)
    time.sleep(2.0) # give camera warmup time

    while True:
        success, frame = vs.read()
        if not success:
            break
            
        # Optional: resize the frame to have a maximum width of 400 pixels to speed up processing
        # import imutils; frame = imutils.resize(frame, width=400)

        # Detect faces and predict mask
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # Per-frame counters
        mask_count = 0
        no_mask_count = 0

        # Loop over the detected face locations and their corresponding predictions
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            
            # Since alphabetical sorting places "with_mask" at index 0 and "without_mask" at index 1
            # We assume label 0 is mask, label 1 is without_mask
            (mask, withoutMask) = pred

            # Determine the class label and color we'll use to draw the bounding box and text
            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0) # Green for Mask
                mask_count += 1
            else:
                label = "No Mask"
                color = (0, 0, 255) # Red for No Mask
                no_mask_count += 1
                
            # Note: We simplified to 2 classes. So Yellow (incorrect) is omitted here.

            # Include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        total_count = mask_count + no_mask_count

        # Update global counts for the /counts API endpoint
        current_counts = {
            "mask": mask_count,
            "no_mask": no_mask_count,
            "total": total_count
        }

        # Draw count overlay on the top-left corner of the frame
        overlay_y = 30
        cv2.putText(frame, f"Total People: {total_count}", (10, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"With Mask: {mask_count}", (10, overlay_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Without Mask: {no_mask_count}", (10, overlay_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the output frame in the byte format for HTTP multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    vs.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    if maskNet is None:
        return "Model not found. Please run train_model.py first."
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def counts():
    """Returns the current frame's mask/no-mask/total counts as JSON."""
    return jsonify(current_counts)

if __name__ == '__main__':
    # Initialize models before the first request
    init_models()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

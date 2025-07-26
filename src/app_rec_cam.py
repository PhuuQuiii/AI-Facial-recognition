from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.disable_eager_execution() # Important to keep for your existing Facenet model

import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face # This is MTCNN
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import base64

# --- Global Variables & Model Initialization ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Configuration (Adjust paths as needed) ---
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160

# Ensure these paths are correct relative to where you run the Flask app
CLASSIFIER_PATH = r'D:\E\DoANChuyenNganh\Facial_recognition\Models\facemodel.pkl'
FACENET_MODEL_PATH = r'D:\E\DoANChuyenNganh\Facial_recognition\Models\20180402-114759.pb'
# ### DEBUG ###: Verify this directory and its contents
MTCNN_MODEL_DIR = r'D:\E\DoANChuyenNganh\Facial_recognition\src\align'
print(f"### DEBUG ###: Checking MTCNN_MODEL_DIR: {MTCNN_MODEL_DIR}")
if not os.path.isdir(MTCNN_MODEL_DIR):
    print(f"### DEBUG ###: ERROR - MTCNN_MODEL_DIR does not exist or is not a directory!")
else:
    print(f"### DEBUG ###: Contents of MTCNN_MODEL_DIR: {os.listdir(MTCNN_MODEL_DIR)}")


# --- Load Models ---
print("Loading Facenet model...")
tf_graph = tf.Graph()
with tf_graph.as_default():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    tf_sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with tf_sess.as_default():
        facenet.load_model(FACENET_MODEL_PATH)
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        print("Facenet model loaded.")

        print("Loading MTCNN model...")
        try:
            pnet, rnet, onet = align.detect_face.create_mtcnn(tf_sess, MTCNN_MODEL_DIR)
            print("MTCNN model loaded successfully.")
        except Exception as e_mtcnn_load:
            print(f"### DEBUG ###: CRITICAL ERROR loading MTCNN models: {e_mtcnn_load}")
            pnet, rnet, onet = None, None, None # Ensure they are None if loading failed

print(f"Loading classifier model from {CLASSIFIER_PATH}...")
if not os.path.exists(CLASSIFIER_PATH):
    print(f"Error: Classifier model file not found at {CLASSIFIER_PATH}")
    sys.exit(1) # Or handle more gracefully if this is a non-critical part for pure detection
with open(CLASSIFIER_PATH, 'rb') as infile:
    model, class_names = pickle.load(infile)
print("Classifier model loaded.")
print(f"Class names: {class_names}")

# Biến toàn cục lưu MSSV nhận diện gần nhất
last_mssv = None

# --- Flask API Endpoint ---
@app.route('/recognize', methods=['POST'])
def recognize_face():
    global last_mssv
    start_time = datetime.datetime.now() # For overall processing time measurement

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    date_param = request.form.get('date', str(datetime.date.today()))
    class_id_param = request.form.get('classId', 'default_class')
    print(f"### DEBUG ###: Received request for date: {date_param}, classId: {class_id_param}")

    try:
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("### DEBUG ###: cv2.imdecode returned None. Image could not be decoded.")
            return jsonify({"error": "Could not decode image"}), 400
        
        print(f"### DEBUG ###: Image decoded successfully. Frame shape: {frame.shape}, dtype: {frame.dtype}")

        # Check if MTCNN models loaded properly before trying to use them
        if pnet is None or rnet is None or onet is None:
            print("### DEBUG ###: MTCNN models are not loaded. Cannot perform face detection.")
            return jsonify({"error": "MTCNN models not loaded on server", "details": "Face detection cannot be performed."}), 500


        # --- Face Detection ---
        # MTCNN expects BGR image, which frame is.
        # It's good practice to convert to RGB if your `align.detect_face` expects it,
        # but typically David Sandberg's implementation handles BGR from OpenCV.
        # If issues persist, try: frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # and pass frame_rgb to detect_face.
        print(f"### DEBUG ###: Calling align.detect_face.detect_face with MINSIZE={MINSIZE}, THRESHOLD={THRESHOLD}, FACTOR={FACTOR}")
        bounding_boxes, points = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        
        if bounding_boxes is None:
            print("### DEBUG ###: align.detect_face.detect_face returned None for bounding_boxes.")
            faces_found = 0
        else:
            faces_found = bounding_boxes.shape[0]
            print(f"### DEBUG ###: Raw bounding_boxes from MTCNN (shape: {bounding_boxes.shape}):\n{bounding_boxes}")
            if points is not None:
                 print(f"### DEBUG ###: Raw points (landmarks) from MTCNN (shape: {points.shape}):\n{points}")
            else:
                 print("### DEBUG ###: MTCNN did not return points (landmarks).")


        print("### DEBUG ###: Số khuôn mặt phát hiện bởi MTCNN:", faces_found)

        results = []
        recognition_success = False # Overall success if at least one face is confidently recognized
        mssv_to_send = None

        if faces_found > 0:
            det = bounding_boxes[:, 0:4] # x1, y1, x2, y2
            confidences = bounding_boxes[:, 4] # Confidence score for each bounding box

            bb = np.zeros((faces_found, 4), dtype=np.int32)

            for i in range(faces_found):
                print(f"### DEBUG ###: Processing face {i+1}/{faces_found}")
                print(f"### DEBUG ###:   MTCNN raw det[{i}]: {det[i]}")
                print(f"### DEBUG ###:   MTCNN confidence[{i}]: {confidences[i]}")

                # Ensure coordinates are within image bounds and valid
                bb[i][0] = np.maximum(int(det[i][0]), 0)
                bb[i][1] = np.maximum(int(det[i][1]), 0)
                bb[i][2] = np.minimum(int(det[i][2]), frame.shape[1])
                bb[i][3] = np.minimum(int(det[i][3]), frame.shape[0])
                print(f"### DEBUG ###:   Calculated bb[{i}] (x1,y1,x2,y2): {bb[i]}")

                # Check for valid bounding box dimensions
                if bb[i][2] <= bb[i][0] or bb[i][3] <= bb[i][1]:
                    print(f"### DEBUG ###:   Skipping face {i} due to invalid bbox dimensions: w={bb[i][2]-bb[i][0]}, h={bb[i][3]-bb[i][1]}")
                    results.append({
                        "MSSV": "ErrorInvalidBBox",
                        "error": "Invalid bounding box dimensions from MTCNN.",
                        "bbox_raw_mtcnn": [float(d) for d in det[i]],
                        "bbox_calculated": [int(b) for b in bb[i]]
                    })
                    continue
                
                try:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    if cropped.size == 0:
                        print(f"### DEBUG ###:   Warning: Cropped area for face {i} is empty. BBox: {bb[i]}. Skipping.")
                        results.append({
                            "MSSV": "ErrorEmptyCrop",
                            "error": "Cropped face region is empty.",
                            "bbox": [int(b) for b in bb[i]]
                        })
                        continue
                    print(f"### DEBUG ###:   Cropped face {i} shape: {cropped.shape}")

                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = tf_sess.run(embeddings, feed_dict=feed_dict)

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    
                    if not class_names: # Should not happen if classifier loaded
                        print("### DEBUG ###: class_names is empty!")
                        best_name = "ErrorNoClassNames"
                    elif best_class_indices[0] >= len(class_names):
                        print(f"### DEBUG ###: best_class_indices[0] ({best_class_indices[0]}) is out of bounds for class_names (len: {len(class_names)})")
                        best_name = "ErrorIndexOutOfBounds"
                    else:
                        best_name = class_names[best_class_indices[0]]


                    print(f"### DEBUG ###:   Face {i} - Detected: {best_name}, Probability: {best_class_probabilities[0]:.4f}")

                    current_recognition = {
                        "MSSV": best_name,
                        "probability": float(best_class_probabilities[0]),
                        "bbox": [int(b) for b in bb[i]] # Bounding box [x1, y1, x2, y2]
                    }

                    if best_class_probabilities[0] > 0.70: # Confidence threshold for "known"
                        recognition_success = True # Mark overall success
                        if mssv_to_send is None:
                            mssv_to_send = best_name
                        # results.append(current_recognition) # Appended below
                    else:
                        current_recognition["MSSV"] = "Unknown" # If below threshold, classify as Unknown
                        current_recognition["original_prediction"] = best_name # Keep what it thought it was
                    
                    results.append(current_recognition)

                except Exception as e_face:
                    print(f"### DEBUG ###:   Error processing face {i}: {e_face}")
                    results.append({
                        "MSSV": "ErrorInFaceProcessingLoop",
                        "error": str(e_face),
                        "bbox": [int(b) for b in bb[i]] if 'bb' in locals() and i < len(bb) else "N/A"
                    })
        
        end_time = datetime.datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000

        response_payload = {
            "success": recognition_success,
            "message": "",
            "date": date_param,
            "classId": class_id_param,
            "recognitions": results,
            "faces_detected_count": faces_found,
            "processing_time_ms": round(processing_time_ms, 2),
            "MSSV": mssv_to_send,
        }

        if faces_found == 0:
            response_payload["message"] = "No faces detected in the image."
        elif not results and faces_found > 0: # Faces detected, but all resulted in errors or were skipped
             response_payload["message"] = "Faces detected, but could not recognize or process any."
        elif not recognition_success and faces_found > 0: # Faces processed, but none met high confidence
            response_payload["message"] = "Recognition process completed, but no one identified with high confidence."
        elif recognition_success:
            response_payload["message"] = "Recognition process completed."
        
        # Lưu MSSV nhận diện gần nhất (nếu có)
        if mssv_to_send:
            last_mssv = mssv_to_send

        return jsonify(response_payload), 200

    except Exception as e:
        print(f"### DEBUG ###: Critical Error in /recognize endpoint: {e}")
        import traceback
        traceback.print_exc() # Print full traceback to Flask console
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

@app.route('/get_last_mssv', methods=['GET'])
def get_last_mssv():
    global last_mssv
    if last_mssv:
        return jsonify({"MSSV": last_mssv, "success": True})
    else:
        return jsonify({"MSSV": None, "success": False, "message": "No MSSV recognized yet."})


if __name__ == '__main__':
    print("### DEBUG ###: Starting Flask app for recognition service...")
    # For production, consider using Gunicorn or another WSGI server.
    # threaded=False is often recommended for TensorFlow in Flask's dev server.
    with tf_graph.as_default():
        with tf_sess.as_default():
            if pnet and rnet and onet and images_placeholder is not None: # Check if models loaded
                print("### DEBUG ###: Performing TensorFlow warm-up...")
                try:
                    # Create a dummy image tensor of the expected input size
                    dummy_image_np = np.zeros((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3), dtype=np.float32)
                    dummy_image_np = facenet.prewhiten(dummy_image_np)
                    dummy_image_reshaped = dummy_image_np.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                    # Warm-up Facenet
                    _ = tf_sess.run(embeddings, feed_dict={
                        images_placeholder: dummy_image_reshaped,
                        phase_train_placeholder: False
                    })

                    # Warm-up MTCNN (requires a frame-like input)
                    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Example frame size
                    _ = align.detect_face.detect_face(dummy_frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                    print("### DEBUG ###: TensorFlow warm-up completed.")
                except Exception as e_warmup:
                    print(f"### DEBUG ###: Error during TensorFlow warm-up: {e_warmup}")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=False)
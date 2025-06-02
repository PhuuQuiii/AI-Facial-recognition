# -*- coding: utf-8 -*-

from __future__ import print_function, division

import cv2
import torch
from torchvision import transforms
import time
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
import os

# --- Phần import model CDCNpp và Conv2d_cd ---
try:
    from models.CDCNs import CDCNpp, Conv2d_cd
    print("INFO: Đã import CDCNpp và Conv2d_cd từ models.CDCNs")
except ImportError:
    print("LỖI: Không thể import CDCNpp từ models.CDCNs. Sử dụng định nghĩa DUMMY.")
    if 'CDCNpp' not in globals():
        class CDCNpp(torch.nn.Module):
            def __init__(self, basic_conv=None, theta=0.7):
                super(CDCNpp, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)
                print("CẢNH BÁO: Sử dụng định nghĩa CDCNpp DUMMY. Model thực tế không được tải.")
            def forward(self, x):
                if x.shape[1] == 1:
                    temp_conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1).to(x.device)
                    map_x = temp_conv(x)
                else:
                    map_x = self.conv1(x)
                return map_x, None, None, None, None, None
    if 'Conv2d_cd' not in globals():
         class Conv2d_cd(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, theta=0.0):
                super(Conv2d_cd, self).__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            def forward(self, x):
                return self.conv(x)
# --- Kết thúc phần import model ---

app = Flask(__name__)

MODEL_PATH = "CDCNpp_BinaryMask_P1_07/CDCNpp_BinaryMask_P1_07_30.pkl"
TRAINING_THETA_VALUE = 0.7
DECISION_THRESHOLD = 0.9

actual_model = None
device = None
face_cascade = None

def load_global_model_and_detector():
    global actual_model, device, face_cascade
    print("INFO: Đang tải model và face detector...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"INFO: Sử dụng device: {device}")

    try:
        if 'models.CDCNs' in str(CDCNpp):
             actual_model = CDCNpp(basic_conv=Conv2d_cd, theta=TRAINING_THETA_VALUE).to(device)
        else:
            actual_model = CDCNpp().to(device)

        if os.path.exists(MODEL_PATH):
            try:
                if 'models.CDCNs' in str(CDCNpp) or (hasattr(actual_model, 'conv1') and actual_model.conv1.in_channels == 3):
                    state_dict = torch.load(MODEL_PATH, map_location=device)
                    actual_model.load_state_dict(state_dict)
                    print(f"INFO: Model weights loaded successfully from {MODEL_PATH}")
                else:
                    print("INFO: Model là DUMMY, bỏ qua việc tải state_dict.")
            except Exception as e:
                 print(f"LỖI khi tải model state_dict: {e}. Model có thể không hoạt động đúng.")
                 if not ('models.CDCNs' in str(CDCNpp)):
                      print("INFO: Tiếp tục với DUMMY model mà không có trọng số đã huấn luyện.")
                 else:
                      actual_model = None
        else:
            print(f"LỖI: Không tìm thấy file model tại: {MODEL_PATH}")
            if not ('models.CDCNs' in str(CDCNpp)):
                 print("INFO: Tiếp tục với DUMMY model.")
            else:
                 actual_model = None
        
        if actual_model:
            actual_model.eval()
    except Exception as e:
        print(f"LỖI nghiêm trọng khi khởi tạo CDCNpp: {e}")
        actual_model = None

    if actual_model is None and 'models.CDCNs' in str(CDCNpp):
        print("LỖI: Không thể tải model CDCNpp thực tế.")
    elif actual_model is None:
        print("LỖI: Không thể khởi tạo cả model DUMMY.")

    try:
        face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(face_cascade_path):
            print(f"LỖI: Không tìm thấy file Haar Cascade tại: {face_cascade_path}")
            face_cascade = None
        else:
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if face_cascade.empty():
                print(f"LỖI: Không thể tải file cascade từ: {face_cascade_path}")
                face_cascade = None
            else:
                print("INFO: Face detector (Haar Cascade) loaded successfully.")
    except Exception as e:
        print(f"LỖI: Không thể tải bộ phát hiện khuôn mặt Haar Cascade: {e}")
        face_cascade = None

# Hàm tiền xử lý TOÀN BỘ ảnh cho model
def preprocess_full_image_for_model(full_image_np, device_to_use):
    # Input `full_image_np` là TOÀN BỘ ảnh OpenCV (BGR).
    # print(f"DEBUG: Preprocessing full image with shape: {full_image_np.shape}")
    image_rgb = cv2.cvtColor(full_image_np, cv2.COLOR_BGR2RGB)
    transform_ops = [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), # Resize TOÀN BỘ ảnh về kích thước đầu vào của model
        transforms.ToTensor(),
        # Cân nhắc thêm Normalize nếu model của bạn được huấn luyện với nó
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_ops)
    return transform(image_rgb).unsqueeze(0).to(device_to_use)

# Hàm thực hiện dự đoán
def get_prediction(image_np): # image_np là ảnh full frame dạng numpy array (BGR)
    global actual_model, device, face_cascade, DECISION_THRESHOLD

    if actual_model is None:
        return {"error": "Model không được tải. Vui lòng kiểm tra console của server."}, 500
    # Face cascade có thể None, nhưng chúng ta vẫn xử lý ảnh, chỉ không vẽ box
    # if face_cascade is None:
    #     return {"error": "Face detector không được tải. Vui lòng kiểm tra console của server."}, 500

    start_time = time.time()

    # Phát hiện khuôn mặt bằng Haar Cascade (chỉ để hiển thị, không ảnh hưởng input model)
    num_haar_faces = 0
    main_face_display_coord = None
    if face_cascade: # Chỉ thực hiện nếu face_cascade được tải thành công
        gray_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces_detected_by_haar = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        num_haar_faces = len(faces_detected_by_haar)

        if num_haar_faces > 0:
            # Sắp xếp để lấy khuôn mặt lớn nhất cho việc hiển thị bounding box (nếu có nhiều)
            sorted_faces = sorted(faces_detected_by_haar, key=lambda f: f[2]*f[3], reverse=True)
            x_disp, y_disp, w_disp, h_disp = sorted_faces[0]
            main_face_display_coord = {"x": int(x_disp), "y": int(y_disp), "w": int(w_disp), "h": int(h_disp)}
    else:
        print("CẢNH BÁO: Face detector (Haar Cascade) không khả dụng. Sẽ không vẽ bounding box.")


    # Tiền xử lý và dự đoán TRÊN TOÀN BỘ ẢNH
    try:
        input_tensor = preprocess_full_image_for_model(image_np, device)
    except Exception as e:
        print(f"Lỗi trong quá trình tiền xử lý toàn bộ ảnh: {e}")
        processing_time = time.time() - start_time
        return {"error": f"Lỗi tiền xử lý ảnh: {str(e)}", "processing_time": f"{processing_time:.3f}s"}, 500

    label = "N/A"
    pred_score = 0.0
    pred_score_unclipped = 0.0

    with torch.no_grad():
        try:
            output_from_model = actual_model(input_tensor)
            
            if isinstance(output_from_model, (tuple, list)):
                if not output_from_model or output_from_model[0] is None:
                    print(f"Lỗi: Model trả về output không hợp lệ hoặc map_x là None. Output: {output_from_model}")
                    processing_time = time.time() - start_time
                    return {"error": "Model trả về output không hợp lệ.", "processing_time": f"{processing_time:.3f}s", "num_faces": num_haar_faces, "face_coords": main_face_display_coord}, 500
                map_x = output_from_model[0]
            else:
                map_x = output_from_model

            if not isinstance(map_x, torch.Tensor):
                 print(f"Lỗi: map_x không phải là Tensor. Type: {type(map_x)}")
                 processing_time = time.time() - start_time
                 return {"error": "map_x không phải là Tensor.", "processing_time": f"{processing_time:.3f}s", "num_faces": num_haar_faces, "face_coords": main_face_display_coord}, 500
            if map_x.numel() == 0:
                print("Lỗi: map_x rỗng (không có phần tử).")
                processing_time = time.time() - start_time
                return {"error": "map_x rỗng.", "processing_time": f"{processing_time:.3f}s", "num_faces": num_haar_faces, "face_coords": main_face_display_coord}, 500

            pred_score_unclipped = torch.mean(map_x).item()
            # print(f"DEBUG: Raw pred_score_unclipped: {pred_score_unclipped}") # Để debug
            
            pred_score = max(0.0, min(1.0, pred_score_unclipped))
            
            label = "Thật (Real)" if pred_score > DECISION_THRESHOLD else "Giả (Fake)"

        except Exception as e:
            print(f"Lỗi khi thực thi model hoặc xử lý output: {e}")
            import traceback
            traceback.print_exc()
            processing_time = time.time() - start_time
            return {"error": f"Lỗi model execution: {str(e)}", "processing_time": f"{processing_time:.3f}s", "num_faces": num_haar_faces, "face_coords": main_face_display_coord}, 500

    processing_time = time.time() - start_time
    
    return {
        "label": label,
        "score": f"{pred_score:.4f}",
        "raw_score": f"{pred_score_unclipped:.4f}", # Trả về raw score để debug
        "processing_time": f"{processing_time:.3f}s",
        "num_faces": num_haar_faces, # Số khuôn mặt Haar phát hiện
        "face_coords": main_face_display_coord # Tọa độ khuôn mặt lớn nhất (nếu có) để vẽ
    }, 200


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_data' not in request.form and 'file' not in request.files:
        return jsonify({"error": "Không có dữ liệu ảnh"}), 400

    try:
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if image_np is None:
                 return jsonify({"error": "Không thể decode file ảnh tải lên."}), 400
            # print("INFO: Đã nhận ảnh từ file tải lên.")
        elif 'image_data' in request.form:
            image_data_base64 = request.form['image_data'].split(',')[1]
            image_bytes = base64.b64decode(image_data_base64)
            npimg = np.frombuffer(image_bytes, np.uint8)
            image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if image_np is None:
                 return jsonify({"error": "Không thể decode ảnh base64 từ webcam."}), 400
            # print("INFO: Đã nhận ảnh từ webcam.")
        else:
            return jsonify({"error": "Dữ liệu không hợp lệ"}), 400

        result, status_code = get_prediction(image_np)
        return jsonify(result), status_code
    except Exception as e:
        print(f"Lỗi không xác định trong route /predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Lỗi server: {str(e)}"}), 500

if __name__ == '__main__':
    load_global_model_and_detector()
    if actual_model is None: # Chỉ cần kiểm tra model chính
        print("CẢNH BÁO QUAN TRỌNG: Model chính (actual_model) không tải được. Ứng dụng sẽ không thể nhận diện Thật/Giả.")
    if face_cascade is None:
        print("CẢNH BÁO: Face detector (Haar Cascade) không tải được. Sẽ không thể vẽ ô vuông quanh khuôn mặt.")
    
    app.run(host='0.0.0.0', port=3000, debug=True) # Giữ port 3000 như bạn đã sửax
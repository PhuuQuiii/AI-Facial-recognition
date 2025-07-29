from flask import Flask, render_template, request, session, jsonify, redirect, url_for
import numpy as np
import mysql.connector
import align.detect_face
import facenet
import tensorflow as tf
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv() 
import os
from datetime import datetime, timedelta, time
import base64
import subprocess
import cv2
import pickle
from imutils.video import VideoStream
import imutils
import socket
import requests  # Đã có
from socketio import Client  # Thêm dòng này


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

ALLOWED_IP = '192.168.1.151'  # IP WiFi được phép truy cập

print("Chỉ cho phép truy cập từ IP:", ALLOWED_IP)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') # Thay đổi secret key cho bảo mật
CORS(app)  # Để cho phép tất cả các nguồn kết nối
CORS(app, resources={r"/*": {"origins": "*"}})  # Cho phép tất cả nguồn gốc

socketio = SocketIO(app, cors_allowed_origins="*")

# Các biến toàn cục cần thiết
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = '../Models/facemodel.pkl'
FACENET_MODEL_PATH = '../Models/20180402-114759.pb'

# Khởi tạo TensorFlow và model FaceNet
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
facenet.load_model('../Models/20180402-114759.pb')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/student')
def student():
    return render_template('student.html')

@app.route('/login_student', methods=['GET', 'POST'])
def login_student():
    if request.method == 'POST':
        data = request.get_json()
        student_id = data['studentId']
        password = data['password']

        print(student_id, password)

        # Kết nối đến cơ sở dữ liệu
        connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
        cursor = connection.cursor()

        # Truy vấn thông tin sinh viên
        cursor.execute("SELECT * FROM Student WHERE MSSV = %s", (student_id,))
        student = cursor.fetchone()

        if student:
            # Kiểm tra mật khẩu
            if student[2] == password:  
                session['student_id'] = student[0]  
                session['student_name'] = student[1] 

                return jsonify({"success": True, "name": student[1]})

        return jsonify({"success": False, "message": "Mã số sinh viên hoặc mật khẩu không đúng."})

    return render_template('login_student.html')
# Route cho trang dashboard sinh viên
@app.route('/student_dashboard') 
def student_dashboard():
    return render_template('student_dashboard.html')

@app.route('/login_teacher', methods=['GET', 'POST'])
def login_teacher():
    if request.method == 'POST':
        data = request.get_json()
        teacher_id = data['teacherId']
        password = data['password']

        # Kết nối đến cơ sở dữ liệu
        connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
        cursor = connection.cursor()

        # Truy vấn thông tin giáo viên
        cursor.execute("SELECT * FROM Teacher WHERE MSGV = %s", (teacher_id,))
        teacher = cursor.fetchone()

        if teacher:
            # Kiểm tra mật khẩu
            if teacher[2] == password:  # So sánh với trường password
                session['teacher_id'] = teacher[0]  # MSGV
                session['teacher_name'] = teacher[1]  # Tên giáo viên

                # Lấy thông tin lớp mà giáo viên phụ trách
                cursor.execute("SELECT id_class, name FROM Class WHERE teacher_id = %s", (teacher_id,))
                class_info = cursor.fetchone()

                if class_info:
                    session['class'] = class_info[0]  # Mã lớp
                    session['class_name'] = class_info[1]  # Tên lớp
                    return jsonify({"success": True, "name": teacher[1], "class": class_info[1]})  # Trả về tên lớp

                return jsonify({"success": True, "name": teacher[1], "class": None})

        return jsonify({"success": False, "message": "Mã số giáo viên hoặc mật khẩu không đúng."})

    return render_template('login_teacher.html')



# Route cho trang dashboard giáo viên
@app.route('/teacher_dashboard')
def teacher_dashboard():
    return render_template('teacher_dashboard.html')

# Route logout trang dashboard giáo viên
@app.route('/teacher_dashboard/logout')
def logout():
    session.clear()
    return redirect(url_for('login_teacher'))

@socketio.on('connect')
def handle_connect():
    print("Client đã kết nối")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client đã ngắt kết nối")

@socketio.on('response')
def handle_response(data):

    if 'MSSV' in data:
        emit('response', { "message": f"MSSV: {data['MSSV']} thuộc lớp {data['classId']} đã điểm danh vào ngày {data['date']}", "MSSV": data['MSSV'], "classId": data['classId'], "date": data['date'] }, broadcast=True)    
    

def get_class_id_by_student_id(student_id): # lấy class_id từ cơ sở dữ liệu
    connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    cursor = connection.cursor()
    cursor.execute("""
        SELECT sc.class_id 
        FROM Student_Class sc 
        WHERE sc.student_id = %s
    """, (student_id,))
    class_id = cursor.fetchone()
    cursor.close()
    connection.close()
    
    if class_id:
        return class_id[0]  # Trả về class_id
    return None  # Nếu không tìm thấy

@app.route('/get_classes')
def get_classes():
    teacher_id = request.args.get('teacher_id')
    
    try:
        connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    cursor = connection.cursor()
    cursor.execute("SELECT id_class, name, room FROM Class WHERE teacher_id = %s", (teacher_id,))
    classes = cursor.fetchall()
    print("Classes fetched:", classes)
    return jsonify({"classes": [{"id_class": cls[0], "name": cls[1], "room": cls[2]} for cls in classes]})

@app.route('/get_weeks')
def get_weeks():
    class_id = request.args.get('class_id')
    connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    cursor = connection.cursor()
    cursor.execute("""
        SELECT DISTINCT date 
        FROM Attendance 
        WHERE class_id = %s 
        ORDER BY date
    """, (class_id,))
    result = cursor.fetchall()
    connection.close()

    if not result:
        return jsonify({"success": True, "weeks": []})  # Không có ngày học

    # Danh sách ngày
    dates = [row[0] for row in result]
    weeks = []
    delta = 0
    # Tính toán các tuần dựa trên ngày học
    for date in dates:
        week_label = f"Tuần {delta + 1}"
        weeks.append({"label": week_label, "value": str(date)})
        delta += 1

    return jsonify({"success": True, "weeks": weeks})

@app.route('/get_students')
def get_students():
    class_id = request.args.get('class_id')
    date = request.args.get('week')
    print(class_id, date)
    connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    cursor = connection.cursor()
    cursor.execute("""
        SELECT s.MSSV, s.name, a.date, a.status, a.class_id
        FROM student s
        JOIN attendance a ON s.MSSV = a.student_id
        WHERE a.class_id = %s  AND a.date = %s
    """, (class_id, date))
    students = cursor.fetchall()
    return jsonify({"students": [{"MSSV": student[0], "name": student[1], "date": student[2], "status": student[3], "class_id": student[4]} for student in students]})

# def get_student_info(student_id):
#     connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
#     cursor = connection.cursor()
#     cursor.execute("SELECT MSSV, name FROM Student WHERE MSSV = %s", (student_id,))
#     student = cursor.fetchone()
#     if student:
#         return {
#             "MSSV": student[0],
#             "name": student[1],
#         }
#     return {"error": "Student not found"}

@app.route('/update_attendance', methods=['POST'])
def update_attendance():
    data = request.get_json()  # Lấy dữ liệu JSON từ request
    mssv = data.get('MSSV')  # Lấy giá trị MSSV
    class_id = data.get('class_id')  # Lấy class_id
    date = data.get('date')
    status = data.get('status')  # Lấy status

    # Kiểm tra nếu thiếu dữ liệu
    if not all([mssv, class_id, date, status]):
        return jsonify({"success": False, "message": "Thiếu dữ liệu bắt buộc."}), 400

    # Kết nối cơ sở dữ liệu
    connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
    cursor = connection.cursor()

    # Lấy startTime từ Attendance (hoặc Class nếu cần)
    cursor.execute("""
        SELECT startTime FROM Attendance
        WHERE student_id = %s AND class_id = %s AND date = %s
    """, (mssv, class_id, date))
    result = cursor.fetchone()
    if not result:
        cursor.close()
        connection.close()
        return jsonify({"success": False, "message": "Không tìm thấy thông tin buổi học."}), 404

    start_time = result[0]  # kiểu datetime.time hoặc timedelta
    now = datetime.now().time()
    # Chuyển đổi start_time sang datetime nếu cần
    if isinstance(start_time, timedelta):
        start_time = (datetime.min + start_time).time()

    # Tính chênh lệch phút
    start_dt = datetime.combine(datetime.today(), start_time)
    now_dt = datetime.combine(datetime.today(), now)
    diff_minutes = (now_dt - start_dt).total_seconds() / 60

    if diff_minutes > 30:
        cursor.close()
        connection.close()
        return jsonify({"success": False, "message": "Đã quá 30 phút đầu giờ bạn đã vắng =))"}), 403

    # Nếu trong 30 phút đầu, cho phép điểm danh
    cursor.execute("""
        UPDATE Attendance 
        SET status = %s 
        WHERE student_id = %s AND class_id = %s AND date = %s
    """, (status, mssv, class_id,  date))
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({"success": True, "message": status})


#Phần xư lý ip wifi học viện

@app.before_request
def before_request_func():
    server_ip = get_local_ip()
    print(f"ALLOWED_IP: {ALLOWED_IP} | SERVER_IP: {server_ip}")
    if server_ip != ALLOWED_IP:
        return "Truy cập bị từ chối: chỉ cho phép từ WiFi Học Viện", 403


@app.route('/save_images', methods=['POST'])
def save_images():
    data = request.get_json()
    student_id = data['studentId']
    images = data['images']

    # Đường dẫn đến thư mục raw trong thư mục FaceData
    directory = os.path.join(os.getcwd(), "Dataset", "FaceData", "raw", student_id)
    os.makedirs(directory, exist_ok=True)
  

    # Lưu từng ảnh vào thư mục
    for i, image in enumerate(images):
        # Xử lý dữ liệu base64
        image_data = image.split(",")[1]
        try:
            with open(f"{directory}/{student_id}_{i + 1}.jpg", "wb") as fh:
                fh.write(base64.b64decode(image_data))
            print(f"Saved image: {student_id}_{i + 1}.jpg")
        except Exception as e:
            print(f"Error saving image {i + 1}: {e}")

    return jsonify({"success": True})

@app.route('/preprocess_data', methods=['POST'])
def preprocess_data():
    try:
        # Chạy lệnh tiền xử lý
        subprocess.run(['python', 'align_dataset_mtcnn.py', '../Dataset/FaceData/raw', '../Dataset/FaceData/processed', '--image_size', '160', '--margin', '32', '--random_order', '--gpu_memory_fraction', '0.25'], check=True)
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # Chạy lệnh huấn luyện
        subprocess.run(['python', 'classifier.py', 'TRAIN', '../Dataset/FaceData/processed', '../Models/20180402-114759.pb', '../Models/facemodel.pkl', '--batch_size', '1000'], check=True)
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({"success": False, "message": str(e)})

# Cam Python
@app.route('/open_camera', methods=['POST'])
def open_camera():
    try:
        data = request.get_json()
        date = data.get('date')
        classId = data.get('classId')
        image_data = data.get('image')

        # Nếu có ảnh từ web gửi lên thì giải mã và lưu tạm
        temp_img_path = None
        if image_data:
            # Giải mã base64
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            temp_img_path = os.path.join(os.path.dirname(__file__), 'temp_web_capture.jpg')
            with open(temp_img_path, 'wb') as f:
                f.write(img_bytes)

        # Gửi ảnh tới API liveness (http://127.0.0.1:3000/predict)
        liveness_result = None
        if temp_img_path and os.path.exists(temp_img_path):
            with open(temp_img_path, 'rb') as img_file:
                files = {'file': ('temp_web_capture.jpg', img_file, 'image/jpeg')}
                try:
                    liveness_response = requests.post('http://127.0.0.1:3000/predict', files=files, timeout=10)
                    if liveness_response.ok:
                        liveness_result = liveness_response.json()
                        print("Liveness result:", liveness_result)
                    else:
                        print("Liveness API error:", liveness_response.status_code)
                except Exception as ex:
                    print("Exception when calling liveness API:", ex)
        
        # Kiểm tra nếu liveness thành công trước khi gọi recognition
        if liveness_result and liveness_result.get('label') == 'Thật (Real)':
            print("Liveness PASSED. Calling recognition API...")
            
            # Gọi API recognition với cùng file ảnh
            recognition_result = None
            if temp_img_path and os.path.exists(temp_img_path):
                with open(temp_img_path, 'rb') as img_file:
                    files = {'image': ('temp_web_capture.jpg', img_file, 'image/jpeg')}
                    form_data = {
                        'date': date,
                        'classId': classId
                    }
                    try:
                        recognition_response = requests.post('http://127.0.0.1:5001/recognize', 
                                                           files=files, 
                                                           data=form_data, 
                                                           timeout=15)
                        if recognition_response.ok:
                            recognition_result = recognition_response.json()
                            print("Recognition result:", recognition_result)
                        else:
                            print("Recognition API error:", recognition_response.status_code)
                            print("Recognition API response:", recognition_response.text)
                    except Exception as ex:
                        print("Exception when calling recognition API:", ex)

        # Sau khi nhận diện xong, gọi tiếp API lấy MSSV
        mssv_result = None
        mssv_value = None
        try:
            # Đợi một chút để đảm bảo recognition API đã xử lý xong
            import time
            time.sleep(0.5)

            mssv_response = requests.get('http://127.0.0.1:5001/get_last_mssv', timeout=5)
            if mssv_response.ok:
                mssv_result = mssv_response.json()
                print("MSSV từ API /get_last_mssv:", mssv_result)
                if mssv_result.get("success") and mssv_result.get("MSSV"):
                    mssv_value = mssv_result["MSSV"]
            else:
                print("Lỗi lấy MSSV từ API /get_last_mssv:", mssv_response.status_code)
        except Exception as ex:
            print("Exception khi gọi API lấy MSSV:", ex)

        # Gửi socket nếu có MSSV
        if mssv_value:
            try:
                sio = Client()
                sio.connect('http://localhost:5000')  # Kết nối tới server socketio (cùng port Flask hoặc port riêng)
                sio.emit('response', {
                    "success": True,
                    "MSSV": mssv_value,
                    "date": date,
                    "classId": classId
                })
                print(f"Đã emit socket với MSSV: {mssv_value}, date: {date}, classId: {classId}")
                sio.disconnect()
            except Exception as ex:
                print("Lỗi khi emit socketio:", ex)

        return jsonify({"success": True, "liveness_result": liveness_result, "mssv_result": mssv_result})
    except Exception as e:
        print(f"Error opening camera: {e}")
        return jsonify({"success": False, "message": str(e)})


# # Production web
# @app.route('/process_image', methods=['POST'])
# def process_image():
#     data = request.json
#     image_data = data['image']
    
#     if not image_data.startswith('data:image/jpeg;base64,'):
#         return jsonify({"status": "error", "message": "Invalid image format"}), 400
    
#     # Chuyển đổi từ base64 về numpy array
#     img_data = base64.b64decode(image_data.split(',')[1])
#     np_arr = np.frombuffer(img_data, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     # Xử lý nhận diện khuôn mặt
#     MINSIZE = 20
#     THRESHOLD = [0.6, 0.7, 0.7]
#     FACTOR = 0.709
#     INPUT_IMAGE_SIZE = 160  
#     CLASSIFIER_PATH = '../Models/facemodel.pkl'
#     FACENET_MODEL_PATH = '../Models/20180402-114759.pb'

#     # Load The Custom Classifier
#     with open(CLASSIFIER_PATH, 'rb') as file:
#         model, class_names = pickle.load(file)
#     # print("Custom Classifier, Successfully loaded")

#     with tf.Graph().as_default():
#         # Cài đặt GPU nếu có
#         gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
#         sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

#         with sess.as_default():
#             # Load the model
#             # print('Loading feature extraction model')
#             facenet.load_model(FACENET_MODEL_PATH)

#             # Get input and output tensors
#             images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
#             embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
#             phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

#             # Phát hiện khuôn mặt
#             pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "../src/align")
#             bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

#             # if bounding_boxes.shape[0] == 0:
#             #     return jsonify({"status": "no_face_detected"}), 200

#             faces_found = bounding_boxes.shape[0]
#             # print(f"Number of faces found: {faces_found}")
#             if faces_found > 0:
#                 for i in range(faces_found):
#                     det = bounding_boxes[i]
#                     bb = np.zeros((1, 4), dtype=np.int32)
#                     bb[0][0] = int(det[0])
#                     bb[0][1] = int(det[1])
#                     bb[0][2] = int(det[2])
#                     bb[0][3] = int(det[3])

#                     # Cắt và tiền xử lý khuôn mặt
#                     cropped = frame[bb[0][1]:bb[0][3], bb[0][0]:bb[0][2], :]
#                     scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
#                     scaled = facenet.prewhiten(scaled)
#                     scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

#                     # Trích xuất đặc trưng khuôn mặt
#                     feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
#                     emb_array = sess.run(embeddings, feed_dict=feed_dict)

#                     # Nhận diện khuôn mặt
#                     predictions = model.predict_proba(emb_array)
#                     best_class_indices = np.argmax(predictions, axis=1)
#                     best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
#                     best_name = class_names[best_class_indices[0]]

#                     if best_class_probabilities > 0.8:
#                         print(f"Detected: {best_name} with probability: {best_class_probabilities}")
#                         return jsonify({"MSSV": best_name, "status": "success"})
#                     else:
#                         print("Face not recognized.")
#                         return jsonify({"status": "unknown"})

#     return jsonify({"status": "no_face_detected"})

# @socketio.on('image')
# def handle_image(data):
#     image_data = data.split(',')[1]
#     img_data = base64.b64decode(image_data)
#     np_arr = np.frombuffer(img_data, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     # Xử lý nhận diện khuôn mặt
#     bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
#     faces_found = bounding_boxes.shape[0]

#     if faces_found > 0:
#         for i in range(faces_found):
#             det = bounding_boxes[i]
#             bb = np.zeros((1, 4), dtype=np.int32)
#             bb[0][0] = int(det[0])
#             bb[0][1] = int(det[1])
#             bb[0][2] = int(det[2])
#             bb[0][3] = int(det[3])

#             cropped = frame[bb[0][1]:bb[0][3], bb[0][0]:bb[0][2], :]
#             scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
#             scaled = facenet.prewhiten(scaled)
#             scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

#             feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
#             emb_array = sess.run(embeddings, feed_dict=feed_dict)

#             predictions = model.predict_proba(emb_array)
#             best_class_indices = np.argmax(predictions, axis=1)
#             best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
#             best_name = class_names[best_class_indices[0]]

#             if best_class_probabilities > 0.8:
#                 print(f"Detected: {best_name} with probability: {best_class_probabilities}")
#                 emit('response', {"MSSV": best_name, "status": "success"})
#                 return
#             else:
#                 print("Face not recognized.")
#                 emit('response', {"status": "unknown"})
#                 return

#     emit('response', {"status": "no_face_detected"})

def timedelta_to_string(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def timedelta_to_time(td):
    if isinstance(td, timedelta):
        # Chuyển timedelta sang time
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return time(hours, minutes, seconds)
    return td  # Nếu không phải timedelta, trả về giá trị gốc

def determine_period(start_time):
    # Chuyển đổi start_time sang datetime.time nếu cần
    start_time = timedelta_to_time(start_time)

    if time(7, 0) <= start_time <= time(11, 25):
        return "Sáng"
    elif time(12, 0) <= start_time <= time(16, 25):
        return "Chiều"
    elif time(17, 0) <= start_time <= time(20, 50):
        return "Tối"
    return "Ngoài giờ"

time_slots = [
    time(7, 0), time(7, 50), time(8, 55), time(9, 45), time(10, 35),  # Sáng
    time(12, 0), time(12, 50), time(13, 55), time(14, 45), time(15, 35),  # Chiều
    time(16, 25), time(17, 15), time(18, 20), time(19, 10), time(20, 0), time(20, 50)  # Tối
]

def get_lesson_index(start_time, end_time, time_slots):
    start_period = -1
    end_period = -1

    # Tìm tiết bắt đầu
    for i in range(len(time_slots) - 1):
        if time_slots[i] <= start_time < time_slots[i + 1]:
            start_period = i + 1
            break

    # Tìm tiết kết thúc
    for i in range(len(time_slots) - 1):
        if time_slots[i] <= end_time <= time_slots[i + 1]:
            end_period = i + 1
            break

    return start_period, end_period
@app.route('/weekly_schedule', methods=['POST']) 
def weekly_schedule():
    try:
        data = request.get_json()
        mssv = data['studentId']
        start_date = data['startDate']
        end_date = data['endDate']

        # Kết nối cơ sở dữ liệu
        connection = mysql.connector.connect(host='localhost', user='root', database='face_recognition')
        cursor = connection.cursor()

        # Lấy thông tin lịch học
        cursor.execute("""
            SELECT c.name, c.id_class, a.startTime, a.endTime, a.date, a.status, c.room, t.name 
            FROM class c
            JOIN attendance a ON c.id_class = a.class_id
            JOIN teacher t ON c.teacher_id = t.MSGV
            WHERE a.student_id = %s AND a.date BETWEEN %s AND %s
        """, (mssv, start_date, end_date))

        schedule = cursor.fetchall()

        cursor.close()
        connection.close()


        # Tính tiết học
        result = []
        for cls in schedule:
            start_time = timedelta_to_time(cls[2])
            end_time = timedelta_to_time(cls[3])
            start_period, end_period = get_lesson_index(start_time, end_time, time_slots)
            result.append({
                "name": cls[0],
                "classId": cls[1],
                "startTime": start_time.strftime('%H:%M'),
                "endTime": end_time.strftime('%H:%M'),
                "startPeriod": start_period,
                "endPeriod": end_period,
                "period": determine_period(cls[2]),
                "date": cls[4].strftime('%Y-%m-%d'),
                "status": cls[5],
                "room": cls[6],
                "teacher": cls[7]
            })

        return jsonify({"success": True, "schedule": result})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    
if __name__ == '__main__':
    socketio.run(app, debug=True)
